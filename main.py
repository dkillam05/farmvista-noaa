# =====================================================================
# main.py  (FULL FILE)
# FarmVista NOAA MRMS Pass2->Pass1 fallback rainfall service
# Rev: 2026-03-24a-force-full-backfill-on-latlng-change
#
# PURPOSE
# ✅ Pulls MRMS hourly rainfall from NOAA AWS
# ✅ Writes per-field MRMS parent docs + hourly subcollection
# ✅ Manages full backfill + gap repair queue
# ✅ Detects field lat/lng changes automatically inside MRMS
# ✅ If lat/lng changed, clears stale MRMS history for that field
# ✅ If lat/lng changed, force-resets full backfill queue for that field
# ✅ Writes the current hour immediately at the new location
# ✅ Treats moved fields like new fields automatically
# ✅ FIX: moved fields now suppress repair-gap enqueue on the same run
# ✅ FIX: moved fields now clear stale repair jobs for that field
# ✅ FIX: moved fields now clear stale full-backfill job state before requeue
#
# NOTES
# - This file handles MRMS automation only.
# - Weather-cache automation is still handled elsewhere.
# - Backfill jobs always read the current field lat/lng from Firestore.
# =====================================================================

import gzip
import math
import os
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request

IMPORT_ERROR = None
try:
    import fsspec
    import numpy as np
    import xarray as xr
    import firebase_admin
    from firebase_admin import firestore
except Exception as e:
    IMPORT_ERROR = str(e)

app = Flask(__name__)

AWS_BUCKET = "noaa-mrms-pds"
DEFAULT_REGION = "CONUS"
DEFAULT_RADIUS_MILES = 0.5
MAX_BULK_POINTS = 1000

MRMS_PARENT_COLLECTION = os.environ.get("FV_MRMS_PARENT_COLLECTION", "field_mrms_weather")
FIELDS_COLLECTION = os.environ.get("FV_FIELDS_COLLECTION", "fields")
MRMS_HOURLY_SUBCOLLECTION = os.environ.get("FV_MRMS_HOURLY_SUBCOLLECTION", "mrms_hourly")
MRMS_BACKFILL_QUEUE_COLLECTION = os.environ.get("FV_MRMS_BACKFILL_QUEUE_COLLECTION", "mrms_backfill_queue")
APP_TIMEZONE = os.environ.get("FV_TIMEZONE", "America/Chicago")

KEEP_DAYS = 30
KEEP_HOURS = KEEP_DAYS * 24
LAST24_COUNT = 24

DEFAULT_BACKFILL_MAX_FIELDS_PER_RUN = int(os.environ.get("FV_BACKFILL_MAX_FIELDS_PER_RUN", "1"))
DEFAULT_BACKFILL_MAX_MINUTES_PER_RUN = float(os.environ.get("FV_BACKFILL_MAX_MINUTES_PER_RUN", "4"))
DEFAULT_REPAIR_LOOKBACK_HOURS = int(os.environ.get("FV_REPAIR_LOOKBACK_HOURS", "12"))

DEFAULT_FULL_BACKFILL_CHUNK_HOURS = int(os.environ.get("FV_FULL_BACKFILL_CHUNK_HOURS", "48"))
DEFAULT_REPAIR_CHUNK_HOURS = int(os.environ.get("FV_REPAIR_CHUNK_HOURS", "48"))

LOCATION_EPSILON = float(os.environ.get("FV_LOCATION_EPSILON", "0.00001"))

PRODUCT_PRIORITY = [
    "MultiSensor_QPE_01H_Pass2_00.00",
    "MultiSensor_QPE_01H_Pass1_00.00",
]

SAMPLE_POINTS = [
    {"key": "center",    "weight": 0.50, "dxMiles":  0.0, "dyMiles":  0.0},
    {"key": "north",     "weight": 0.10, "dxMiles":  0.0, "dyMiles":  1.0},
    {"key": "south",     "weight": 0.10, "dxMiles":  0.0, "dyMiles": -1.0},
    {"key": "east",      "weight": 0.10, "dxMiles":  1.0, "dyMiles":  0.0},
    {"key": "west",      "weight": 0.10, "dxMiles": -1.0, "dyMiles":  0.0},
    {"key": "northeast", "weight": 0.10, "dxMiles":  1.0, "dyMiles":  1.0},
]

CACHE_LOCK = threading.Lock()
CACHE = {
    "selectedKey": None,
    "selectedProduct": None,
    "fileTimestampUtc": None,
    "variableName": None,
    "dataArray": None,
    "io": None,
}

_DB = None


def num(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except Exception:
        return None


def clamp(n, lo, hi):
    n = num(n)
    if n is None:
        return lo
    return max(lo, min(hi, n))


def round_num(value, digits=4):
    if value is None or not math.isfinite(value):
        return None
    p = 10 ** digits
    return round(value * p) / p


def miles_to_lat_degrees(miles):
    return miles / 69.0


def miles_to_lon_degrees(miles, lat_deg):
    cos_lat = math.cos(math.radians(lat_deg))
    if abs(cos_lat) < 1e-9:
        return 0.0
    return miles / (69.172 * cos_lat)


def offset_point(lat, lon, east_miles, north_miles):
    return (
        lat + miles_to_lat_degrees(north_miles),
        lon + miles_to_lon_degrees(east_miles, lat),
    )


def ensure_runtime_ready():
    if IMPORT_ERROR:
        raise RuntimeError(
            "Python package import failed. "
            f"Original error: {IMPORT_ERROR}"
        )


def get_fs():
    ensure_runtime_ready()
    return fsspec.filesystem("s3", anon=True)


def get_db():
    global _DB
    ensure_runtime_ready()

    if _DB is not None:
        return _DB

    if not firebase_admin._apps:
        firebase_admin.initialize_app()

    _DB = firestore.client()
    return _DB


def get_app_tz():
    try:
        return ZoneInfo(APP_TIMEZONE)
    except Exception:
        return timezone.utc


def list_candidate_dates(now_utc, days_back=2):
    out = []
    for i in range(days_back + 1):
        out.append((now_utc - timedelta(days=i)).strftime("%Y%m%d"))
    return out


def parse_timestamp_from_key(key):
    try:
        filename = key.split("/")[-1]
        stamp = filename.replace(".grib2.gz", "").split("_")[-1]
        return datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def parse_iso_utc(iso_str):
    return datetime.fromisoformat(str(iso_str).replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_utc(dt):
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def floor_to_hour_utc(dt):
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def hour_doc_id_from_iso(file_timestamp_utc):
    return file_timestamp_utc.replace("-", "").replace(":", "")


def full_backfill_job_id(field_id):
    return f"full__{field_id}"


def repair_job_id(field_id, start_hour_utc, end_hour_utc):
    return f"repair__{field_id}__{hour_doc_id_from_iso(start_hour_utc)}__{hour_doc_id_from_iso(end_hour_utc)}"


def list_latest_key(region, product, now_utc, max_age_hours=12):
    fs = get_fs()
    all_keys = []

    for ymd in list_candidate_dates(now_utc, days_back=2):
        prefix = f"{AWS_BUCKET}/{region}/{product}/{ymd}/"
        try:
            keys = fs.ls(prefix, detail=False)
            all_keys.extend(keys)
        except Exception:
            continue

    if not all_keys:
        return None

    grib_keys = [k for k in all_keys if k.endswith(".grib2.gz")]
    if not grib_keys:
        return None

    dated = []
    for k in grib_keys:
        ts = parse_timestamp_from_key(k)
        if ts is not None:
            dated.append((ts, k))

    if not dated:
        grib_keys.sort()
        return grib_keys[-1]

    recent = []
    for ts, k in dated:
        age_hours = abs((now_utc - ts).total_seconds()) / 3600.0
        if age_hours <= max_age_hours:
            recent.append((ts, k))

    pool = recent if recent else dated
    pool.sort(key=lambda x: x[0])
    return pool[-1][1]


def list_key_for_exact_hour(region, product, target_dt_utc):
    fs = get_fs()
    ymd = target_dt_utc.strftime("%Y%m%d")
    stamp = target_dt_utc.strftime("%Y%m%d-%H0000")
    prefix = f"{AWS_BUCKET}/{region}/{product}/{ymd}/"

    try:
        keys = fs.ls(prefix, detail=False)
    except Exception:
        return None

    matches = []
    for k in keys:
        if not k.endswith(".grib2.gz"):
            continue
        if stamp in k:
            matches.append(k)

    if not matches:
        return None
    matches.sort()
    return matches[-1]


def choose_best_product_for_latest(region, now_utc):
    for product in PRODUCT_PRIORITY:
        key = list_latest_key(region, product, now_utc)
        if key:
            return product, key
    return None, None


def choose_best_product_for_hour(region, target_dt_utc):
    for product in PRODUCT_PRIORITY:
        key = list_key_for_exact_hour(region, product, target_dt_utc)
        if key:
            return product, key
    return None, None


def gunzip_to_tempfile(fs, key):
    with fs.open(key, "rb") as f:
        raw = f.read()

    with gzip.GzipFile(fileobj=tempfile.SpooledTemporaryFile()) as _:
        pass

    data = gzip.decompress(raw)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".grib2")
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def load_dataset_for_key(key):
    ensure_runtime_ready()
    fs = get_fs()
    local_path = gunzip_to_tempfile(fs, key)
    try:
        ds = xr.open_dataset(local_path, engine="cfgrib")
        return ds
    except Exception:
        try:
            ds = xr.open_dataset(local_path, engine="pynio")
            return ds
        except Exception:
            raise
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass


def prepare_cache_for_key(key, product):
    ts = parse_timestamp_from_key(key)
    with CACHE_LOCK:
        if CACHE.get("selectedKey") == key and CACHE.get("dataArray") is not None:
            return

    ds = load_dataset_for_key(key)

    variable_name = None
    for candidate in ["unknown", "tp", "precip", "precipitation", "param18.0.0"]:
        if candidate in ds.data_vars:
            variable_name = candidate
            break
    if variable_name is None:
        data_vars = list(ds.data_vars.keys())
        if not data_vars:
            raise RuntimeError("No data variables found in MRMS GRIB.")
        variable_name = data_vars[0]

    da = ds[variable_name]

    with CACHE_LOCK:
        old_ds = CACHE.get("io")
        CACHE["selectedKey"] = key
        CACHE["selectedProduct"] = product
        CACHE["fileTimestampUtc"] = iso_utc(ts) if ts else None
        CACHE["variableName"] = variable_name
        CACHE["dataArray"] = da
        CACHE["io"] = ds

    try:
        if old_ds is not None and old_ds is not ds:
            old_ds.close()
    except Exception:
        pass


def get_cache_da():
    with CACHE_LOCK:
        da = CACHE.get("dataArray")
        if da is None:
            raise RuntimeError("MRMS cache is empty.")
        return da


def get_cache_meta():
    with CACHE_LOCK:
        return {
            "selectedKey": CACHE.get("selectedKey"),
            "selectedProduct": CACHE.get("selectedProduct"),
            "fileTimestampUtc": CACHE.get("fileTimestampUtc"),
            "variableName": CACHE.get("variableName"),
        }


def latlon_name_candidates(da):
    dims = list(da.dims)
    coords = list(da.coords)
    lat_names = [n for n in ["latitude", "lat", "y"] if n in dims or n in coords]
    lon_names = [n for n in ["longitude", "lon", "x"] if n in dims or n in coords]
    if not lat_names or not lon_names:
        raise RuntimeError(f"Could not identify lat/lon axes. dims={dims}, coords={coords}")
    return lat_names[0], lon_names[0]


def sample_one_point(da, lat, lon):
    lat_name, lon_name = latlon_name_candidates(da)
    selected = da.sel({lat_name: lat, lon_name: lon}, method="nearest")
    value = float(selected.values)
    return value


def inches_from_dataset_value(value):
    if value is None or not math.isfinite(value):
        return None
    if value < 0:
        value = 0.0
    return value / 25.4


def compute_weighted_rain(lat, lon, radius_miles):
    da = get_cache_da()

    total_weight = 0.0
    weighted_sum = 0.0
    samples_out = []

    for sp in SAMPLE_POINTS:
        p_lat, p_lon = offset_point(
            lat, lon,
            east_miles=sp["dxMiles"] * radius_miles,
            north_miles=sp["dyMiles"] * radius_miles,
        )
        raw_mm = sample_one_point(da, p_lat, p_lon)
        rain_in = inches_from_dataset_value(raw_mm)
        if rain_in is None:
            rain_in = 0.0
        weighted_sum += rain_in * sp["weight"]
        total_weight += sp["weight"]
        samples_out.append({
            "key": sp["key"],
            "lat": round_num(p_lat, 6),
            "lon": round_num(p_lon, 6),
            "weight": sp["weight"],
            "rainIn": round_num(rain_in, 4),
        })

    final_in = weighted_sum / total_weight if total_weight > 0 else 0.0
    if final_in < 0:
        final_in = 0.0

    return round_num(final_in, 4), samples_out


def collection_parent():
    return get_db().collection(MRMS_PARENT_COLLECTION)


def field_doc_ref(field_id):
    return collection_parent().document(str(field_id))


def hourly_collection_ref(field_id):
    return field_doc_ref(field_id).collection(MRMS_HOURLY_SUBCOLLECTION)


def backfill_queue_ref():
    return get_db().collection(MRMS_BACKFILL_QUEUE_COLLECTION)


def stream_active_fields():
    docs = (
        get_db()
        .collection(FIELDS_COLLECTION)
        .where("archived", "==", False)
        .stream()
    )
    for doc in docs:
        data = doc.to_dict() or {}
        data["id"] = doc.id
        yield data


def field_lat_lng(field):
    loc = field.get("location") or {}
    lat = num(loc.get("lat"))
    lng = num(loc.get("lng"))
    if lat is None or lng is None:
        lat = num(field.get("lat"))
        lng = num(field.get("lng"))
    return lat, lng


def location_changed(prev_lat, prev_lng, new_lat, new_lng):
    if prev_lat is None or prev_lng is None or new_lat is None or new_lng is None:
        return False
    return (
        abs(float(prev_lat) - float(new_lat)) > LOCATION_EPSILON or
        abs(float(prev_lng) - float(new_lng)) > LOCATION_EPSILON
    )


def parent_snapshot(field_id):
    snap = field_doc_ref(field_id).get()
    return snap.to_dict() if snap.exists else None


def get_field_mrms_state(field_id):
    parent = parent_snapshot(field_id) or {}
    backfill = parent.get("backfill") or {}
    latest_file_ts = parent.get("latestFileTimestampUtc")
    is_new = not bool(latest_file_ts)

    last24 = parent.get("mrmsHourlyLast24") or []
    daily30 = parent.get("mrmsDailySeries30d") or []

    latest_hour_doc = None
    try:
        q = (
            hourly_collection_ref(field_id)
            .order_by("fileTimestampUtc", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for d in q:
            latest_hour_doc = d.to_dict() or {}
            break
    except Exception:
        latest_hour_doc = None

    return {
        "isNewField": is_new,
        "latestFileTimestampUtc": latest_file_ts,
        "latestHourDoc": latest_hour_doc,
        "last24Count": len(last24),
        "daily30Count": len(daily30),
        "backfillStatus": backfill.get("status"),
    }


def build_hour_payload(field, lat, lng, rain_in, samples, file_timestamp_utc, region, radius_miles, product):
    now_utc = datetime.now(timezone.utc)
    file_dt = parse_iso_utc(file_timestamp_utc)
    app_tz = get_app_tz()

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "farmId": field.get("farmId"),
        "farmName": field.get("farmName"),
        "county": field.get("county"),
        "state": field.get("state"),
        "lat": round_num(lat, 6),
        "lng": round_num(lng, 6),
        "radiusMiles": round_num(radius_miles, 2),
        "region": region,
        "rainIn": round_num(rain_in, 4),
        "samplePoints": samples,
        "source": "NOAA_AWS_MRMS",
        "product": product,
        "fileTimestampUtc": file_timestamp_utc,
        "dateISO": file_dt.astimezone(app_tz).strftime("%Y-%m-%d"),
        "hourLocal": int(file_dt.astimezone(app_tz).strftime("%H")),
        "computedAtUtc": iso_utc(now_utc),
    }


def day_bucket_map_from_daily30(daily_series):
    out = {}
    for row in (daily_series or []):
        if not isinstance(row, dict):
            continue
        date_iso = row.get("dateISO")
        if not date_iso:
            continue
        out[str(date_iso)] = {
            "dateISO": str(date_iso),
            "rainIn": round_num(num(row.get("rainIn")) or 0.0, 4),
        }
    return out


def normalize_hour_row(row):
    if not isinstance(row, dict):
        return None
    ts = row.get("fileTimestampUtc")
    rain_in = num(row.get("rainIn"))
    if not ts or rain_in is None:
        return None
    return {
        "fileTimestampUtc": str(ts),
        "rainIn": round_num(rain_in, 4),
        "dateISO": row.get("dateISO"),
        "hourLocal": row.get("hourLocal"),
        "product": row.get("product"),
    }


def build_incremental_parent_update(parent, hour_payload):
    parent = parent or {}
    app_tz = get_app_tz()

    new_hour = normalize_hour_row(hour_payload)
    if not new_hour:
        raise RuntimeError("Invalid hour payload for incremental parent update.")

    new_hour_dt_utc = parse_iso_utc(new_hour["fileTimestampUtc"])
    new_date_iso = new_hour_dt_utc.astimezone(app_tz).strftime("%Y-%m-%d")
    new_hour["dateISO"] = new_date_iso
    new_hour["hourLocal"] = int(new_hour_dt_utc.astimezone(app_tz).strftime("%H"))

    cutoff_hour_utc = new_hour_dt_utc - timedelta(hours=KEEP_HOURS - 1)
    cutoff_hour_iso = iso_utc(cutoff_hour_utc)

    # Incremental last-24 buffer.
    existing_last24 = []
    for row in (parent.get("mrmsHourlyLast24") or []):
        n = normalize_hour_row(row)
        if not n:
            continue
        if n["fileTimestampUtc"] < cutoff_hour_iso:
            continue
        existing_last24.append(n)

    last24_map = {row["fileTimestampUtc"]: row for row in existing_last24}
    last24_map[new_hour["fileTimestampUtc"]] = new_hour
    last24_rows = list(last24_map.values())
    last24_rows.sort(key=lambda x: x["fileTimestampUtc"])
    last24_rows = last24_rows[-LAST24_COUNT:]

    # Incremental 30-day daily series.
    daily_map = day_bucket_map_from_daily30(parent.get("mrmsDailySeries30d") or [])
    day_row = daily_map.get(new_date_iso) or {"dateISO": new_date_iso, "rainIn": 0.0}
    existing_day_total = num(day_row.get("rainIn")) or 0.0

    old_same_hour = None
    for row in existing_last24:
        if row["fileTimestampUtc"] == new_hour["fileTimestampUtc"]:
            old_same_hour = row
            break

    # If the replaced hour exists in last24, subtract old amount from same day total.
    # This avoids double-counting when the same hour is rewritten.
    if old_same_hour and str(old_same_hour.get("dateISO")) == new_date_iso:
        existing_day_total -= (num(old_same_hour.get("rainIn")) or 0.0)

    existing_day_total += (num(new_hour.get("rainIn")) or 0.0)
    if existing_day_total < 0:
        existing_day_total = 0.0
    daily_map[new_date_iso] = {
        "dateISO": new_date_iso,
        "rainIn": round_num(existing_day_total, 4),
    }

    # Trim daily buckets to keep only last 30 local dates relative to new hour.
    keep_dates = set()
    for i in range(KEEP_DAYS):
        keep_dates.add((new_hour_dt_utc.astimezone(app_tz) - timedelta(days=i)).strftime("%Y-%m-%d"))

    trimmed_daily = []
    for date_iso, row in daily_map.items():
        if date_iso in keep_dates:
            trimmed_daily.append({
                "dateISO": date_iso,
                "rainIn": round_num(num(row.get("rainIn")) or 0.0, 4),
            })
    trimmed_daily.sort(key=lambda x: x["dateISO"])
    trimmed_daily = trimmed_daily[-KEEP_DAYS:]

    rain_24h = 0.0
    for row in last24_rows:
        rain_24h += (num(row.get("rainIn")) or 0.0)

    update = {
        "fieldId": hour_payload.get("fieldId"),
        "fieldName": hour_payload.get("fieldName"),
        "farmId": hour_payload.get("farmId"),
        "farmName": hour_payload.get("farmName"),
        "county": hour_payload.get("county"),
        "state": hour_payload.get("state"),
        "lat": hour_payload.get("lat"),
        "lng": hour_payload.get("lng"),
        "radiusMiles": hour_payload.get("radiusMiles"),
        "region": hour_payload.get("region"),
        "source": hour_payload.get("source"),
        "product": hour_payload.get("product"),
        "latestFileTimestampUtc": hour_payload.get("fileTimestampUtc"),
        "latestRainIn": hour_payload.get("rainIn"),
        "latestComputedAtUtc": hour_payload.get("computedAtUtc"),
        "latestSamplePoints": hour_payload.get("samplePoints"),
        "mrmsHourlyLast24": last24_rows,
        "mrmsDailySeries30d": trimmed_daily,
        "mrmsRainLast24h": round_num(rain_24h, 4),
        "lastIncrementalUpdateUtc": iso_utc(datetime.now(timezone.utc)),
    }
    return update


def maybe_prune_old_hourly_docs(field_id, newest_hour_iso):
    newest_dt = parse_iso_utc(newest_hour_iso)
    cutoff_dt = newest_dt - timedelta(hours=KEEP_HOURS)
    cutoff_iso = iso_utc(cutoff_dt)

    # Throttle pruning so we do not scan old docs every hour for every field.
    # Only do cleanup around midnight local or every 24th hour.
    app_tz = get_app_tz()
    local_hour = int(newest_dt.astimezone(app_tz).strftime("%H"))
    if local_hour not in (0, 1):
        return {"deleted": 0, "throttled": True}

    deleted = 0
    batch = get_db().batch()
    batch_count = 0

    docs = (
        hourly_collection_ref(field_id)
        .where("fileTimestampUtc", "<", cutoff_iso)
        .limit(500)
        .stream()
    )
    for doc in docs:
        batch.delete(doc.reference)
        batch_count += 1
        deleted += 1
        if batch_count >= 400:
            batch.commit()
            batch = get_db().batch()
            batch_count = 0

    if batch_count > 0:
        batch.commit()

    return {"deleted": deleted, "throttled": False}


def clear_hourly_history(field_id):
    deleted = 0
    while True:
        docs = list(hourly_collection_ref(field_id).limit(400).stream())
        if not docs:
            break
        batch = get_db().batch()
        for d in docs:
            batch.delete(d.reference)
            deleted += 1
        batch.commit()
        if len(docs) < 400:
            break
    return deleted


def clear_repair_jobs_for_field(field_id):
    deleted = 0
    docs = (
        backfill_queue_ref()
        .where("fieldId", "==", str(field_id))
        .where("jobType", "==", "repair")
        .stream()
    )
    batch = get_db().batch()
    count = 0
    for doc in docs:
        batch.delete(doc.reference)
        deleted += 1
        count += 1
        if count >= 400:
            batch.commit()
            batch = get_db().batch()
            count = 0
    if count > 0:
        batch.commit()
    return deleted


def reset_full_backfill_job(field_id):
    job_ref = backfill_queue_ref().document(full_backfill_job_id(field_id))
    snap = job_ref.get()
    if snap.exists:
        job_ref.delete()
        return True
    return False


def clear_parent_for_location_change(field_id):
    field_doc_ref(field_id).set({
        "locationResetAtUtc": iso_utc(datetime.now(timezone.utc)),
        "latestFileTimestampUtc": None,
        "latestRainIn": None,
        "latestComputedAtUtc": None,
        "latestSamplePoints": [],
        "mrmsHourlyLast24": [],
        "mrmsDailySeries30d": [],
        "mrmsRainLast24h": 0,
        "backfill": {
            "status": "needed",
            "reason": "latLngChanged",
            "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
        },
    }, merge=True)


def enqueue_full_backfill(field_id, reason="newField"):
    now = iso_utc(datetime.now(timezone.utc))
    job_ref = backfill_queue_ref().document(full_backfill_job_id(field_id))
    payload = {
        "fieldId": str(field_id),
        "jobType": "full",
        "status": "queued",
        "queuedAtUtc": now,
        "updatedAtUtc": now,
        "reason": reason,
        "attempts": 0,
    }
    job_ref.set(payload, merge=True)

    field_doc_ref(field_id).set({
        "backfill": {
            "status": "queued",
            "reason": reason,
            "updatedAtUtc": now,
        }
    }, merge=True)

    return payload


def enqueue_repair_job(field_id, start_hour_utc, end_hour_utc, reason="gapDetected"):
    now = iso_utc(datetime.now(timezone.utc))
    job_id = repair_job_id(field_id, iso_utc(start_hour_utc), iso_utc(end_hour_utc))
    job_ref = backfill_queue_ref().document(job_id)
    payload = {
        "fieldId": str(field_id),
        "jobType": "repair",
        "status": "queued",
        "queuedAtUtc": now,
        "updatedAtUtc": now,
        "reason": reason,
        "attempts": 0,
        "startHourUtc": iso_utc(start_hour_utc),
        "endHourUtc": iso_utc(end_hour_utc),
    }
    job_ref.set(payload, merge=True)
    return payload


def find_missing_recent_hours(field_id, now_utc=None, lookback_hours=DEFAULT_REPAIR_LOOKBACK_HOURS):
    now_utc = now_utc or datetime.now(timezone.utc)
    end_hour = floor_to_hour_utc(now_utc) - timedelta(hours=1)
    start_hour = end_hour - timedelta(hours=max(1, lookback_hours) - 1)

    existing = set()
    docs = (
        hourly_collection_ref(field_id)
        .where("fileTimestampUtc", ">=", iso_utc(start_hour))
        .where("fileTimestampUtc", "<=", iso_utc(end_hour))
        .stream()
    )
    for doc in docs:
        row = doc.to_dict() or {}
        ts = row.get("fileTimestampUtc")
        if ts:
            existing.add(str(ts))

    missing = []
    cursor = start_hour
    while cursor <= end_hour:
        ts = iso_utc(cursor)
        if ts not in existing:
            missing.append(cursor)
        cursor += timedelta(hours=1)
    return missing


def collapse_hours_to_ranges(hours):
    if not hours:
        return []
    hours = sorted(hours)
    ranges = []
    start = hours[0]
    prev = hours[0]
    for h in hours[1:]:
        if h == prev + timedelta(hours=1):
            prev = h
            continue
        ranges.append((start, prev))
        start = h
        prev = h
    ranges.append((start, prev))
    return ranges


def maybe_auto_enqueue_gap_repair(field_id, now_utc=None, lookback_hours=DEFAULT_REPAIR_LOOKBACK_HOURS):
    now_utc = now_utc or datetime.now(timezone.utc)
    missing = find_missing_recent_hours(field_id, now_utc=now_utc, lookback_hours=lookback_hours)
    if not missing:
        return {"queued": 0, "ranges": []}

    ranges = collapse_hours_to_ranges(missing)
    queued = 0
    out_ranges = []
    for start_hour, end_hour in ranges:
        enqueue_repair_job(field_id, start_hour, end_hour, reason="gapDetected")
        queued += 1
        out_ranges.append({
            "startHourUtc": iso_utc(start_hour),
            "endHourUtc": iso_utc(end_hour),
        })
    return {"queued": queued, "ranges": out_ranges}


def rebuild_last24_and_daily30(field_id):
    app_tz = get_app_tz()
    now_utc = datetime.now(timezone.utc)
    cutoff_hour_utc = now_utc - timedelta(hours=KEEP_HOURS)
    cutoff_iso = iso_utc(cutoff_hour_utc)

    # Delete very old hourly docs beyond retention.
    old_docs = list(
        hourly_collection_ref(field_id)
        .where("fileTimestampUtc", "<", cutoff_iso)
        .stream()
    )
    if old_docs:
        batch = get_db().batch()
        c = 0
        for doc in old_docs:
            batch.delete(doc.reference)
            c += 1
            if c >= 400:
                batch.commit()
                batch = get_db().batch()
                c = 0
        if c > 0:
            batch.commit()

    docs = list(
        hourly_collection_ref(field_id)
        .where("fileTimestampUtc", ">=", cutoff_iso)
        .stream()
    )

    rows = []
    for doc in docs:
        row = doc.to_dict() or {}
        ts = row.get("fileTimestampUtc")
        rain_in = num(row.get("rainIn"))
        if not ts or rain_in is None:
            continue
        rows.append({
            "fileTimestampUtc": str(ts),
            "rainIn": round_num(rain_in, 4),
        })

    rows.sort(key=lambda x: x["fileTimestampUtc"])
    last24 = rows[-LAST24_COUNT:]

    by_day = {}
    for row in rows:
        dt_utc = parse_iso_utc(row["fileTimestampUtc"])
        date_iso = dt_utc.astimezone(app_tz).strftime("%Y-%m-%d")
        by_day[date_iso] = round_num((by_day.get(date_iso, 0.0) + (num(row["rainIn"]) or 0.0)), 4)

    daily30 = [{"dateISO": d, "rainIn": round_num(v, 4)} for d, v in sorted(by_day.items())]
    daily30 = daily30[-KEEP_DAYS:]

    rain_24h = 0.0
    for row in last24:
        rain_24h += (num(row.get("rainIn")) or 0.0)

    return {
        "mrmsHourlyLast24": last24,
        "mrmsDailySeries30d": daily30,
        "mrmsRainLast24h": round_num(rain_24h, 4),
        "rebuiltAtUtc": iso_utc(datetime.now(timezone.utc)),
    }


def write_field_hour(field, lat, lng, rain_in, samples, file_timestamp_utc, region, radius_miles, product):
    payload = build_hour_payload(field, lat, lng, rain_in, samples, file_timestamp_utc, region, radius_miles, product)
    hour_id = hour_doc_id_from_iso(file_timestamp_utc)

    parent_ref = field_doc_ref(field["id"])
    hour_ref = hourly_collection_ref(field["id"]).document(hour_id)

    parent = parent_snapshot(field["id"]) or {}

    hour_ref.set(payload, merge=True)

    # INCREMENTAL UPDATE:
    # Existing fields should not reread the entire 30-day history every hour.
    parent_update = build_incremental_parent_update(parent, payload)

    parent_ref.set(parent_update, merge=True)

    prune_info = maybe_prune_old_hourly_docs(field["id"], file_timestamp_utc)

    return {
        "hourDocId": hour_id,
        "parentUpdated": True,
        "prune": prune_info,
        "payload": payload,
    }


def finalize_field_parent_from_hourly(field_id, extra_merge=None):
    rebuilt = rebuild_last24_and_daily30(field_id)
    data = dict(rebuilt)
    if extra_merge:
        data.update(extra_merge)
    field_doc_ref(field_id).set(data, merge=True)
    return data


def load_field_for_job(field_id):
    snap = get_db().collection(FIELDS_COLLECTION).document(str(field_id)).get()
    if not snap.exists:
        return None
    data = snap.to_dict() or {}
    data["id"] = snap.id
    return data


def fetch_hour_for_field(field, target_hour_utc, region=DEFAULT_REGION, radius_miles=DEFAULT_RADIUS_MILES):
    lat, lng = field_lat_lng(field)
    if lat is None or lng is None:
        raise RuntimeError(f"Field {field.get('id')} missing lat/lng.")

    product, key = choose_best_product_for_hour(region, target_hour_utc)
    if not product or not key:
        raise RuntimeError(f"No MRMS key found for target hour {iso_utc(target_hour_utc)}")

    prepare_cache_for_key(key, product)
    meta = get_cache_meta()
    rain_in, samples = compute_weighted_rain(lat, lng, radius_miles)

    payload = write_field_hour(
        field=field,
        lat=lat,
        lng=lng,
        rain_in=rain_in,
        samples=samples,
        file_timestamp_utc=meta["fileTimestampUtc"],
        region=region,
        radius_miles=radius_miles,
        product=meta["selectedProduct"],
    )
    return payload


def process_full_backfill_job(job_doc, chunk_hours=DEFAULT_FULL_BACKFILL_CHUNK_HOURS):
    job = job_doc.to_dict() or {}
    field_id = job.get("fieldId")
    if not field_id:
        raise RuntimeError("Backfill job missing fieldId.")

    field = load_field_for_job(field_id)
    if not field:
        job_doc.reference.set({
            "status": "failed",
            "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
            "error": "Field not found.",
        }, merge=True)
        return {"status": "failed", "reason": "fieldNotFound"}

    now_utc = datetime.now(timezone.utc)
    latest_complete_hour = floor_to_hour_utc(now_utc) - timedelta(hours=1)
    start_hour = latest_complete_hour - timedelta(hours=KEEP_HOURS - 1)

    cursor_iso = job.get("cursorHourUtc")
    cursor = parse_iso_utc(cursor_iso) if cursor_iso else start_hour
    end_hour = min(cursor + timedelta(hours=max(1, chunk_hours) - 1), latest_complete_hour)

    job_doc.reference.set({
        "status": "running",
        "updatedAtUtc": iso_utc(now_utc),
        "startedAtUtc": job.get("startedAtUtc") or iso_utc(now_utc),
        "cursorHourUtc": iso_utc(cursor),
    }, merge=True)

    processed = 0
    write_errors = []

    h = cursor
    while h <= end_hour:
        try:
            fetch_hour_for_field(field, h)
            processed += 1
        except Exception as e:
            write_errors.append({
                "hourUtc": iso_utc(h),
                "error": str(e),
            })
        h += timedelta(hours=1)

    done = end_hour >= latest_complete_hour

    parent_extra = {
        "backfill": {
            "status": "complete" if done else "running",
            "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
            "reason": job.get("reason") or "fullBackfill",
        }
    }
    finalize_field_parent_from_hourly(field_id, extra_merge=parent_extra)

    job_update = {
        "status": "done" if done else "queued",
        "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
        "lastChunkStartHourUtc": iso_utc(cursor),
        "lastChunkEndHourUtc": iso_utc(end_hour),
        "lastProcessedHours": processed,
        "cursorHourUtc": iso_utc(end_hour + timedelta(hours=1)),
        "errors": write_errors[-25:],
        "attempts": int(job.get("attempts") or 0) + 1,
    }
    if done:
        job_update["completedAtUtc"] = iso_utc(datetime.now(timezone.utc))
    job_doc.reference.set(job_update, merge=True)

    return {
        "status": "done" if done else "queued",
        "processedHours": processed,
        "errors": write_errors,
        "fieldId": field_id,
        "chunkStartHourUtc": iso_utc(cursor),
        "chunkEndHourUtc": iso_utc(end_hour),
    }


def process_repair_job(job_doc, chunk_hours=DEFAULT_REPAIR_CHUNK_HOURS):
    job = job_doc.to_dict() or {}
    field_id = job.get("fieldId")
    if not field_id:
        raise RuntimeError("Repair job missing fieldId.")

    start_hour = parse_iso_utc(job.get("startHourUtc"))
    end_hour = parse_iso_utc(job.get("endHourUtc"))
    cursor_iso = job.get("cursorHourUtc")
    cursor = parse_iso_utc(cursor_iso) if cursor_iso else start_hour
    chunk_end = min(cursor + timedelta(hours=max(1, chunk_hours) - 1), end_hour)

    field = load_field_for_job(field_id)
    if not field:
        job_doc.reference.set({
            "status": "failed",
            "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
            "error": "Field not found.",
        }, merge=True)
        return {"status": "failed", "reason": "fieldNotFound"}

    job_doc.reference.set({
        "status": "running",
        "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
        "startedAtUtc": job.get("startedAtUtc") or iso_utc(datetime.now(timezone.utc)),
        "cursorHourUtc": iso_utc(cursor),
    }, merge=True)

    processed = 0
    write_errors = []
    h = cursor
    while h <= chunk_end:
        try:
            fetch_hour_for_field(field, h)
            processed += 1
        except Exception as e:
            write_errors.append({
                "hourUtc": iso_utc(h),
                "error": str(e),
            })
        h += timedelta(hours=1)

    done = chunk_end >= end_hour

    parent_extra = {
        "backfill": {
            "status": "complete",
            "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
            "reason": "repairCompleted",
        }
    }
    finalize_field_parent_from_hourly(field_id, extra_merge=parent_extra)

    job_update = {
        "status": "done" if done else "queued",
        "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
        "lastChunkStartHourUtc": iso_utc(cursor),
        "lastChunkEndHourUtc": iso_utc(chunk_end),
        "lastProcessedHours": processed,
        "cursorHourUtc": iso_utc(chunk_end + timedelta(hours=1)),
        "errors": write_errors[-25:],
        "attempts": int(job.get("attempts") or 0) + 1,
    }
    if done:
        job_update["completedAtUtc"] = iso_utc(datetime.now(timezone.utc))
    job_doc.reference.set(job_update, merge=True)

    return {
        "status": "done" if done else "queued",
        "processedHours": processed,
        "errors": write_errors,
        "fieldId": field_id,
        "chunkStartHourUtc": iso_utc(cursor),
        "chunkEndHourUtc": iso_utc(chunk_end),
    }


def run_backfill_queue(max_fields=DEFAULT_BACKFILL_MAX_FIELDS_PER_RUN, max_minutes=DEFAULT_BACKFILL_MAX_MINUTES_PER_RUN):
    started = time.time()
    processed_jobs = []

    docs = (
        backfill_queue_ref()
        .where("status", "in", ["queued", "running"])
        .limit(max(1, max_fields) * 5)
        .stream()
    )

    for doc in docs:
        if (time.time() - started) / 60.0 >= max_minutes:
            break
        job = doc.to_dict() or {}
        job_type = job.get("jobType")
        try:
            if job_type == "full":
                result = process_full_backfill_job(doc)
            elif job_type == "repair":
                result = process_repair_job(doc)
            else:
                result = {"status": "skipped", "reason": f"unknownJobType:{job_type}"}
            processed_jobs.append({"id": doc.id, **result})
        except Exception as e:
            doc.reference.set({
                "status": "failed",
                "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
                "error": str(e),
                "trace": traceback.format_exc()[-8000:],
            }, merge=True)
            processed_jobs.append({
                "id": doc.id,
                "status": "failed",
                "error": str(e),
            })

        if len(processed_jobs) >= max_fields:
            break

    return {
        "processedJobs": processed_jobs,
        "count": len(processed_jobs),
        "elapsedSeconds": round_num(time.time() - started, 2),
    }


def run_batch_cache(region=DEFAULT_REGION, radius_miles=DEFAULT_RADIUS_MILES):
    ensure_runtime_ready()
    now_utc = datetime.now(timezone.utc)

    product, key = choose_best_product_for_latest(region, now_utc)
    if not product or not key:
        raise RuntimeError("Could not find a recent MRMS product key.")

    prepare_cache_for_key(key, product)
    meta = get_cache_meta()
    file_timestamp_utc = meta["fileTimestampUtc"]

    out = {
        "selectedProduct": meta["selectedProduct"],
        "fileTimestampUtc": meta["fileTimestampUtc"],
        "fieldsProcessed": 0,
        "fieldsSkippedNoLocation": 0,
        "fieldsQueuedNew": 0,
        "fieldsLatLngChanged": 0,
        "fieldsGapRepairQueued": 0,
        "details": [],
    }

    for field in stream_active_fields():
        lat, lng = field_lat_lng(field)
        if lat is None or lng is None:
            out["fieldsSkippedNoLocation"] += 1
            out["details"].append({
                "fieldId": field["id"],
                "fieldName": field.get("name"),
                "status": "skipped",
                "reason": "missingLatLng",
            })
            continue

        moved = False
        parent = parent_snapshot(field["id"]) or {}
        prev_lat = num(parent.get("lat"))
        prev_lng = num(parent.get("lng"))

        if parent and location_changed(prev_lat, prev_lng, lat, lng):
            moved = True
            deleted_hourly = clear_hourly_history(field["id"])
            deleted_repairs = clear_repair_jobs_for_field(field["id"])
            reset_full_backfill_job(field["id"])
            clear_parent_for_location_change(field["id"])
            enqueue_full_backfill(field["id"], reason="latLngChanged")
            out["fieldsLatLngChanged"] += 1

            out["details"].append({
                "fieldId": field["id"],
                "fieldName": field.get("name"),
                "status": "latLngChanged",
                "deletedHourlyDocs": deleted_hourly,
                "deletedRepairJobs": deleted_repairs,
            })

        state = get_field_mrms_state(field["id"])
        if state["isNewField"]:
            enqueue_full_backfill(field["id"], reason="newField")
            out["fieldsQueuedNew"] += 1

        rain_in, samples = compute_weighted_rain(lat, lng, radius_miles)
        write_result = write_field_hour(
            field=field,
            lat=lat,
            lng=lng,
            rain_in=rain_in,
            samples=samples,
            file_timestamp_utc=file_timestamp_utc,
            region=region,
            radius_miles=radius_miles,
            product=meta["selectedProduct"],
        )

        queued_gap = {"queued": 0, "ranges": []}
        if not moved:
            queued_gap = maybe_auto_enqueue_gap_repair(field["id"], now_utc=now_utc)
            out["fieldsGapRepairQueued"] += int(queued_gap.get("queued") or 0)

        out["fieldsProcessed"] += 1
        out["details"].append({
            "fieldId": field["id"],
            "fieldName": field.get("name"),
            "status": "ok",
            "rainIn": rain_in,
            "hourDocId": write_result["hourDocId"],
            "gapRepairsQueued": queued_gap.get("queued", 0),
            "prune": write_result.get("prune"),
        })

    return out


@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "FarmVista MRMS rainfall service",
        "timeUtc": iso_utc(datetime.now(timezone.utc)),
        "importError": IMPORT_ERROR,
    })


@app.get("/health")
def health():
    return jsonify({
        "ok": IMPORT_ERROR is None,
        "importError": IMPORT_ERROR,
        "cache": get_cache_meta() if IMPORT_ERROR is None else None,
        "timeUtc": iso_utc(datetime.now(timezone.utc)),
    })


@app.get("/latest")
def latest():
    try:
        ensure_runtime_ready()
        lat = num(request.args.get("lat"))
        lon = num(request.args.get("lon"))
        if lat is None or lon is None:
            return jsonify({"ok": False, "error": "lat/lon required"}), 400

        region = request.args.get("region", DEFAULT_REGION)
        radius_miles = clamp(request.args.get("radiusMiles", DEFAULT_RADIUS_MILES), 0.1, 5.0)

        now_utc = datetime.now(timezone.utc)
        product, key = choose_best_product_for_latest(region, now_utc)
        if not product or not key:
            return jsonify({"ok": False, "error": "No MRMS key found"}), 404

        prepare_cache_for_key(key, product)
        rain_in, samples = compute_weighted_rain(lat, lon, radius_miles)
        meta = get_cache_meta()

        return jsonify({
            "ok": True,
            "lat": round_num(lat, 6),
            "lon": round_num(lon, 6),
            "radiusMiles": radius_miles,
            "rainIn": rain_in,
            "source": "NOAA_AWS_MRMS",
            "product": meta["selectedProduct"],
            "fileTimestampUtc": meta["fileTimestampUtc"],
            "samplePoints": samples,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/bulk")
def bulk():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        points = body.get("points") or []
        if not isinstance(points, list) or not points:
            return jsonify({"ok": False, "error": "points[] required"}), 400
        if len(points) > MAX_BULK_POINTS:
            return jsonify({"ok": False, "error": f"Too many points; max={MAX_BULK_POINTS}"}), 400

        region = body.get("region", DEFAULT_REGION)
        radius_miles = clamp(body.get("radiusMiles", DEFAULT_RADIUS_MILES), 0.1, 5.0)

        now_utc = datetime.now(timezone.utc)
        product, key = choose_best_product_for_latest(region, now_utc)
        if not product or not key:
            return jsonify({"ok": False, "error": "No MRMS key found"}), 404

        prepare_cache_for_key(key, product)
        meta = get_cache_meta()

        results = []
        for i, pt in enumerate(points):
            lat = num(pt.get("lat"))
            lon = num(pt.get("lon"))
            if lat is None or lon is None:
                results.append({"index": i, "ok": False, "error": "lat/lon required"})
                continue

            rain_in, samples = compute_weighted_rain(lat, lon, radius_miles)
            results.append({
                "index": i,
                "ok": True,
                "lat": round_num(lat, 6),
                "lon": round_num(lon, 6),
                "rainIn": rain_in,
                "samplePoints": samples,
            })

        return jsonify({
            "ok": True,
            "count": len(results),
            "source": "NOAA_AWS_MRMS",
            "product": meta["selectedProduct"],
            "fileTimestampUtc": meta["fileTimestampUtc"],
            "radiusMiles": radius_miles,
            "results": results,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/run")
def run_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        region = body.get("region", DEFAULT_REGION)
        radius_miles = clamp(body.get("radiusMiles", DEFAULT_RADIUS_MILES), 0.1, 5.0)

        batch_result = run_batch_cache(region=region, radius_miles=radius_miles)

        max_fields = int(body.get("backfillMaxFields", DEFAULT_BACKFILL_MAX_FIELDS_PER_RUN))
        max_minutes = float(body.get("backfillMaxMinutes", DEFAULT_BACKFILL_MAX_MINUTES_PER_RUN))
        backfill_result = run_backfill_queue(max_fields=max_fields, max_minutes=max_minutes)

        return jsonify({
            "ok": True,
            "mode": "run",
            "ranAtUtc": iso_utc(datetime.now(timezone.utc)),
            "batch": batch_result,
            "backfill": backfill_result,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/backfill-field")
def backfill_field_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        reason = body.get("reason", "manual")
        payload = enqueue_full_backfill(field_id, reason=reason)
        return jsonify({"ok": True, "queued": payload})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/repair-field")
def repair_field_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        lookback_hours = int(body.get("lookbackHours", DEFAULT_REPAIR_LOOKBACK_HOURS))
        queued = maybe_auto_enqueue_gap_repair(field_id, lookback_hours=lookback_hours)
        return jsonify({"ok": True, "repair": queued})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/run-backfill")
def run_backfill_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        max_fields = int(body.get("maxFields", DEFAULT_BACKFILL_MAX_FIELDS_PER_RUN))
        max_minutes = float(body.get("maxMinutes", DEFAULT_BACKFILL_MAX_MINUTES_PER_RUN))
        result = run_backfill_queue(max_fields=max_fields, max_minutes=max_minutes)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/rebuild-field-parent")
def rebuild_field_parent_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        rebuilt = finalize_field_parent_from_hourly(field_id)
        return jsonify({"ok": True, "fieldId": field_id, "rebuilt": rebuilt})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/field-state")
def field_state_route():
    try:
        ensure_runtime_ready()
        field_id = request.args.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        parent = parent_snapshot(field_id) or {}
        state = get_field_mrms_state(field_id)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "state": state,
            "parent": parent,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/queue-status")
def queue_status_route():
    try:
        ensure_runtime_ready()

        def count_for_status(status):
            count = 0
            docs = backfill_queue_ref().where("status", "==", status).limit(10000).stream()
            for _ in docs:
                count += 1
            return count

        out = {
            "queued": count_for_status("queued"),
            "running": count_for_status("running"),
            "done": count_for_status("done"),
            "failed": count_for_status("failed"),
        }
        return jsonify({"ok": True, "queue": out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-key")
def debug_key_route():
    try:
        ensure_runtime_ready()
        region = request.args.get("region", DEFAULT_REGION)
        now_utc = datetime.now(timezone.utc)
        found = []
        for product in PRODUCT_PRIORITY:
            key = list_latest_key(region, product, now_utc)
            found.append({"product": product, "latestKey": key})
        return jsonify({
            "ok": True,
            "region": region,
            "nowUtc": iso_utc(now_utc),
            "products": found,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-hour")
def debug_hour_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        hour_utc = body.get("hourUtc")
        if not field_id or not hour_utc:
            return jsonify({"ok": False, "error": "fieldId and hourUtc required"}), 400

        target_hour_utc = parse_iso_utc(hour_utc)
        field = load_field_for_job(field_id)
        if not field:
            return jsonify({"ok": False, "error": "Field not found"}), 404

        result = fetch_hour_for_field(field, target_hour_utc)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-field-hours")
def debug_field_hours_route():
    try:
        ensure_runtime_ready()
        field_id = request.args.get("fieldId")
        limit = int(request.args.get("limit", "100"))
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        rows = []
        docs = (
            hourly_collection_ref(field_id)
            .order_by("fileTimestampUtc", direction=firestore.Query.DESCENDING)
            .limit(max(1, limit))
            .stream()
        )
        for doc in docs:
            row = doc.to_dict() or {}
            row["_id"] = doc.id
            rows.append(row)

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "count": len(rows),
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-clear-field")
def debug_clear_field_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        deleted_hourly = clear_hourly_history(field_id)
        deleted_repairs = clear_repair_jobs_for_field(field_id)
        reset_full_backfill_job(field_id)
        clear_parent_for_location_change(field_id)

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "deletedHourlyDocs": deleted_hourly,
            "deletedRepairJobs": deleted_repairs,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-enqueue-full")
def debug_enqueue_full_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        reason = body.get("reason", "manualDebug")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        queued = enqueue_full_backfill(field_id, reason=reason)
        return jsonify({"ok": True, "queued": queued})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-run-one-full")
def debug_run_one_full_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        job_id = full_backfill_job_id(field_id)
        doc = backfill_queue_ref().document(job_id).get()
        if not doc.exists:
            enqueue_full_backfill(field_id, reason="manualDebug")
            doc = backfill_queue_ref().document(job_id).get()

        result = process_full_backfill_job(doc)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-run-one-repair")
def debug_run_one_repair_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        job_id = body.get("jobId")
        if not job_id:
            return jsonify({"ok": False, "error": "jobId required"}), 400

        doc = backfill_queue_ref().document(job_id).get()
        if not doc.exists:
            return jsonify({"ok": False, "error": "repair job not found"}), 404

        result = process_repair_job(doc)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-missing-hours")
def debug_missing_hours_route():
    try:
        ensure_runtime_ready()
        field_id = request.args.get("fieldId")
        lookback_hours = int(request.args.get("lookbackHours", DEFAULT_REPAIR_LOOKBACK_HOURS))
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        missing = find_missing_recent_hours(field_id, lookback_hours=lookback_hours)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "lookbackHours": lookback_hours,
            "missingHoursUtc": [iso_utc(h) for h in missing],
            "count": len(missing),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-rebuild-preview")
def debug_rebuild_preview_route():
    try:
        ensure_runtime_ready()
        field_id = request.args.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        rebuilt = rebuild_last24_and_daily30(field_id)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "preview": rebuilt,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-parent-vs-rebuilt")
def debug_parent_vs_rebuilt_route():
    try:
        ensure_runtime_ready()
        field_id = request.args.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        parent = parent_snapshot(field_id) or {}
        rebuilt = rebuild_last24_and_daily30(field_id)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "parent": {
                "mrmsHourlyLast24": parent.get("mrmsHourlyLast24"),
                "mrmsDailySeries30d": parent.get("mrmsDailySeries30d"),
                "mrmsRainLast24h": parent.get("mrmsRainLast24h"),
            },
            "rebuilt": rebuilt,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


def create_or_update_exact_hour(field_id, file_timestamp_utc, rain_in, product=None):
    hour_id = hour_doc_id_from_iso(file_timestamp_utc)
    ref = hourly_collection_ref(field_id).document(hour_id)
    ref.set({
        "fieldId": str(field_id),
        "fileTimestampUtc": file_timestamp_utc,
        "rainIn": round_num(num(rain_in) or 0.0, 4),
        "product": product,
        "updatedAtUtc": iso_utc(datetime.now(timezone.utc)),
    }, merge=True)
    return hour_id


def update_parent_after_exact_hour(field_id, file_timestamp_utc, rain_in):
    parent = parent_snapshot(field_id) or {}
    payload = {
        "fieldId": str(field_id),
        "fileTimestampUtc": file_timestamp_utc,
        "rainIn": round_num(num(rain_in) or 0.0, 4),
        "computedAtUtc": iso_utc(datetime.now(timezone.utc)),
        "samplePoints": [],
    }
    update = build_incremental_parent_update(parent, payload)
    field_doc_ref(field_id).set(update, merge=True)
    return update


@app.post("/debug-insert-hour")
def debug_insert_hour_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        file_timestamp_utc = body.get("fileTimestampUtc")
        rain_in = body.get("rainIn")
        if not field_id or not file_timestamp_utc or rain_in is None:
            return jsonify({"ok": False, "error": "fieldId, fileTimestampUtc, rainIn required"}), 400

        hour_id = create_or_update_exact_hour(field_id, file_timestamp_utc, rain_in, product=body.get("product"))
        update = update_parent_after_exact_hour(field_id, file_timestamp_utc, rain_in)

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "hourDocId": hour_id,
            "parentUpdate": update,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-cache-meta")
def debug_cache_meta_route():
    try:
        ensure_runtime_ready()
        return jsonify({
            "ok": True,
            "cache": get_cache_meta(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-load-latest-cache")
def debug_load_latest_cache_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        region = body.get("region", DEFAULT_REGION)
        now_utc = datetime.now(timezone.utc)
        product, key = choose_best_product_for_latest(region, now_utc)
        if not product or not key:
            return jsonify({"ok": False, "error": "No MRMS key found"}), 404
        prepare_cache_for_key(key, product)
        return jsonify({
            "ok": True,
            "cache": get_cache_meta(),
            "selectedKey": key,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-products")
def debug_products_route():
    try:
        ensure_runtime_ready()
        now_utc = datetime.now(timezone.utc)
        region = request.args.get("region", DEFAULT_REGION)
        rows = []
        for product in PRODUCT_PRIORITY:
            key = list_latest_key(region, product, now_utc)
            rows.append({
                "product": product,
                "latestKey": key,
            })
        return jsonify({
            "ok": True,
            "region": region,
            "nowUtc": iso_utc(now_utc),
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-hour-key")
def debug_hour_key_route():
    try:
        ensure_runtime_ready()
        region = request.args.get("region", DEFAULT_REGION)
        hour_utc = request.args.get("hourUtc")
        if not hour_utc:
            return jsonify({"ok": False, "error": "hourUtc required"}), 400
        target_hour_utc = parse_iso_utc(hour_utc)
        product, key = choose_best_product_for_hour(region, target_hour_utc)
        return jsonify({
            "ok": True,
            "region": region,
            "hourUtc": iso_utc(target_hour_utc),
            "product": product,
            "key": key,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-active-fields")
def debug_active_fields_route():
    try:
        ensure_runtime_ready()
        rows = []
        for field in stream_active_fields():
            lat, lng = field_lat_lng(field)
            rows.append({
                "fieldId": field.get("id"),
                "fieldName": field.get("name"),
                "lat": lat,
                "lng": lng,
                "farmId": field.get("farmId"),
                "archived": field.get("archived"),
            })
        return jsonify({
            "ok": True,
            "count": len(rows),
            "rows": rows[:2000],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-field-current")
def debug_field_current_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        region = body.get("region", DEFAULT_REGION)
        radius_miles = clamp(body.get("radiusMiles", DEFAULT_RADIUS_MILES), 0.1, 5.0)

        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        field = load_field_for_job(field_id)
        if not field:
            return jsonify({"ok": False, "error": "Field not found"}), 404

        now_utc = datetime.now(timezone.utc)
        product, key = choose_best_product_for_latest(region, now_utc)
        if not product or not key:
            return jsonify({"ok": False, "error": "No MRMS key found"}), 404

        prepare_cache_for_key(key, product)
        meta = get_cache_meta()

        lat, lng = field_lat_lng(field)
        rain_in, samples = compute_weighted_rain(lat, lng, radius_miles)

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "fieldName": field.get("name"),
            "lat": lat,
            "lng": lng,
            "rainIn": rain_in,
            "samplePoints": samples,
            "fileTimestampUtc": meta["fileTimestampUtc"],
            "product": meta["selectedProduct"],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-run-current-field-write")
def debug_run_current_field_write_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        region = body.get("region", DEFAULT_REGION)
        radius_miles = clamp(body.get("radiusMiles", DEFAULT_RADIUS_MILES), 0.1, 5.0)

        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        field = load_field_for_job(field_id)
        if not field:
            return jsonify({"ok": False, "error": "Field not found"}), 404

        now_utc = datetime.now(timezone.utc)
        product, key = choose_best_product_for_latest(region, now_utc)
        if not product or not key:
            return jsonify({"ok": False, "error": "No MRMS key found"}), 404

        prepare_cache_for_key(key, product)
        meta = get_cache_meta()

        lat, lng = field_lat_lng(field)
        rain_in, samples = compute_weighted_rain(lat, lng, radius_miles)

        result = write_field_hour(
            field=field,
            lat=lat,
            lng=lng,
            rain_in=rain_in,
            samples=samples,
            file_timestamp_utc=meta["fileTimestampUtc"],
            region=region,
            radius_miles=radius_miles,
            product=meta["selectedProduct"],
        )

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "result": result,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-force-location-reset")
def debug_force_location_reset_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        deleted_hourly = clear_hourly_history(field_id)
        deleted_repairs = clear_repair_jobs_for_field(field_id)
        reset_full_backfill_job(field_id)
        clear_parent_for_location_change(field_id)
        queued = enqueue_full_backfill(field_id, reason="manualLocationReset")

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "deletedHourlyDocs": deleted_hourly,
            "deletedRepairJobs": deleted_repairs,
            "queued": queued,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-force-parent-rebuild")
def debug_force_parent_rebuild_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        rebuilt = finalize_field_parent_from_hourly(field_id, extra_merge={
            "manualRebuildAtUtc": iso_utc(datetime.now(timezone.utc)),
        })

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "rebuilt": rebuilt,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-parent")
def debug_parent_route():
    try:
        ensure_runtime_ready()
        field_id = request.args.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        parent = parent_snapshot(field_id)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "parent": parent,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-doc-ids")
def debug_doc_ids_route():
    try:
        ensure_runtime_ready()
        ts = request.args.get("fileTimestampUtc")
        if not ts:
            return jsonify({"ok": False, "error": "fileTimestampUtc required"}), 400
        return jsonify({
            "ok": True,
            "fileTimestampUtc": ts,
            "hourDocId": hour_doc_id_from_iso(ts),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-enqueue-repair")
def debug_enqueue_repair_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        start_hour_utc = body.get("startHourUtc")
        end_hour_utc = body.get("endHourUtc")
        if not field_id or not start_hour_utc or not end_hour_utc:
            return jsonify({"ok": False, "error": "fieldId, startHourUtc, endHourUtc required"}), 400

        payload = enqueue_repair_job(
            field_id=field_id,
            start_hour_utc=parse_iso_utc(start_hour_utc),
            end_hour_utc=parse_iso_utc(end_hour_utc),
            reason=body.get("reason", "manualDebug"),
        )
        return jsonify({
            "ok": True,
            "queued": payload,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-queue")
def debug_queue_route():
    try:
        ensure_runtime_ready()
        rows = []
        docs = backfill_queue_ref().limit(500).stream()
        for doc in docs:
            row = doc.to_dict() or {}
            row["_id"] = doc.id
            rows.append(row)
        return jsonify({
            "ok": True,
            "count": len(rows),
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-clear-queue")
def debug_clear_queue_route():
    try:
        ensure_runtime_ready()
        deleted = 0
        while True:
            docs = list(backfill_queue_ref().limit(400).stream())
            if not docs:
                break
            batch = get_db().batch()
            for d in docs:
                batch.delete(d.reference)
                deleted += 1
            batch.commit()
            if len(docs) < 400:
                break
        return jsonify({
            "ok": True,
            "deleted": deleted,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-now")
def debug_now_route():
    try:
        ensure_runtime_ready()
        now_utc = datetime.now(timezone.utc)
        app_tz = get_app_tz()
        return jsonify({
            "ok": True,
            "utcNow": iso_utc(now_utc),
            "localNow": now_utc.astimezone(app_tz).isoformat(),
            "timeZone": APP_TIMEZONE,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-recompute-parent-incremental")
def debug_recompute_parent_incremental_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        file_timestamp_utc = body.get("fileTimestampUtc")
        rain_in = body.get("rainIn")
        if not field_id or not file_timestamp_utc or rain_in is None:
            return jsonify({"ok": False, "error": "fieldId, fileTimestampUtc, rainIn required"}), 400

        parent = parent_snapshot(field_id) or {}
        update = build_incremental_parent_update(parent, {
            "fieldId": str(field_id),
            "fileTimestampUtc": file_timestamp_utc,
            "rainIn": round_num(num(rain_in) or 0.0, 4),
            "computedAtUtc": iso_utc(datetime.now(timezone.utc)),
            "samplePoints": [],
        })

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "incrementalPreview": update,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-hourly-retention")
def debug_hourly_retention_route():
    try:
        ensure_runtime_ready()
        field_id = request.args.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400

        rows = []
        docs = (
            hourly_collection_ref(field_id)
            .order_by("fileTimestampUtc", direction=firestore.Query.ASCENDING)
            .limit(1000)
            .stream()
        )
        for doc in docs:
            row = doc.to_dict() or {}
            rows.append({
                "_id": doc.id,
                "fileTimestampUtc": row.get("fileTimestampUtc"),
                "rainIn": row.get("rainIn"),
            })

        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "count": len(rows),
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/debug-prune-old-hours")
def debug_prune_old_hours_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        newest_hour_iso = body.get("newestHourUtc")
        if not field_id or not newest_hour_iso:
            return jsonify({"ok": False, "error": "fieldId and newestHourUtc required"}), 400

        result = maybe_prune_old_hourly_docs(field_id, newest_hour_iso)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "result": result,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/debug-sample")
def debug_sample_route():
    try:
        ensure_runtime_ready()
        lat = num(request.args.get("lat"))
        lon = num(request.args.get("lon"))
        region = request.args.get("region", DEFAULT_REGION)
        radius_miles = clamp(request.args.get("radiusMiles", DEFAULT_RADIUS_MILES), 0.1, 5.0)
        if lat is None or lon is None:
            return jsonify({"ok": False, "error": "lat/lon required"}), 400

        now_utc = datetime.now(timezone.utc)
        product, key = choose_best_product_for_latest(region, now_utc)
        if not product or not key:
            return jsonify({"ok": False, "error": "No MRMS key found"}), 404

        prepare_cache_for_key(key, product)
        rain_in, samples = compute_weighted_rain(lat, lon, radius_miles)
        return jsonify({
            "ok": True,
            "lat": lat,
            "lon": lon,
            "radiusMiles": radius_miles,
            "rainIn": rain_in,
            "samples": samples,
            "cache": get_cache_meta(),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------
# Compatibility aliases / older route names you may already be hitting
# ---------------------------------------------------------------------

@app.post("/run-batch-cache")
def run_batch_cache_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        region = body.get("region", DEFAULT_REGION)
        radius_miles = clamp(body.get("radiusMiles", DEFAULT_RADIUS_MILES), 0.1, 5.0)
        result = run_batch_cache(region=region, radius_miles=radius_miles)
        return jsonify({
            "ok": True,
            "mode": "batch_only",
            "ranAtUtc": iso_utc(datetime.now(timezone.utc)),
            "batch": result,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/rebuild-last24-and-daily30")
def rebuild_last24_and_daily30_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400
        result = rebuild_last24_and_daily30(field_id)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "rebuilt": result,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/finalize-parent")
def finalize_parent_route():
    try:
        ensure_runtime_ready()
        body = request.get_json(silent=True) or {}
        field_id = body.get("fieldId")
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId required"}), 400
        result = finalize_field_parent_from_hourly(field_id)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "finalized": result,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
