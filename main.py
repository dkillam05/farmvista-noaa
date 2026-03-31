# =====================================================================
# main.py  (FULL FILE)
# FarmVista NOAA MRMS Pass2->Pass1 fallback rainfall service
# Rev: 2026-03-30b-hourly-authoritative-subcollection-fix
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
# ✅ FIX: existing fields no longer reread all historical hourly docs on each normal write
# ✅ FIX: parent last24 + daily30 rollups now update incrementally on normal writes
# ✅ FIX: if incremental rollup fails, fall back to full rebuild from saved hourly docs
# ✅ FIX: parent payload no longer tries to store non-Firestore-safe objects
# ✅ FIX: full rebuild path is preserved for full backfill / repair completion only
# ✅ FIX: restore robust longitude/grid sampling logic from prior working version
# ✅ FIX: normal writes skip duplicate/stale latest hour for each field
# ✅ FIX: hourly next-needed targeting now trusts actual hourly subcollection first,
#         not just the parent doc latest timestamp
# ✅ FIX: prevents false "up to date" when parent doc got ahead of saved hourly docs
#
# NOTES
# - This file keeps the original route/helper structure.
# - This file handles MRMS automation only.
# - Weather-cache automation is still handled elsewhere.
# - Backfill jobs always read the current field lat/lng from Firestore.
# =====================================================================

import gzip
import math
import os
import tempfile
import threading
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
    "cacheHit": False,
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


def in_hourly_pause_window():
    now = datetime.now(get_app_tz())
    minute = now.minute
    return 18 <= minute <= 24


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

    dated.sort(key=lambda x: x[0])
    return dated[-1][1]
    
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


def pick_best_key_for_latest():
    now_utc = datetime.now(timezone.utc)
    checked = []
    candidates = []

    for product in PRODUCT_PRIORITY:
        key = list_latest_key(DEFAULT_REGION, product, now_utc)

        parsed_ts = None
        try:
            parsed_ts = parse_timestamp_from_key(key) if key else None
        except Exception:
            parsed_ts = None

        checked.append({
            "product": product,
            "found": bool(key),
            "selectedKey": key.replace(f"{AWS_BUCKET}/", "") if key else None,
            "parsedTimestampUtc": iso_utc(parsed_ts) if parsed_ts else None,
        })

        if key and parsed_ts:
            candidates.append({
                "product": product,
                "key": key,
                "ts": parsed_ts,
            })

    if not candidates:
        return None, None, checked

    priority_rank = {name: idx for idx, name in enumerate(PRODUCT_PRIORITY)}
    candidates.sort(key=lambda x: (x["ts"], -priority_rank.get(x["product"], 999)))
    winner = candidates[-1]

    print(
        f"[MRMS Latest Winner] product={winner['product']} "
        f"fileTimestampUtc={iso_utc(winner['ts'])} "
        f"selectedKey={winner['key'].replace(f'{AWS_BUCKET}/', '')}",
        flush=True
    )

    return winner["product"], winner["key"], checked


def pick_best_key_for_hour(target_dt_utc):
    checked = []

    for product in PRODUCT_PRIORITY:
        key = list_key_for_exact_hour(DEFAULT_REGION, product, target_dt_utc)
        checked.append({
            "product": product,
            "found": bool(key),
            "selectedKey": key.replace(f"{AWS_BUCKET}/", "") if key else None,
        })
        if key:
            return product, key, checked

    return None, None, checked


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
        try:
            ds = xr.open_dataset(local_path, engine="cfgrib")
            ds.load()
            return ds
        except Exception:
            ds = xr.open_dataset(local_path, engine="pynio")
            ds.load()
            return ds
    finally:
        try:
            os.remove(local_path)
        except Exception:
            pass


def get_data_var(ds):
    preferred = ["unknown", "tp", "precipitation", "precip", "param18.0.0", "paramId_0"]

    for name in preferred:
        if name in ds.data_vars:
            return ds[name], name

    data_vars = list(ds.data_vars.keys())
    if not data_vars:
        raise RuntimeError("No data variables found in MRMS GRIB.")
    return ds[data_vars[0]], data_vars[0]


def normalize_data_array(da):
    rename_map = {}

    if "latitude" not in da.coords and "lat" in da.coords:
        rename_map["lat"] = "latitude"
    if "longitude" not in da.coords and "lon" in da.coords:
        rename_map["lon"] = "longitude"

    if rename_map:
        da = da.rename(rename_map)

    if "latitude" not in da.coords or "longitude" not in da.coords:
        raise RuntimeError(
            f"Dataset missing usable latitude/longitude coordinates. Found coords: {list(da.coords)}"
        )

    for dim in list(da.dims):
        if dim not in ("latitude", "longitude") and da.sizes.get(dim, 0) == 1:
            da = da.isel({dim: 0})

    return da


def detect_lon_convention(da):
    lon_vals = da["longitude"].values
    lon_min = float(np.nanmin(lon_vals))
    lon_max = float(np.nanmax(lon_vals))

    if lon_min >= 0.0 and lon_max > 180.0:
        return "0_360"

    return "signed"


def to_dataset_lon(lon, lon_mode):
    if lon_mode == "0_360":
        return lon if lon >= 0 else lon + 360.0
    return lon


def to_signed_lon(lon):
    return lon - 360.0 if lon > 180.0 else lon


def prepare_cache_for_key(key, product):
    ts = parse_timestamp_from_key(key)
    with CACHE_LOCK:
        if CACHE.get("selectedKey") == key and CACHE.get("dataArray") is not None:
            CACHE["cacheHit"] = True
            return

    ds = load_dataset_for_key(key)
    da_raw, variable_name = get_data_var(ds)
    da = normalize_data_array(da_raw)

    with CACHE_LOCK:
        old_ds = CACHE.get("io")
        CACHE["selectedKey"] = key
        CACHE["selectedProduct"] = product
        CACHE["fileTimestampUtc"] = iso_utc(ts) if ts else None
        CACHE["variableName"] = variable_name
        CACHE["dataArray"] = da
        CACHE["io"] = ds
        CACHE["cacheHit"] = False

    try:
        if old_ds is not None and old_ds is not ds:
            old_ds.close()
    except Exception:
        pass


def get_cached_dataset():
    product, key, checked = pick_best_key_for_latest()
    if not product or not key:
        raise RuntimeError("Could not find recent MRMS data from Pass2 or Pass1.")

    prepare_cache_for_key(key, product)
    with CACHE_LOCK:
        return {
            "selectedProduct": CACHE.get("selectedProduct"),
            "selectedKey": CACHE.get("selectedKey"),
            "fileTimestampUtc": CACHE.get("fileTimestampUtc"),
            "variableName": CACHE.get("variableName"),
            "dataArray": CACHE.get("dataArray"),
            "cacheHit": bool(CACHE.get("cacheHit")),
            "io": None,
            "checkedProducts": checked,
        }


def get_dataset_for_key_uncached(product, key):
    ds = load_dataset_for_key(key)
    da_raw, variable_name = get_data_var(ds)
    da = normalize_data_array(da_raw)
    ts = parse_timestamp_from_key(key)

    return {
        "selectedProduct": product,
        "selectedKey": key,
        "fileTimestampUtc": iso_utc(ts) if ts else None,
        "variableName": variable_name,
        "dataArray": da,
        "cacheHit": False,
        "io": ds,
    }


def get_cached_dataset_for_exact_hour(target_dt_utc, latest_meta=None):
    target_dt_utc = floor_to_hour_utc(target_dt_utc)

    if latest_meta:
        try:
            latest_dt = floor_to_hour_utc(parse_iso_utc(latest_meta["fileTimestampUtc"]))
            if latest_dt == target_dt_utc:
                return latest_meta
        except Exception:
            pass

    product, key, checked = pick_best_key_for_hour(target_dt_utc)
    if not product or not key:
        return None

    prepare_cache_for_key(key, product)
    with CACHE_LOCK:
        return {
            "selectedProduct": CACHE.get("selectedProduct"),
            "selectedKey": CACHE.get("selectedKey"),
            "fileTimestampUtc": CACHE.get("fileTimestampUtc"),
            "variableName": CACHE.get("variableName"),
            "dataArray": CACHE.get("dataArray"),
            "cacheHit": bool(CACHE.get("cacheHit")),
            "io": None,
            "checkedProducts": checked,
        }


def sample_nearest_with_grid(da, lat, lon):
    lon_mode = detect_lon_convention(da)
    query_lon = to_dataset_lon(lon, lon_mode)

    sampled = da.sel(latitude=lat, longitude=query_lon, method="nearest")

    value = sampled.values
    if isinstance(value, np.ndarray):
        value = np.asarray(value).squeeze()
        if np.size(value) != 1:
            raise RuntimeError("Unexpected non-scalar sampled value from MRMS dataset.")
        value = float(value)
    else:
        value = float(value)

    grid_lat = float(sampled["latitude"].values)
    raw_grid_lon = float(sampled["longitude"].values)
    signed_grid_lon = to_signed_lon(raw_grid_lon)

    if not math.isfinite(value) or value < 0:
        value = 0.0

    return {
        "mm": value,
        "nearestGridLatitude": grid_lat,
        "nearestGridLongitude": signed_grid_lon,
        "nearestGridLongitudeRaw": raw_grid_lon,
        "queryLongitudeUsed": query_lon,
        "longitudeMode": lon_mode,
    }


def mm_from_dataset_value(value):
    if value is None or not math.isfinite(value):
        return None
    if value < 0:
        value = 0.0
    return value


def compute_weighted_mm(da, lat, lon, radius_miles):
    total_weight = 0.0
    weighted_sum = 0.0
    samples_out = []

    for sp in SAMPLE_POINTS:
        p_lat, p_lon = offset_point(
            lat, lon,
            east_miles=sp["dxMiles"] * radius_miles,
            north_miles=sp["dyMiles"] * radius_miles,
        )
        sampled = sample_nearest_with_grid(da, p_lat, p_lon)
        rain_mm = mm_from_dataset_value(sampled["mm"])
        if rain_mm is None:
            rain_mm = 0.0

        weighted_sum += rain_mm * sp["weight"]
        total_weight += sp["weight"]
        samples_out.append({
            "key": sp["key"],
            "lat": round_num(p_lat, 6),
            "lon": round_num(p_lon, 6),
            "weight": sp["weight"],
            "mm": round_num(rain_mm, 4),
            "nearestGridLatitude": round_num(sampled["nearestGridLatitude"], 6),
            "nearestGridLongitude": round_num(sampled["nearestGridLongitude"], 6),
            "nearestGridLongitudeRaw": round_num(sampled["nearestGridLongitudeRaw"], 6),
            "queryLongitudeUsed": round_num(sampled["queryLongitudeUsed"], 6),
            "longitudeMode": sampled["longitudeMode"],
            "ok": True,
        })

    final_mm = weighted_sum / total_weight if total_weight > 0 else 0.0
    if final_mm < 0:
        final_mm = 0.0

    return round_num(final_mm, 4), samples_out


def parse_bulk_points_from_query(points_str):
    points = []
    if not points_str:
        return points

    chunks = [x.strip() for x in points_str.split(";") if x.strip()]
    for idx, chunk in enumerate(chunks):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise RuntimeError(f"Invalid points format at item {idx + 1}. Use lat,lon;lat,lon")
        lat = num(parts[0])
        lon = num(parts[1])
        if lat is None or lon is None:
            raise RuntimeError(f"Invalid numeric lat/lon at item {idx + 1}")
        points.append({"lat": lat, "lon": lon})
    return points


def validate_point(lat, lon):
    if lat is None or lon is None:
        raise RuntimeError("Missing or invalid lat/lon")
    if lat < -90 or lat > 90 or lon < -180 or lon > 180:
        raise RuntimeError("lat/lon out of range")


def build_single_result(da, lat, lon):
    sampled = sample_nearest_with_grid(da, lat, lon)
    hourly_rain_mm = mm_from_dataset_value(sampled["mm"])
    if hourly_rain_mm is None:
        hourly_rain_mm = 0.0

    return {
        "lat": round_num(lat, 6),
        "lon": round_num(lon, 6),
        "hourlyRainMm": round_num(hourly_rain_mm, 4),
        "nearestGridLatitude": round_num(sampled["nearestGridLatitude"], 6),
        "nearestGridLongitude": round_num(sampled["nearestGridLongitude"], 6),
        "nearestGridLongitudeRaw": round_num(sampled["nearestGridLongitudeRaw"], 6),
        "queryLongitudeUsed": round_num(sampled["queryLongitudeUsed"], 6),
        "longitudeMode": sampled["longitudeMode"],
    }


def build_weighted_result(da, lat, lon, radius_miles):
    weighted_mm, samples = compute_weighted_mm(da, lat, lon, radius_miles)

    return {
        "lat": round_num(lat, 6),
        "lon": round_num(lon, 6),
        "radiusMiles": radius_miles,
        "weightedHourlyRainMm": round_num(weighted_mm, 4),
        "attemptedPointCount": len(SAMPLE_POINTS),
        "successfulPointCount": len(samples),
        "samples": samples,
    }


def extract_field_lat_lng(data):
    d = data or {}
    loc = d.get("location") or {}

    lat = num(loc.get("lat") if isinstance(loc, dict) else None)
    lng = num(loc.get("lng") if isinstance(loc, dict) else None)

    if lat is None:
        lat = num(d.get("lat"))
    if lng is None:
        lng = num(d.get("lng"))
    if lng is None:
        lng = num(d.get("lon"))
    if lng is None:
        lng = num(d.get("long"))

    return lat, lng


def field_doc_is_active(data):
    d = data or {}
    status = str(d.get("status") or "").strip().lower()

    archived = d.get("archived")
    is_archived = archived is True

    active_flag = d.get("active")
    is_active_flag = active_flag is True

    enabled_flag = d.get("enabled")
    is_enabled_flag = enabled_flag is True

    if status in {"archived", "inactive", "deleted", "disabled"}:
        return False
    if is_archived:
        return False
    if status == "active":
        return True
    if is_active_flag or is_enabled_flag:
        return True
    if status == "":
        return True

    return False


def build_field_stub(doc_id, data):
    d = data or {}
    lat, lng = extract_field_lat_lng(d)
    if lat is None or lng is None:
        return None

    return {
        "id": doc_id,
        "name": str(d.get("name") or d.get("fieldName") or ""),
        "farmId": d.get("farmId"),
        "farmName": d.get("farmName"),
        "lat": lat,
        "lng": lng,
        "status": str(d.get("status") or "").strip().lower(),
        "active": d.get("active"),
        "archived": d.get("archived"),
    }


def load_active_fields_for_batch():
    db = get_db()
    raw = []
    seen = set()

    try:
        snap = db.collection(FIELDS_COLLECTION).where("status", "==", "active").stream()
        for doc in snap:
            if doc.id in seen:
                continue
            seen.add(doc.id)
            raw.append({"id": doc.id, "data": doc.to_dict() or {}})
    except Exception as e:
        print(f"[Batch] fields query(status==active) failed: {e}", flush=True)

    try:
        snap2 = db.collection(FIELDS_COLLECTION).stream()
        for doc in snap2:
            if doc.id in seen:
                continue
            seen.add(doc.id)
            raw.append({"id": doc.id, "data": doc.to_dict() or {}})
    except Exception as e:
        print(f"[Batch] fields query(all) failed: {e}", flush=True)

    out = []
    for r in raw:
        d = r["data"] or {}
        if not field_doc_is_active(d):
            continue

        field = build_field_stub(r["id"], d)
        if not field:
            continue

        out.append(field)

    return out


def get_field_by_id(field_id):
    db = get_db()
    doc = db.collection(FIELDS_COLLECTION).document(field_id).get()
    if not doc.exists:
        return None

    d = doc.to_dict() or {}
    field = build_field_stub(doc.id, d)
    if not field:
        raise RuntimeError("Field exists but location.lat/lng is missing or invalid")

    return field


def extract_rain_value(mode, result):
    return result["hourlyRainMm"] if mode == "single" else result["weightedHourlyRainMm"]


def get_doc_rain_mm(doc_dict):
    rain = num(doc_dict.get("rainMm"))
    if rain is not None:
        return rain
    return num(doc_dict.get("rainInches"))


def build_hour_history_entry(meta, mode, radius_miles, result):
    entry = {
        "source": "noaa-mrms-aws",
        "selectedProduct": meta["selectedProduct"],
        "selectedKey": meta["selectedKey"].replace(f"{AWS_BUCKET}/", ""),
        "fileTimestampUtc": meta["fileTimestampUtc"],
        "variableName": meta["variableName"],
        "mode": mode,
        "radiusMiles": radius_miles if mode == "weighted" else None,
        "rainMm": extract_rain_value(mode, result),
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    if mode == "single":
        entry["nearestGridLatitude"] = result["nearestGridLatitude"]
        entry["nearestGridLongitude"] = result["nearestGridLongitude"]
        entry["nearestGridLongitudeRaw"] = result["nearestGridLongitudeRaw"]
        entry["queryLongitudeUsed"] = result["queryLongitudeUsed"]
        entry["longitudeMode"] = result["longitudeMode"]
    else:
        entry["attemptedPointCount"] = result["attemptedPointCount"]
        entry["successfulPointCount"] = result["successfulPointCount"]
        entry["samples"] = result["samples"]

    return entry


def locations_match(loc_a, loc_b):
    if not loc_a or not loc_b:
        return False

    a_lat = num(loc_a.get("lat"))
    a_lng = num(loc_a.get("lng"))
    b_lat = num(loc_b.get("lat"))
    b_lng = num(loc_b.get("lng"))

    if a_lat is None or a_lng is None or b_lat is None or b_lng is None:
        return False

    return (
        abs(a_lat - b_lat) <= LOCATION_EPSILON and
        abs(a_lng - b_lng) <= LOCATION_EPSILON
    )


def get_field_mrms_state(field_id):
    db = get_db()
    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field_id)
    snap = parent_ref.get()

    if not snap.exists:
        return {
            "docExists": False,
            "hasAnyHistory": False,
            "fullBackfillComplete": False,
            "daily30Count": 0,
            "historyMeta": {},
            "location": None,
        }

    data = snap.to_dict() or {}
    history_meta = data.get("mrmsHistoryMeta") or {}
    daily30 = data.get("mrmsDailySeries30d") or []

    has_any_history = False
    if isinstance(daily30, list) and len(daily30) > 0:
        has_any_history = True
    else:
        subdocs = list(
            parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION)
            .limit(1)
            .stream()
        )
        has_any_history = len(subdocs) > 0

    return {
        "docExists": True,
        "hasAnyHistory": has_any_history,
        "fullBackfillComplete": bool(history_meta.get("fullBackfillComplete", False)),
        "daily30Count": len(daily30) if isinstance(daily30, list) else 0,
        "historyMeta": history_meta,
        "location": data.get("location"),
    }


def build_latest_payload(field, meta, mode, radius_miles, result, last24, daily30, existing_state):
    history_meta_old = existing_state.get("historyMeta") or {}
    full_complete = bool(history_meta_old.get("fullBackfillComplete", False))
    backfill_completed_at = history_meta_old.get("backfillCompletedAt")
    latest_repair_enqueued_at = history_meta_old.get("latestRepairEnqueuedAt")

    latest = {
        "source": "noaa-mrms-aws",
        "selectedProduct": meta["selectedProduct"],
        "selectedKey": meta["selectedKey"].replace(f"{AWS_BUCKET}/", ""),
        "fileTimestampUtc": meta["fileTimestampUtc"],
        "variableName": meta["variableName"],
        "mode": mode,
        "radiusMiles": radius_miles if mode == "weighted" else None,
        "cacheHit": meta["cacheHit"],
        "checkedProducts": meta.get("checkedProducts"),
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "rainMm": extract_rain_value(mode, result),
    }

    if mode == "single":
        latest["hourlyRainMm"] = result["hourlyRainMm"]
        latest["nearestGridLatitude"] = result["nearestGridLatitude"]
        latest["nearestGridLongitude"] = result["nearestGridLongitude"]
        latest["nearestGridLongitudeRaw"] = result["nearestGridLongitudeRaw"]
        latest["queryLongitudeUsed"] = result["queryLongitudeUsed"]
        latest["longitudeMode"] = result["longitudeMode"]
    else:
        latest["weightedHourlyRainMm"] = result["weightedHourlyRainMm"]
        latest["attemptedPointCount"] = result["attemptedPointCount"]
        latest["successfulPointCount"] = result["successfulPointCount"]
        latest["samples"] = result["samples"]

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name") or None,
        "farmId": field.get("farmId"),
        "farmName": field.get("farmName"),
        "location": {"lat": field["lat"], "lng": field["lng"]},
        "mrmsHourlyLatest": latest,
        "mrmsHourlyLast24": last24,
        "mrmsDailySeries30d": daily30,
        "mrmsHistoryMeta": {
            "subcollection": MRMS_HOURLY_SUBCOLLECTION,
            "keepDays": KEEP_DAYS,
            "keepHours": KEEP_HOURS,
            "units": "mm",
            "latestFileTimestampUtc": meta["fileTimestampUtc"],
            "latestSelectedProduct": meta["selectedProduct"],
            "last24Count": len(last24),
            "daily30Count": len(daily30),
            "fullBackfillComplete": full_complete,
            "backfillCompletedAt": backfill_completed_at,
            "latestRepairEnqueuedAt": latest_repair_enqueued_at,
        },
        "mrmsLastUpdatedAt": firestore.SERVER_TIMESTAMP,
    }


def normalize_hour_history_row(row, fallback_hour_key=None):
    if not isinstance(row, dict):
        return None

    ts = str(row.get("fileTimestampUtc") or "").strip()
    if not ts:
        return None

    rain_mm = get_doc_rain_mm(row)
    if rain_mm is None:
        return None

    return {
        "hourKey": row.get("hourKey") or fallback_hour_key or hour_doc_id_from_iso(ts),
        "fileTimestampUtc": ts,
        "rainMm": round_num(rain_mm, 4),
        "selectedProduct": row.get("selectedProduct"),
        "mode": row.get("mode"),
        "source": row.get("source"),
    }


def trim_daily30_map(daily_map, anchor_dt_utc):
    tz = get_app_tz()
    anchor_local_date = anchor_dt_utc.astimezone(tz).date()
    keep_dates = set()
    for i in range(KEEP_DAYS):
        keep_dates.add((anchor_local_date - timedelta(days=i)).isoformat())

    daily30 = []
    for date_iso, row in daily_map.items():
        if date_iso not in keep_dates:
            continue
        daily30.append({
            "dateISO": date_iso,
            "rainMm": round_num(num(row.get("rainMm")) or 0.0, 4),
            "hoursCount": int(num(row.get("hoursCount")) or 0),
        })

    daily30.sort(key=lambda x: x["dateISO"])
    return daily30[-KEEP_DAYS:]


def build_incremental_rollups(existing_parent, new_history_entry, previous_hour_entry=None):
    tz = get_app_tz()
    new_row = normalize_hour_history_row(new_history_entry)
    if not new_row:
        raise RuntimeError("Unable to build incremental MRMS rollups; invalid hour entry.")

    new_dt = parse_iso_utc(new_row["fileTimestampUtc"])
    new_local_date = new_dt.astimezone(tz).date().isoformat()

    existing_last24 = []
    for row in (existing_parent.get("mrmsHourlyLast24") or []):
        norm = normalize_hour_history_row(row)
        if not norm:
            continue
        existing_last24.append(norm)

    row_map = {row["fileTimestampUtc"]: row for row in existing_last24}
    row_map[new_row["fileTimestampUtc"]] = new_row
    last24 = list(row_map.values())
    last24.sort(key=lambda x: x["fileTimestampUtc"], reverse=True)
    last24 = last24[:LAST24_COUNT]

    daily_map = {}
    for row in (existing_parent.get("mrmsDailySeries30d") or []):
        if not isinstance(row, dict):
            continue
        date_iso = str(row.get("dateISO") or "").strip()
        if not date_iso:
            continue
        daily_map[date_iso] = {
            "dateISO": date_iso,
            "rainMm": round_num(num(row.get("rainMm")) or 0.0, 4),
            "hoursCount": int(num(row.get("hoursCount")) or 0),
        }

    bucket = daily_map.get(new_local_date)
    if bucket is None:
        bucket = {"dateISO": new_local_date, "rainMm": 0.0, "hoursCount": 0}
        daily_map[new_local_date] = bucket

    prev_row = normalize_hour_history_row(previous_hour_entry) if previous_hour_entry else None
    prev_local_date = None
    if prev_row:
        try:
            prev_dt = parse_iso_utc(prev_row["fileTimestampUtc"])
            prev_local_date = prev_dt.astimezone(tz).date().isoformat()
        except Exception:
            prev_row = None
            prev_local_date = None

    if prev_row and prev_local_date:
        prev_bucket = daily_map.get(prev_local_date)
        if prev_bucket is None:
            prev_bucket = {"dateISO": prev_local_date, "rainMm": 0.0, "hoursCount": 0}
            daily_map[prev_local_date] = prev_bucket

        prev_bucket["rainMm"] = round_num(
            (num(prev_bucket.get("rainMm")) or 0.0) - (num(prev_row.get("rainMm")) or 0.0),
            4
        )
        if prev_bucket["rainMm"] < 0:
            prev_bucket["rainMm"] = 0.0

        if prev_local_date != new_local_date:
            prev_bucket["hoursCount"] = max(0, int(num(prev_bucket.get("hoursCount")) or 0) - 1)

    bucket["rainMm"] = round_num((num(bucket.get("rainMm")) or 0.0) + (num(new_row.get("rainMm")) or 0.0), 4)
    if bucket["rainMm"] < 0:
        bucket["rainMm"] = 0.0

    if not prev_row or prev_row["fileTimestampUtc"] != new_row["fileTimestampUtc"]:
        bucket["hoursCount"] = int(num(bucket.get("hoursCount")) or 0) + 1

    daily30 = trim_daily30_map(daily_map, new_dt)
    return last24, daily30


def get_parent_latest_file_timestamp(existing_parent):
    if not isinstance(existing_parent, dict):
        return None

    history_meta = existing_parent.get("mrmsHistoryMeta") or {}
    current_ts = str(history_meta.get("latestFileTimestampUtc") or "").strip()
    if current_ts:
        return current_ts

    latest = existing_parent.get("mrmsHourlyLatest") or {}
    current_ts = str(latest.get("fileTimestampUtc") or "").strip()
    if current_ts:
        return current_ts

    return None


def get_authoritative_latest_saved_hour(field_id, existing_parent=None):
    db = get_db()
    parent_latest_ts = get_parent_latest_file_timestamp(existing_parent or {})
    hourly_latest_ts = None

    try:
        latest_docs = list(
            db.collection(MRMS_PARENT_COLLECTION)
            .document(field_id)
            .collection(MRMS_HOURLY_SUBCOLLECTION)
            .order_by("fileTimestampUtc", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        if latest_docs:
            latest_doc = latest_docs[0].to_dict() or {}
            hourly_latest_ts = str(latest_doc.get("fileTimestampUtc") or "").strip() or None
    except Exception as e:
        print(f"[MRMS Latest Hour Lookup] fieldId={field_id} error={e}", flush=True)

    authoritative_ts = hourly_latest_ts or parent_latest_ts

    source = "none"
    if hourly_latest_ts:
        source = "hourly_subcollection"
    elif parent_latest_ts:
        source = "parent_fallback"

    return {
        "authoritativeLatestHourUtc": authoritative_ts,
        "hourlyLatestHourUtc": hourly_latest_ts,
        "parentLatestHourUtc": parent_latest_ts,
        "source": source,
    }


def get_next_needed_hour_for_field(field_id, existing_parent, latest_available_hour_utc=None):
    latest_info = get_authoritative_latest_saved_hour(
        field_id=field_id,
        existing_parent=existing_parent,
    )
    authoritative_ts = latest_info.get("authoritativeLatestHourUtc")

    if authoritative_ts:
        current_dt = floor_to_hour_utc(parse_iso_utc(authoritative_ts))
        return {
            "targetHourUtc": iso_utc(current_dt + timedelta(hours=1)),
            "reason": "next_after_authoritative_saved_hour",
            "currentLatestHourUtc": iso_utc(current_dt),
            "currentLatestHourSource": latest_info.get("source"),
            "hourlyLatestHourUtc": latest_info.get("hourlyLatestHourUtc"),
            "parentLatestHourUtc": latest_info.get("parentLatestHourUtc"),
        }

    if latest_available_hour_utc:
        latest_dt = floor_to_hour_utc(parse_iso_utc(latest_available_hour_utc))
        return {
            "targetHourUtc": iso_utc(latest_dt),
            "reason": "seed_with_latest_available",
            "currentLatestHourUtc": None,
            "currentLatestHourSource": "none",
            "hourlyLatestHourUtc": None,
            "parentLatestHourUtc": latest_info.get("parentLatestHourUtc"),
        }

    return {
        "targetHourUtc": None,
        "reason": "no_known_target",
        "currentLatestHourUtc": None,
        "currentLatestHourSource": "none",
        "hourlyLatestHourUtc": latest_info.get("hourlyLatestHourUtc"),
        "parentLatestHourUtc": latest_info.get("parentLatestHourUtc"),
    }


def maybe_prune_old_hourly_docs(field_id, anchor_hour_utc):
    db = get_db()
    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field_id)
    hourly_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION)
    anchor_dt = floor_to_hour_utc(parse_iso_utc(anchor_hour_utc))
    tz = get_app_tz()

    if anchor_dt.astimezone(tz).hour != 0:
        return {"deleted": 0, "skipped": True}

    cutoff_dt = anchor_dt - timedelta(days=KEEP_DAYS)
    cutoff_iso = iso_utc(cutoff_dt)

    old_docs = list(hourly_ref.where("fileTimestampUtc", "<", cutoff_iso).stream())
    if not old_docs:
        return {"deleted": 0, "skipped": False}

    deleted = 0
    b = db.batch()
    count = 0
    for doc in old_docs:
        b.delete(doc.reference)
        deleted += 1
        count += 1
        if count >= 400:
            b.commit()
            b = db.batch()
            count = 0
    if count > 0:
        b.commit()

    return {"deleted": deleted, "skipped": False}


def rebuild_last24_and_daily30(field_id):
    db = get_db()
    tz = get_app_tz()

    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field_id)
    hourly_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION)

    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)
    cutoff_iso = iso_utc(cutoff_dt)

    old_docs = list(hourly_ref.where("fileTimestampUtc", "<", cutoff_iso).stream())
    if old_docs:
        b = db.batch()
        count = 0
        for doc in old_docs:
            b.delete(doc.reference)
            count += 1
            if count >= 400:
                b.commit()
                b = db.batch()
                count = 0
        if count > 0:
            b.commit()

    docs = list(hourly_ref.where("fileTimestampUtc", ">=", cutoff_iso).stream())
    rows = []
    for doc in docs:
        d = doc.to_dict() or {}
        norm = normalize_hour_history_row(d, fallback_hour_key=doc.id)
        if not norm:
            continue
        rows.append(norm)

    rows.sort(key=lambda x: x["fileTimestampUtc"], reverse=True)
    last24 = rows[:LAST24_COUNT]

    daily_map = {}
    for row in rows:
        try:
            dt = datetime.fromisoformat(row["fileTimestampUtc"].replace("Z", "+00:00"))
            local_date = dt.astimezone(tz).date().isoformat()
        except Exception:
            continue

        bucket = daily_map.get(local_date)
        if bucket is None:
            bucket = {"dateISO": local_date, "rainMm": 0.0, "hoursCount": 0}
            daily_map[local_date] = bucket

        bucket["rainMm"] = round_num((num(bucket.get("rainMm")) or 0.0) + (num(row.get("rainMm")) or 0.0), 4)
        bucket["hoursCount"] = int(num(bucket.get("hoursCount")) or 0) + 1

    daily30 = [
        {
            "dateISO": v["dateISO"],
            "rainMm": round_num(v["rainMm"], 4),
            "hoursCount": v["hoursCount"],
        }
        for _, v in sorted(daily_map.items())
    ][-KEEP_DAYS:]

    return last24, daily30


def should_skip_normal_write(existing_parent, incoming_file_timestamp_utc):
    try:
        if not incoming_file_timestamp_utc:
            return False

        incoming_dt = parse_iso_utc(incoming_file_timestamp_utc)

        history_meta = existing_parent.get("mrmsHistoryMeta") or {}
        current_ts = str(history_meta.get("latestFileTimestampUtc") or "").strip()

        if not current_ts:
            latest = existing_parent.get("mrmsHourlyLatest") or {}
            current_ts = str(latest.get("fileTimestampUtc") or "").strip()

        if not current_ts:
            return False

        current_dt = parse_iso_utc(current_ts)

        if incoming_dt < current_dt:
            return True

        return False

    except Exception as e:
        print(
            f"[MRMS Skip Check Error] incoming={incoming_file_timestamp_utc} "
            f"error={e}",
            flush=True
        )
        return False


def write_field_hour(parent_ref, hour_ref, field, meta, mode, radius_miles, result):
    history_entry = build_hour_history_entry(meta, mode, radius_miles, result)

    parent_snap = parent_ref.get()
    existing_parent = parent_snap.to_dict() or {}

    previous_hour_entry = None
    previous_hour_snap = hour_ref.get()
    if previous_hour_snap.exists:
        previous_hour_entry = previous_hour_snap.to_dict() or {}

    hour_ref.set(history_entry, merge=True)

    history_meta = existing_parent.get("mrmsHistoryMeta") or {}
    existing_daily30 = existing_parent.get("mrmsDailySeries30d") or []

    needs_rebuild = (
        not bool(history_meta.get("fullBackfillComplete", False))
        or not isinstance(existing_daily30, list)
        or len(existing_daily30) < 5
    )

    if needs_rebuild:
        last24, daily30 = rebuild_last24_and_daily30(field["id"])
        rollup_mode = "full_rebuild_pre_backfill"
    else:
        rollup_mode = "incremental"
        try:
            last24, daily30 = build_incremental_rollups(
                existing_parent=existing_parent,
                new_history_entry=history_entry,
                previous_hour_entry=previous_hour_entry,
            )
        except Exception as e:
            print(f"[MRMS Incremental Rollup Fallback] field={field['id']} error={e}", flush=True)
            last24, daily30 = rebuild_last24_and_daily30(field["id"])
            rollup_mode = "full_rebuild_fallback"

    existing_state = {
        "historyMeta": existing_parent.get("mrmsHistoryMeta") or {},
    }

    parent_payload = build_latest_payload(
        field=field,
        meta=meta,
        mode=mode,
        radius_miles=radius_miles,
        result=result,
        last24=last24,
        daily30=daily30,
        existing_state=existing_state,
    )
    parent_payload["mrmsHistoryMeta"]["lastRollupMode"] = rollup_mode

    parent_ref.set(parent_payload, merge=True)

    try:
        maybe_prune_old_hourly_docs(field["id"], meta["fileTimestampUtc"])
    except Exception as e:
        print(f"[MRMS Prune] failed for {field['id']}: {e}", flush=True)


def queue_job_exists_active(doc_id):
    db = get_db()
    snap = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION).document(doc_id).get()
    if not snap.exists:
        return False
    data = snap.to_dict() or {}
    return str(data.get("status") or "").strip().lower() in {"queued", "running"}


def field_needs_full_backfill(field_id):
    state = get_field_mrms_state(field_id)

    if not state.get("docExists"):
        return True

    if not state.get("hasAnyHistory"):
        return True

    return not bool(state.get("fullBackfillComplete", False))


def delete_queue_jobs_for_field(field_id):
    db = get_db()
    coll = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION)
    deleted = 0

    docs = list(coll.where("fieldId", "==", field_id).stream())
    if not docs:
        return 0

    b = db.batch()
    count = 0
    for doc in docs:
        b.delete(doc.reference)
        deleted += 1
        count += 1
        if count >= 400:
            b.commit()
            b = db.batch()
            count = 0
    if count > 0:
        b.commit()

    return deleted


def enqueue_backfill(field_id, days=30, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    db = get_db()
    field = get_field_by_id(field_id)
    if not field:
        raise RuntimeError("Field not found")

    days = int(clamp(days, 1, 30))
    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    total_hours = days * 24
    chunk_hours = int(clamp(DEFAULT_FULL_BACKFILL_CHUNK_HOURS, 1, 168))
    doc_id = full_backfill_job_id(field_id)
    ref = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION).document(doc_id)

    if queue_job_exists_active(doc_id):
        return {
            "fieldId": field_id,
            "fieldName": field.get("name"),
            "status": "already_active",
            "jobType": "full_backfill",
            "days": days,
            "mode": mode,
            "radiusMiles": radius_miles,
        }

    ref.set({
        "jobType": "full_backfill",
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "status": "queued",
        "days": days,
        "mode": mode,
        "radiusMiles": radius_miles,
        "hoursTotal": total_hours,
        "hoursDone": 0,
        "chunkHours": chunk_hours,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "startedAt": None,
        "finishedAt": None,
        "error": None,
        "attempts": firestore.Increment(1),
        "lastChunkSummary": None,
    }, merge=True)

    return {
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "status": "queued",
        "jobType": "full_backfill",
        "days": days,
        "mode": mode,
        "radiusMiles": radius_miles,
        "hoursTotal": total_hours,
        "chunkHours": chunk_hours,
    }


def delete_hourly_history_for_field(field_id):
    db = get_db()
    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field_id)
    hourly_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION)

    deleted = 0
    while True:
        docs = list(hourly_ref.limit(400).stream())
        if not docs:
            break

        b = db.batch()
        for doc in docs:
            b.delete(doc.reference)
            deleted += 1
        b.commit()

        if len(docs) < 400:
            break

    return deleted


def reset_field_history_for_location_change(field, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    db = get_db()
    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field["id"])

    deleted_hourly_docs = delete_hourly_history_for_field(field["id"])
    deleted_queue_jobs = delete_queue_jobs_for_field(field["id"])

    parent_ref.set({
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "farmId": field.get("farmId"),
        "farmName": field.get("farmName"),
        "location": {"lat": field["lat"], "lng": field["lng"]},
        "mrmsHourlyLatest": firestore.DELETE_FIELD,
        "mrmsHourlyLast24": [],
        "mrmsDailySeries30d": [],
        "mrmsHistoryMeta": {
            "subcollection": MRMS_HOURLY_SUBCOLLECTION,
            "keepDays": KEEP_DAYS,
            "keepHours": KEEP_HOURS,
            "units": "mm",
            "latestFileTimestampUtc": None,
            "latestSelectedProduct": None,
            "last24Count": 0,
            "daily30Count": 0,
            "fullBackfillComplete": False,
            "backfillCompletedAt": None,
            "latestRepairEnqueuedAt": None,
            "locationResetAt": firestore.SERVER_TIMESTAMP,
        },
        "mrmsLastUpdatedAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    queued = enqueue_backfill(
        field_id=field["id"],
        days=KEEP_DAYS,
        mode=mode,
        radius_miles=radius_miles,
    )

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "deletedHourlyDocs": deleted_hourly_docs,
        "deletedQueueJobs": deleted_queue_jobs,
        "queuedBackfill": queued,
    }


def maybe_handle_location_change(field, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    state = get_field_mrms_state(field["id"])
    if not state.get("docExists"):
        return {
            "locationChanged": False,
            "action": "no_parent_doc",
        }

    old_location = state.get("location")
    new_location = {"lat": field["lat"], "lng": field["lng"]}

    if not old_location:
        return {
            "locationChanged": False,
            "action": "no_saved_location",
        }

    if locations_match(old_location, new_location):
        return {
            "locationChanged": False,
            "action": "location_unchanged",
        }

    reset = reset_field_history_for_location_change(
        field=field,
        mode=mode,
        radius_miles=radius_miles,
    )
    return {
        "locationChanged": True,
        "action": "reset_and_requeued",
        "oldLocation": old_location,
        "newLocation": new_location,
        "reset": reset,
    }


def enqueue_gap_repair(field_id, start_hour_utc, end_hour_utc, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    db = get_db()
    field = get_field_by_id(field_id)
    if not field:
        raise RuntimeError("Field not found")

    if field_needs_full_backfill(field_id):
        return enqueue_backfill(
            field_id=field_id,
            days=KEEP_DAYS,
            mode=mode,
            radius_miles=radius_miles,
        )

    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    start_dt = floor_to_hour_utc(parse_iso_utc(start_hour_utc))
    end_dt = floor_to_hour_utc(parse_iso_utc(end_hour_utc))
    if end_dt < start_dt:
        raise RuntimeError("endHourUtc must be >= startHourUtc")

    total_hours = int(((end_dt - start_dt).total_seconds() // 3600) + 1)

    doc_id = repair_job_id(field_id, iso_utc(start_dt), iso_utc(end_dt))
    ref = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION).document(doc_id)

    if queue_job_exists_active(doc_id):
        return {
            "fieldId": field_id,
            "fieldName": field.get("name"),
            "status": "already_active",
            "jobType": "repair_gap",
            "startHourUtc": iso_utc(start_dt),
            "endHourUtc": iso_utc(end_dt),
        }

    ref.set({
        "jobType": "repair_gap",
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "status": "queued",
        "days": None,
        "mode": mode,
        "radiusMiles": radius_miles,
        "startHourUtc": iso_utc(start_dt),
        "endHourUtc": iso_utc(end_dt),
        "repairCursorHourUtc": iso_utc(start_dt),
        "hoursTotal": total_hours,
        "hoursDone": 0,
        "chunkHours": int(clamp(DEFAULT_REPAIR_CHUNK_HOURS, 1, 168)),
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "startedAt": None,
        "finishedAt": None,
        "error": None,
        "attempts": firestore.Increment(1),
        "lastChunkSummary": None,
    }, merge=True)

    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field_id)
    parent_ref.set({
        "mrmsHistoryMeta": {
            "latestRepairEnqueuedAt": firestore.SERVER_TIMESTAMP
        }
    }, merge=True)

    return {
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "status": "queued",
        "jobType": "repair_gap",
        "startHourUtc": iso_utc(start_dt),
        "endHourUtc": iso_utc(end_dt),
        "mode": mode,
        "radiusMiles": radius_miles,
        "hoursTotal": total_hours,
        "chunkHours": int(clamp(DEFAULT_REPAIR_CHUNK_HOURS, 1, 168)),
    }


def enqueue_backfill_all(days=30, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    fields = load_active_fields_for_batch()

    days = int(clamp(days, 1, 30))
    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    queued = 0
    failed = 0
    failures = []

    for field in fields:
        try:
            out = enqueue_backfill(
                field_id=field["id"],
                days=days,
                mode=mode,
                radius_miles=radius_miles,
            )
            if out.get("status") in {"queued", "already_active"}:
                queued += 1
        except Exception as e:
            failed += 1
            failures.append({
                "fieldId": field["id"],
                "fieldName": field.get("name"),
                "error": str(e),
            })

    return {
        "totalActiveFields": len(fields),
        "queuedOrActive": queued,
        "failed": failed,
        "queueCollection": MRMS_BACKFILL_QUEUE_COLLECTION,
        "failures": failures[:50],
    }


def enqueue_repair_all(lookback_hours=DEFAULT_REPAIR_LOOKBACK_HOURS, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    fields = load_active_fields_for_batch()

    lookback_hours = int(clamp(lookback_hours, 1, KEEP_HOURS))
    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    meta = get_cached_dataset()
    latest_hour_utc = meta["fileTimestampUtc"]

    queued = 0
    no_gap = 0
    already_active = 0
    failed = 0
    full_backfill_queued = 0
    failures = []

    for field in fields:
        try:
            if field_needs_full_backfill(field["id"]):
                out = enqueue_backfill(
                    field_id=field["id"],
                    days=KEEP_DAYS,
                    mode=mode,
                    radius_miles=radius_miles,
                )
                if out.get("status") in {"queued", "already_active"}:
                    full_backfill_queued += 1
                continue

            missing = find_missing_recent_hours(
                field_id=field["id"],
                latest_hour_utc=latest_hour_utc,
                lookback_hours=lookback_hours,
            )

            if not missing:
                no_gap += 1
                continue

            out = enqueue_gap_repair(
                field_id=field["id"],
                start_hour_utc=missing[0],
                end_hour_utc=missing[-1],
                mode=mode,
                radius_miles=radius_miles,
            )
            if out.get("status") == "already_active":
                already_active += 1
            else:
                queued += 1
        except Exception as e:
            failed += 1
            failures.append({
                "fieldId": field["id"],
                "fieldName": field.get("name"),
                "error": str(e),
            })

    return {
        "lookbackHours": lookback_hours,
        "latestHourUtc": latest_hour_utc,
        "totalActiveFields": len(fields),
        "queued": queued,
        "fullBackfillQueued": full_backfill_queued,
        "alreadyActive": already_active,
        "noGap": no_gap,
        "failed": failed,
        "queueCollection": MRMS_BACKFILL_QUEUE_COLLECTION,
        "failures": failures[:50],
    }


def maybe_auto_enqueue_backfill(field, mode, radius_miles):
    if not field_needs_full_backfill(field["id"]):
        return {"fieldId": field["id"], "queued": False, "reason": "full_backfill_complete"}

    out = enqueue_backfill(
        field_id=field["id"],
        days=KEEP_DAYS,
        mode=mode,
        radius_miles=radius_miles,
    )
    return {"fieldId": field["id"], "queued": True, "result": out}


def find_missing_recent_hours(field_id, latest_hour_utc, lookback_hours):
    db = get_db()
    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field_id)
    hourly_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION)

    latest_dt = floor_to_hour_utc(parse_iso_utc(latest_hour_utc))
    lookback_hours = int(clamp(lookback_hours, 1, 168))

    start_dt = latest_dt - timedelta(hours=(lookback_hours - 1))
    start_iso = iso_utc(start_dt)
    end_iso = iso_utc(latest_dt)

    docs = list(
        hourly_ref.where("fileTimestampUtc", ">=", start_iso)
        .where("fileTimestampUtc", "<=", end_iso)
        .stream()
    )

    existing = set()
    for doc in docs:
        d = doc.to_dict() or {}
        ts = str(d.get("fileTimestampUtc") or "")
        if ts:
            existing.add(ts)

    expected = []
    for i in range(lookback_hours):
        dt = start_dt + timedelta(hours=i)
        expected.append(iso_utc(dt))

    missing = [x for x in expected if x not in existing]
    return missing


def maybe_auto_enqueue_gap_repair(field, mode, radius_miles, latest_hour_utc):
    state = get_field_mrms_state(field["id"])

    if not state.get("fullBackfillComplete"):
        return {
            "fieldId": field["id"],
            "queued": False,
            "reason": "waiting_on_full_backfill"
        }

    missing = find_missing_recent_hours(
        field_id=field["id"],
        latest_hour_utc=latest_hour_utc,
        lookback_hours=DEFAULT_REPAIR_LOOKBACK_HOURS,
    )

    if not missing:
        return {"fieldId": field["id"], "queued": False, "reason": "no_gap"}

    start_hour = missing[0]
    end_hour = missing[-1]

    out = enqueue_gap_repair(
        field_id=field["id"],
        start_hour_utc=start_hour,
        end_hour_utc=end_hour,
        mode=mode,
        radius_miles=radius_miles,
    )
    return {
        "fieldId": field["id"],
        "queued": True,
        "missingCount": len(missing),
        "result": out,
    }


def run_batch_cache(mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    fields = load_active_fields_for_batch()
    total = len(fields)

    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    latest_meta = get_cached_dataset()
    latest_available_hour_utc = latest_meta["fileTimestampUtc"]
    latest_available_dt = floor_to_hour_utc(parse_iso_utc(latest_available_hour_utc))

    ok = 0
    skipped_stale = 0
    skipped_up_to_date = 0
    skipped_missing_exact_hour = 0
    fail = 0
    failures = []
    auto_enqueued_backfills = 0
    auto_enqueue_backfill_failures = 0
    auto_enqueued_repairs = 0
    auto_enqueue_repair_failures = 0
    auto_location_resets = 0
    auto_location_reset_failures = 0

    for field in fields:
        try:
            location_changed_this_run = False

            try:
                location_change = maybe_handle_location_change(field, mode, radius_miles)
                if location_change.get("locationChanged"):
                    auto_location_resets += 1
                    location_changed_this_run = True
            except Exception as e:
                auto_location_reset_failures += 1
                print(f"[AutoLocationReset] failed for {field['id']}: {e}", flush=True)

            try:
                auto = maybe_auto_enqueue_backfill(field, mode, radius_miles)
                if auto.get("queued"):
                    auto_enqueued_backfills += 1
            except Exception as e:
                auto_enqueue_backfill_failures += 1
                print(f"[AutoEnqueueFull] failed for {field['id']}: {e}", flush=True)

            parent_ref = get_db().collection(MRMS_PARENT_COLLECTION).document(field["id"])
            existing_parent_snap = parent_ref.get()
            existing_parent = existing_parent_snap.to_dict() or {}

            target_info = get_next_needed_hour_for_field(
                field_id=field["id"],
                existing_parent=existing_parent,
                latest_available_hour_utc=latest_available_hour_utc,
            )
            target_hour_utc = target_info.get("targetHourUtc")

            if not target_hour_utc:
                print(
                    f"[MRMS Hourly No Target] fieldId={field['id']} fieldName={field.get('name')} "
                    f"reason={target_info.get('reason')}",
                    flush=True
                )
                skipped_missing_exact_hour += 1
                continue

            target_dt = floor_to_hour_utc(parse_iso_utc(target_hour_utc))

            if target_dt > latest_available_dt:
                print(
                    f"[MRMS Hourly Up To Date] fieldId={field['id']} fieldName={field.get('name')} "
                    f"current={target_info.get('currentLatestHourUtc')} "
                    f"currentSource={target_info.get('currentLatestHourSource')} "
                    f"hourlyLatest={target_info.get('hourlyLatestHourUtc')} "
                    f"parentLatest={target_info.get('parentLatestHourUtc')} "
                    f"nextNeeded={target_hour_utc} latestAvailable={latest_available_hour_utc}",
                    flush=True
                )
                skipped_up_to_date += 1

                try:
                    auto_rep = maybe_auto_enqueue_gap_repair(
                        field=field,
                        mode=mode,
                        radius_miles=radius_miles,
                        latest_hour_utc=latest_available_hour_utc,
                    )
                    if auto_rep.get("queued"):
                        auto_enqueued_repairs += 1
                except Exception as e:
                    auto_enqueue_repair_failures += 1
                    print(f"[AutoEnqueueRepair] failed for {field['id']}: {e}", flush=True)

                continue

            meta = get_cached_dataset_for_exact_hour(target_dt, latest_meta=latest_meta)

            if not meta:
                print(
                    f"[MRMS Hourly Exact Hour Missing → FALLBACK TO LATEST] fieldId={field['id']} "
                    f"target={target_hour_utc} usingLatest={latest_available_hour_utc}",
                    flush=True
                )

                # fallback to latest instead of stalling
                meta = latest_meta

            if (
                not location_changed_this_run
                and should_skip_normal_write(existing_parent, meta["fileTimestampUtc"])
            ):
                print(
                    f"[MRMS Stale Skip] fieldId={field['id']} fieldName={field.get('name')} "
                    f"incoming={meta['fileTimestampUtc']} current={target_info.get('currentLatestHourUtc')} "
                    f"currentSource={target_info.get('currentLatestHourSource')} "
                    f"hourlyLatest={target_info.get('hourlyLatestHourUtc')} "
                    f"parentLatest={target_info.get('parentLatestHourUtc')} "
                    f"target={target_hour_utc}",
                    flush=True
                )
                skipped_stale += 1

                try:
                    auto_rep = maybe_auto_enqueue_gap_repair(
                        field=field,
                        mode=mode,
                        radius_miles=radius_miles,
                        latest_hour_utc=latest_available_hour_utc,
                    )
                    if auto_rep.get("queued"):
                        auto_enqueued_repairs += 1
                except Exception as e:
                    auto_enqueue_repair_failures += 1
                    print(f"[AutoEnqueueRepair] failed for {field['id']}: {e}", flush=True)

                continue

            da = meta["dataArray"]
            result = (
                build_single_result(da, field["lat"], field["lng"])
                if mode == "single"
                else build_weighted_result(da, field["lat"], field["lng"], radius_miles)
            )

            hour_id = hour_doc_id_from_iso(meta["fileTimestampUtc"])
            hour_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION).document(hour_id)
            write_field_hour(parent_ref, hour_ref, field, meta, mode, radius_miles, result)
            ok += 1

            try:
                auto_rep = maybe_auto_enqueue_gap_repair(
                    field=field,
                    mode=mode,
                    radius_miles=radius_miles,
                    latest_hour_utc=latest_available_hour_utc,
                )
                if auto_rep.get("queued"):
                    auto_enqueued_repairs += 1
            except Exception as e:
                auto_enqueue_repair_failures += 1
                print(f"[AutoEnqueueRepair] failed for {field['id']}: {e}", flush=True)

        except Exception as e:
            fail += 1
            failures.append({
                "fieldId": field["id"],
                "fieldName": field.get("name"),
                "error": str(e),
            })

    return {
        "total": total,
        "ok": ok,
        "skippedStaleOrDuplicateLatest": skipped_stale,
        "skippedUpToDateNoNewHour": skipped_up_to_date,
        "skippedMissingExactHour": skipped_missing_exact_hour,
        "fail": fail,
        "collection": MRMS_PARENT_COLLECTION,
        "selectedProduct": latest_meta["selectedProduct"],
        "selectedKey": latest_meta["selectedKey"].replace(f"{AWS_BUCKET}/", ""),
        "fileTimestampUtc": latest_meta["fileTimestampUtc"],
        "cacheHit": latest_meta["cacheHit"],
        "autoLocationResets": auto_location_resets,
        "autoLocationResetFailures": auto_location_reset_failures,
        "autoEnqueuedBackfills": auto_enqueued_backfills,
        "autoEnqueueBackfillFailures": auto_enqueue_backfill_failures,
        "autoEnqueuedRepairJobs": auto_enqueued_repairs,
        "autoEnqueueRepairFailures": auto_enqueue_repair_failures,
        "repairLookbackHours": DEFAULT_REPAIR_LOOKBACK_HOURS,
        "failures": failures[:25],
    }


def finalize_field_parent_from_hourly(field_id, mark_full_backfill_complete=False):
    field = get_field_by_id(field_id)
    if not field:
        raise RuntimeError("Field not found")

    db = get_db()
    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field["id"])

    latest_docs = list(
        parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION)
        .order_by("fileTimestampUtc", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )

    if not latest_docs:
        return {
            "fieldId": field["id"],
            "fieldName": field.get("name"),
            "updatedParent": False,
            "reason": "no_hourly_docs",
        }

    latest_doc = latest_docs[0].to_dict() or {}
    latest_rain_mm = get_doc_rain_mm(latest_doc)

    latest_meta = {
        "selectedProduct": latest_doc.get("selectedProduct"),
        "selectedKey": latest_doc.get("selectedKey"),
        "fileTimestampUtc": latest_doc.get("fileTimestampUtc"),
        "variableName": latest_doc.get("variableName"),
        "cacheHit": False,
        "io": None,
        "checkedProducts": None,
    }
    latest_result = {
        "weightedHourlyRainMm": latest_rain_mm,
        "attemptedPointCount": latest_doc.get("attemptedPointCount"),
        "successfulPointCount": latest_doc.get("successfulPointCount"),
        "samples": latest_doc.get("samples"),
        "hourlyRainMm": latest_rain_mm,
        "nearestGridLatitude": latest_doc.get("nearestGridLatitude"),
        "nearestGridLongitude": latest_doc.get("nearestGridLongitude"),
        "nearestGridLongitudeRaw": latest_doc.get("nearestGridLongitudeRaw"),
        "queryLongitudeUsed": latest_doc.get("queryLongitudeUsed"),
        "longitudeMode": latest_doc.get("longitudeMode"),
    }

    last24, daily30 = rebuild_last24_and_daily30(field["id"])
    existing_state = get_field_mrms_state(field["id"])
    parent_payload = build_latest_payload(
        field=field,
        meta=latest_meta,
        mode=latest_doc.get("mode") or "weighted",
        radius_miles=latest_doc.get("radiusMiles") or DEFAULT_RADIUS_MILES,
        result=latest_result,
        last24=last24,
        daily30=daily30,
        existing_state=existing_state,
    )

    if mark_full_backfill_complete:
        parent_payload["mrmsHistoryMeta"]["fullBackfillComplete"] = True
        parent_payload["mrmsHistoryMeta"]["backfillCompletedAt"] = firestore.SERVER_TIMESTAMP

    parent_ref.set(parent_payload, merge=True)

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "updatedParent": True,
        "latestFileTimestampUtc": latest_doc.get("fileTimestampUtc"),
        "fullBackfillComplete": bool(parent_payload["mrmsHistoryMeta"].get("fullBackfillComplete")),
    }


def backfill_field(field_id, days=30, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    field = get_field_by_id(field_id)
    if not field:
        raise RuntimeError("Field not found")

    days = int(clamp(days, 1, 30))
    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    now_utc = datetime.now(timezone.utc)
    this_hour = floor_to_hour_utc(now_utc)
    total_hours = days * 24

    parent_ref = get_db().collection(MRMS_PARENT_COLLECTION).document(field["id"])

    ok = 0
    skipped = 0
    fail = 0
    failures = []

    for i in range(total_hours):
        target_dt = this_hour - timedelta(hours=i)
        product, s3_key, checked = pick_best_key_for_hour(target_dt)

        if not s3_key:
            skipped += 1
            continue

        try:
            meta = get_dataset_for_key_uncached(product, s3_key)
            meta["checkedProducts"] = checked
            da = meta["dataArray"]

            result = build_single_result(da, field["lat"], field["lng"]) if mode == "single" else build_weighted_result(da, field["lat"], field["lng"], radius_miles)

            hour_id = hour_doc_id_from_iso(meta["fileTimestampUtc"])
            hour_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION).document(hour_id)
            write_field_hour(parent_ref, hour_ref, field, meta, mode, radius_miles, result)
            ok += 1

            try:
                if meta.get("io") is not None:
                    meta["io"].close()
            except Exception:
                pass

        except Exception as e:
            fail += 1
            failures.append({
                "targetHourUtc": iso_utc(target_dt),
                "error": str(e),
            })

    finalized = finalize_field_parent_from_hourly(field["id"], mark_full_backfill_complete=True)

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "days": days,
        "mode": mode,
        "radiusMiles": radius_miles,
        "totalHours": total_hours,
        "ok": ok,
        "skipped": skipped,
        "fail": fail,
        "finalized": finalized,
        "failures": failures[:25],
    }


def process_next_backfill():
    db = get_db()
    queued = list(
        db.collection(MRMS_BACKFILL_QUEUE_COLLECTION)
        .where("status", "==", "queued")
        .limit(1)
        .stream()
    )

    if not queued:
        return {
            "processed": False,
            "reason": "no_queued_jobs",
        }

    job_doc = queued[0]
    job = job_doc.to_dict() or {}
    job_type = str(job.get("jobType") or "").strip().lower()

    if job_type == "repair_gap":
        return process_next_repair_gap()

    field_id = job.get("fieldId")
    mode = job.get("mode") or "weighted"
    radius_miles = num(job.get("radiusMiles")) or DEFAULT_RADIUS_MILES
    days = int(clamp(job.get("days"), 1, 30))
    chunk_hours = int(clamp(job.get("chunkHours") or DEFAULT_FULL_BACKFILL_CHUNK_HOURS, 1, 168))

    field = get_field_by_id(field_id)
    if not field:
        job_doc.reference.set({
            "status": "failed",
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "finishedAt": firestore.SERVER_TIMESTAMP,
            "error": "Field not found",
        }, merge=True)
        return {
            "processed": False,
            "jobId": job_doc.id,
            "status": "failed",
            "reason": "field_not_found",
        }

    now_utc = datetime.now(timezone.utc)
    this_hour = floor_to_hour_utc(now_utc)
    total_hours = days * 24

    hours_done = int(num(job.get("hoursDone")) or 0)
    if hours_done < 0:
        hours_done = 0
    if hours_done >= total_hours:
        job_doc.reference.set({
            "status": "done",
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "finishedAt": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        finalized = finalize_field_parent_from_hourly(field_id, mark_full_backfill_complete=True)
        return {
            "processed": True,
            "jobId": job_doc.id,
            "status": "done",
            "fieldId": field_id,
            "fieldName": field.get("name"),
            "hoursDone": hours_done,
            "hoursTotal": total_hours,
            "finalized": finalized,
        }

    job_doc.reference.set({
        "status": "running",
        "startedAt": firestore.SERVER_TIMESTAMP if not job.get("startedAt") else job.get("startedAt"),
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field["id"])

    ok = 0
    skipped = 0
    fail = 0
    failures = []
    processed_hours = 0

    for offset in range(hours_done, min(hours_done + chunk_hours, total_hours)):
        target_dt = this_hour - timedelta(hours=offset)
        product, s3_key, checked = pick_best_key_for_hour(target_dt)

        if not s3_key:
            skipped += 1
            processed_hours += 1
            continue

        try:
            meta = get_dataset_for_key_uncached(product, s3_key)
            meta["checkedProducts"] = checked
            da = meta["dataArray"]

            result = build_single_result(da, field["lat"], field["lng"]) if mode == "single" else build_weighted_result(da, field["lat"], field["lng"], radius_miles)

            hour_id = hour_doc_id_from_iso(meta["fileTimestampUtc"])
            hour_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION).document(hour_id)
            write_field_hour(parent_ref, hour_ref, field, meta, mode, radius_miles, result)
            ok += 1
            processed_hours += 1

            try:
                if meta.get("io") is not None:
                    meta["io"].close()
            except Exception:
                pass

        except Exception as e:
            fail += 1
            processed_hours += 1
            failures.append({
                "targetHourUtc": iso_utc(target_dt),
                "error": str(e),
            })

    new_hours_done = hours_done + processed_hours
    finished = new_hours_done >= total_hours

    last_chunk_summary = {
        "processedHours": processed_hours,
        "ok": ok,
        "skipped": skipped,
        "fail": fail,
        "finished": finished,
        "updatedAt": iso_utc(datetime.now(timezone.utc)),
    }

    update_payload = {
        "hoursDone": new_hours_done,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "lastChunkSummary": last_chunk_summary,
        "error": None if fail == 0 else failures[:10],
        "status": "done" if finished else "queued",
    }

    if finished:
        update_payload["finishedAt"] = firestore.SERVER_TIMESTAMP

    job_doc.reference.set(update_payload, merge=True)

    finalized = None
    if finished:
        finalized = finalize_field_parent_from_hourly(field_id, mark_full_backfill_complete=True)

    return {
        "processed": True,
        "jobId": job_doc.id,
        "status": "done" if finished else "queued",
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "mode": mode,
        "radiusMiles": radius_miles,
        "hoursDoneBefore": hours_done,
        "hoursDoneAfter": new_hours_done,
        "hoursTotal": total_hours,
        "chunkHours": chunk_hours,
        "chunk": last_chunk_summary,
        "finalized": finalized,
        "failures": failures[:25],
    }


def process_repair_all(max_fields=None, max_minutes=None):
    db = get_db()

    if max_fields is None:
        try:
            max_fields = int(request.args.get("maxFields") or 0)
        except Exception:
            max_fields = 0
    if max_minutes is None:
        try:
            max_minutes = float(request.args.get("maxMinutes") or 0)
        except Exception:
            max_minutes = 0

    max_fields = max(0, int(max_fields or 0))
    max_minutes = max(0.0, float(max_minutes or 0))

    started = datetime.now(timezone.utc)
    processed = 0
    failed = 0
    results = []

    while True:
        if max_fields and processed >= max_fields:
            break

        if max_minutes:
            elapsed_minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60.0
            if elapsed_minutes >= max_minutes:
                break

        out = process_next_repair_gap()
        if not out.get("processed"):
            break

        processed += 1
        if out.get("status") == "failed":
            failed += 1

        results.append({
            "fieldId": out.get("fieldId"),
            "fieldName": out.get("fieldName"),
            "status": out.get("status"),
            "hoursDoneAfter": out.get("hoursDoneAfter"),
            "hoursTotal": out.get("hoursTotal"),
            "finalized": out.get("finalized"),
        })

    remaining_queued = 0
    try:
        remaining = list(
            db.collection(MRMS_BACKFILL_QUEUE_COLLECTION)
            .where("status", "==", "queued")
            .limit(500)
            .stream()
        )
        for doc in remaining:
            job = doc.to_dict() or {}
            if str(job.get("jobType") or "").strip().lower() == "repair_gap":
                remaining_queued += 1
    except Exception:
        pass

    return {
        "processed": processed,
        "failed": failed,
        "remainingQueuedRepairJobs": remaining_queued,
        "results": results[-50:],
    }


def process_next_repair_gap():
    db = get_db()
    queued = list(
        db.collection(MRMS_BACKFILL_QUEUE_COLLECTION)
        .where("status", "==", "queued")
        .limit(25)
        .stream()
    )

    repair_doc = None
    repair_job = None
    for doc in queued:
        job = doc.to_dict() or {}
        if str(job.get("jobType") or "").strip().lower() == "repair_gap":
            repair_doc = doc
            repair_job = job
            break

    if not repair_doc:
        return {
            "processed": False,
            "reason": "no_queued_repair_gap_jobs",
        }

    field_id = repair_job.get("fieldId")
    mode = repair_job.get("mode") or "weighted"
    radius_miles = num(repair_job.get("radiusMiles")) or DEFAULT_RADIUS_MILES
    chunk_hours = int(clamp(repair_job.get("chunkHours") or DEFAULT_REPAIR_CHUNK_HOURS, 1, 168))
    start_hour_utc = repair_job.get("startHourUtc")
    end_hour_utc = repair_job.get("endHourUtc")
    cursor_hour_utc = repair_job.get("repairCursorHourUtc") or start_hour_utc

    field = get_field_by_id(field_id)
    if not field:
        repair_doc.reference.set({
            "status": "failed",
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "finishedAt": firestore.SERVER_TIMESTAMP,
            "error": "Field not found",
        }, merge=True)
        return {
            "processed": False,
            "jobId": repair_doc.id,
            "status": "failed",
            "reason": "field_not_found",
        }

    start_dt = floor_to_hour_utc(parse_iso_utc(start_hour_utc))
    end_dt = floor_to_hour_utc(parse_iso_utc(end_hour_utc))
    cursor_dt = floor_to_hour_utc(parse_iso_utc(cursor_hour_utc))

    if cursor_dt < start_dt:
        cursor_dt = start_dt

    if cursor_dt > end_dt:
        repair_doc.reference.set({
            "status": "done",
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "finishedAt": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        finalized = finalize_field_parent_from_hourly(field_id, mark_full_backfill_complete=False)
        return {
            "processed": True,
            "jobId": repair_doc.id,
            "status": "done",
            "fieldId": field_id,
            "fieldName": field.get("name"),
            "finalized": finalized,
        }

    repair_doc.reference.set({
        "status": "running",
        "startedAt": firestore.SERVER_TIMESTAMP if not repair_job.get("startedAt") else repair_job.get("startedAt"),
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field["id"])

    ok = 0
    skipped = 0
    fail = 0
    failures = []
    processed_hours = 0

    current_dt = cursor_dt
    final_dt_this_chunk = min(end_dt, cursor_dt + timedelta(hours=chunk_hours - 1))

    while current_dt <= final_dt_this_chunk:
        product, s3_key, checked = pick_best_key_for_hour(current_dt)

        if not s3_key:
            skipped += 1
            processed_hours += 1
            current_dt += timedelta(hours=1)
            continue

        try:
            meta = get_dataset_for_key_uncached(product, s3_key)
            meta["checkedProducts"] = checked
            da = meta["dataArray"]

            result = build_single_result(da, field["lat"], field["lng"]) if mode == "single" else build_weighted_result(da, field["lat"], field["lng"], radius_miles)

            hour_id = hour_doc_id_from_iso(meta["fileTimestampUtc"])
            hour_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION).document(hour_id)
            write_field_hour(parent_ref, hour_ref, field, meta, mode, radius_miles, result)
            ok += 1
            processed_hours += 1

            try:
                if meta.get("io") is not None:
                    meta["io"].close()
            except Exception:
                pass

        except Exception as e:
            fail += 1
            processed_hours += 1
            failures.append({
                "targetHourUtc": iso_utc(current_dt),
                "error": str(e),
            })

        current_dt += timedelta(hours=1)

    next_cursor = final_dt_this_chunk + timedelta(hours=1)
    finished = next_cursor > end_dt
    hours_done = int(num(repair_job.get("hoursDone")) or 0) + processed_hours

    last_chunk_summary = {
        "processedHours": processed_hours,
        "ok": ok,
        "skipped": skipped,
        "fail": fail,
        "finished": finished,
        "updatedAt": iso_utc(datetime.now(timezone.utc)),
    }

    update_payload = {
        "hoursDone": hours_done,
        "repairCursorHourUtc": iso_utc(next_cursor),
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "lastChunkSummary": last_chunk_summary,
        "error": None if fail == 0 else failures[:10],
        "status": "done" if finished else "queued",
    }

    if finished:
        update_payload["finishedAt"] = firestore.SERVER_TIMESTAMP

    repair_doc.reference.set(update_payload, merge=True)

    finalized = None
    if finished:
        finalized = finalize_field_parent_from_hourly(field_id, mark_full_backfill_complete=False)

    return {
        "processed": True,
        "jobId": repair_doc.id,
        "status": "done" if finished else "queued",
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "mode": mode,
        "radiusMiles": radius_miles,
        "startHourUtc": iso_utc(start_dt),
        "endHourUtc": iso_utc(end_dt),
        "cursorBefore": iso_utc(cursor_dt),
        "cursorAfter": iso_utc(next_cursor),
        "chunkHours": chunk_hours,
        "chunk": last_chunk_summary,
        "finalized": finalized,
        "failures": failures[:25],
    }


@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "FarmVista NOAA MRMS Pass2->Pass1 fallback rainfall service",
        "productPriority": PRODUCT_PRIORITY,
        "writesToCollection": MRMS_PARENT_COLLECTION,
        "historySubcollection": MRMS_HOURLY_SUBCOLLECTION,
        "queueCollection": MRMS_BACKFILL_QUEUE_COLLECTION,
        "repairLookbackHours": DEFAULT_REPAIR_LOOKBACK_HOURS,
        "fullBackfillChunkHours": DEFAULT_FULL_BACKFILL_CHUNK_HOURS,
        "repairChunkHours": DEFAULT_REPAIR_CHUNK_HOURS,
        "locationEpsilon": LOCATION_EPSILON,
        "rainUnit": "mm",
        "notes": {
            "normalWrites": "incremental parent updates (no full history reread)",
            "fullRebuilds": "used for finalize/backfill/repair completion only",
            "hourlyTargeting": "writes next needed hour per field using authoritative latest saved hourly doc first"
        },
        "routes": {
            "single": "/api/mrms-1h?lat=39.7898&lon=-91.2059",
            "weighted": "/api/mrms-1h?lat=39.7898&lon=-91.2059&radiusMiles=0.5&mode=weighted",
            "runBatchCache": "/run?mode=weighted&radiusMiles=0.5",
            "backfillField": "/backfill-field?fieldId=YOUR_FIELD_ID&days=30&mode=weighted&radiusMiles=0.5",
            "enqueueBackfill": "/enqueue-backfill?fieldId=YOUR_FIELD_ID&days=30&mode=weighted&radiusMiles=0.5",
            "enqueueBackfillAll": "/enqueue-backfill-all?days=30&mode=weighted&radiusMiles=0.5",
            "processNextBackfill": "/process-next-backfill",
            "queueStatus": "/queue-status",
            "repairField": "/repair-field?fieldId=YOUR_FIELD_ID&lookbackHours=24&mode=weighted&radiusMiles=0.5",
            "processNextRepair": "/process-next-repair",
            "fieldState": "/field-state?fieldId=YOUR_FIELD_ID",
        }
    })


@app.get("/health")
def health():
    return jsonify({
        "ok": IMPORT_ERROR is None,
        "importError": IMPORT_ERROR,
        "productPriority": PRODUCT_PRIORITY,
        "repairLookbackHours": DEFAULT_REPAIR_LOOKBACK_HOURS,
        "fullBackfillChunkHours": DEFAULT_FULL_BACKFILL_CHUNK_HOURS,
        "repairChunkHours": DEFAULT_REPAIR_CHUNK_HOURS,
        "locationEpsilon": LOCATION_EPSILON,
    })


@app.get("/api/mrms-1h")
def api_mrms_1h():
    try:
        ensure_runtime_ready()

        lat = num(request.args.get("lat"))
        lon = num(request.args.get("lon"))
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        if lat is None or lon is None:
            return jsonify({"ok": False, "error": "lat and lon are required"}), 400
        if mode not in {"single", "weighted"}:
            return jsonify({"ok": False, "error": "mode must be 'single' or 'weighted'"}), 400
        if radius_miles <= 0:
            return jsonify({"ok": False, "error": "radiusMiles must be > 0"}), 400

        meta = get_cached_dataset()
        da = meta["dataArray"]

        result = build_single_result(da, lat, lon) if mode == "single" else build_weighted_result(da, lat, lon, radius_miles)

        return jsonify({
            "ok": True,
            "source": "noaa-mrms-aws",
            "selectedProduct": meta["selectedProduct"],
            "selectedKey": meta["selectedKey"].replace(f"{AWS_BUCKET}/", ""),
            "fileTimestampUtc": meta["fileTimestampUtc"],
            "variableName": meta["variableName"],
            "mode": mode,
            "radiusMiles": radius_miles if mode == "weighted" else None,
            "cacheHit": meta["cacheHit"],
            "checkedProducts": meta.get("checkedProducts"),
            "lat": lat,
            "lon": lon,
            "rainMm": extract_rain_value(mode, result),
            "result": result,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/run")
def run_route():
    try:
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES
        out = run_batch_cache(mode=mode, radius_miles=radius_miles)
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/backfill-field")
def backfill_field_route():
    try:
        field_id = (request.args.get("fieldId") or "").strip()
        days = int(request.args.get("days") or KEEP_DAYS)
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        if not field_id:
            return jsonify({"ok": False, "error": "fieldId is required"}), 400

        out = backfill_field(
            field_id=field_id,
            days=days,
            mode=mode,
            radius_miles=radius_miles,
        )
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/enqueue-backfill")
def enqueue_backfill_route():
    try:
        field_id = (request.args.get("fieldId") or "").strip()
        days = int(request.args.get("days") or KEEP_DAYS)
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        if not field_id:
            return jsonify({"ok": False, "error": "fieldId is required"}), 400

        out = enqueue_backfill(
            field_id=field_id,
            days=days,
            mode=mode,
            radius_miles=radius_miles,
        )
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/enqueue-backfill-all")
def enqueue_backfill_all_route():
    try:
        days = int(request.args.get("days") or KEEP_DAYS)
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        out = enqueue_backfill_all(
            days=days,
            mode=mode,
            radius_miles=radius_miles,
        )
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/process-next-backfill")
def process_next_backfill_route():
    try:
        out = process_next_backfill()
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/repair-field")
def repair_field_route():
    try:
        field_id = (request.args.get("fieldId") or "").strip()
        lookback_hours = int(request.args.get("lookbackHours") or DEFAULT_REPAIR_LOOKBACK_HOURS)
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        if not field_id:
            return jsonify({"ok": False, "error": "fieldId is required"}), 400

        field = get_field_by_id(field_id)
        if not field:
            return jsonify({"ok": False, "error": "Field not found"}), 404

        if field_needs_full_backfill(field_id):
            out = enqueue_backfill(
                field_id=field_id,
                days=KEEP_DAYS,
                mode=mode,
                radius_miles=radius_miles,
            )
            return jsonify({
                "ok": True,
                "fieldId": field_id,
                "fieldName": field.get("name"),
                "queued": True,
                "jobType": "full_backfill",
                "reason": "field_needs_full_backfill",
                "queue": out,
            })

        meta = get_cached_dataset()
        missing = find_missing_recent_hours(
            field_id=field_id,
            latest_hour_utc=meta["fileTimestampUtc"],
            lookback_hours=lookback_hours,
        )

        if not missing:
            return jsonify({
                "ok": True,
                "fieldId": field_id,
                "fieldName": field.get("name"),
                "queued": False,
                "reason": "no_gap",
                "lookbackHours": lookback_hours,
                "latestHourUtc": meta["fileTimestampUtc"],
            })

        out = enqueue_gap_repair(
            field_id=field_id,
            start_hour_utc=missing[0],
            end_hour_utc=missing[-1],
            mode=mode,
            radius_miles=radius_miles,
        )
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "fieldName": field.get("name"),
            "queued": True,
            "jobType": "repair_gap",
            "lookbackHours": lookback_hours,
            "latestHourUtc": meta["fileTimestampUtc"],
            "missingCount": len(missing),
            "queue": out,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/enqueue-repair-all")
def enqueue_repair_all_route():
    try:
        lookback_hours = int(request.args.get("lookbackHours") or DEFAULT_REPAIR_LOOKBACK_HOURS)
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        out = enqueue_repair_all(
            lookback_hours=lookback_hours,
            mode=mode,
            radius_miles=radius_miles,
        )
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/run-repair-all")
def run_repair_all_route():
    try:
        lookback_hours = int(request.args.get("lookbackHours") or DEFAULT_REPAIR_LOOKBACK_HOURS)
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES
        max_fields = int(request.args.get("maxFields") or 0)
        max_minutes = float(request.args.get("maxMinutes") or 0)

        enq = enqueue_repair_all(
            lookback_hours=lookback_hours,
            mode=mode,
            radius_miles=radius_miles,
        )
        proc = process_repair_all(
            max_fields=max_fields,
            max_minutes=max_minutes,
        )

        return jsonify({
            "ok": True,
            "enqueue": enq,
            "process": proc,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/process-next-repair")
def process_next_repair_route():
    try:
        out = process_next_repair_gap()
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/process-repair-batch")
def process_repair_batch_route():
    try:
        if in_hourly_pause_window():
            return jsonify({
                "ok": True,
                "skipped": True,
                "reason": "paused_for_hourly_window"
            })

        max_fields = int(request.args.get("maxFields") or 10)
        max_minutes = float(request.args.get("maxMinutes") or 2)

        out = process_repair_all(
            max_fields=max_fields,
            max_minutes=max_minutes,
        )
        return jsonify({"ok": True, **out})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/queue-status")
def queue_status_route():
    try:
        db = get_db()
        rows = []

        docs = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION).stream()
        for doc in docs:
            d = doc.to_dict() or {}
            rows.append({
                "id": doc.id,
                "jobType": d.get("jobType"),
                "fieldId": d.get("fieldId"),
                "fieldName": d.get("fieldName"),
                "status": d.get("status"),
                "days": d.get("days"),
                "mode": d.get("mode"),
                "radiusMiles": d.get("radiusMiles"),
                "hoursTotal": d.get("hoursTotal"),
                "hoursDone": d.get("hoursDone"),
                "chunkHours": d.get("chunkHours"),
                "startHourUtc": d.get("startHourUtc"),
                "endHourUtc": d.get("endHourUtc"),
                "repairCursorHourUtc": d.get("repairCursorHourUtc"),
                "createdAt": d.get("createdAt"),
                "updatedAt": d.get("updatedAt"),
                "startedAt": d.get("startedAt"),
                "finishedAt": d.get("finishedAt"),
                "error": d.get("error"),
                "lastChunkSummary": d.get("lastChunkSummary"),
            })

        rows.sort(key=lambda x: (
            str(x.get("status") or ""),
            str(x.get("jobType") or ""),
            str(x.get("fieldName") or ""),
            str(x.get("id") or ""),
        ))

        return jsonify({
            "ok": True,
            "queueCollection": MRMS_BACKFILL_QUEUE_COLLECTION,
            "count": len(rows),
            "rows": rows,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/field-state")
def field_state_route():
    try:
        field_id = (request.args.get("fieldId") or "").strip()
        if not field_id:
            return jsonify({"ok": False, "error": "fieldId is required"}), 400

        field = get_field_by_id(field_id)
        if not field:
            return jsonify({"ok": False, "error": "Field not found"}), 404

        state = get_field_mrms_state(field_id)
        return jsonify({
            "ok": True,
            "fieldId": field_id,
            "fieldName": field.get("name"),
            "state": state,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/queue-simple")
def queue_simple():
    try:
        db = get_db()
        docs = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION).stream()

        queued = 0
        running = 0
        done = 0
        failed = 0

        for doc in docs:
            d = doc.to_dict() or {}
            status = str(d.get("status") or "").strip().lower()

            if status == "queued":
                queued += 1
            elif status == "running":
                running += 1
            elif status == "done":
                done += 1
            elif status == "failed":
                failed += 1

        return jsonify({
            "ok": True,
            "queued": queued,
            "running": running,
            "done": done,
            "failed": failed
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)