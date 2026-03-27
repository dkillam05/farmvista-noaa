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

    for k in keys:
        if k.endswith(f"_{stamp}.grib2.gz"):
            return k
    return None


def pick_best_available_key(now_utc):
    checked = []

    for product in PRODUCT_PRIORITY:
        key = list_latest_key(DEFAULT_REGION, product, now_utc)
        checked.append({
            "product": product,
            "found": bool(key),
            "key": key.replace(f"{AWS_BUCKET}/", "") if key else None,
        })
        if key:
            return product, key, checked

    raise RuntimeError("No usable MRMS Pass2 or Pass1 hourly QPE file found in NOAA AWS bucket.")


def pick_best_key_for_hour(target_dt_utc):
    checked = []

    for product in PRODUCT_PRIORITY:
        key = list_key_for_exact_hour(DEFAULT_REGION, product, target_dt_utc)
        checked.append({
            "product": product,
            "found": bool(key),
            "key": key.replace(f"{AWS_BUCKET}/", "") if key else None,
        })
        if key:
            return product, key, checked

    return None, None, checked


def open_dataset_from_s3_key(s3_key):
    ensure_runtime_ready()
    fs = get_fs()

    with fs.open(s3_key, "rb") as f:
        compressed = f.read()

    raw = gzip.decompress(compressed)

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()

        ds = xr.open_dataset(tmp.name, engine="cfgrib", decode_timedelta=False)
        ds.load()

        io_info = {
            "compressedBytes": len(compressed),
            "decompressedBytes": len(raw),
            "tempFileSize": os.path.getsize(tmp.name),
        }
        return ds, io_info


def get_data_var(ds):
    preferred = ["unknown", "tp", "precipitation", "precip", "paramId_0"]

    for name in preferred:
        if name in ds.data_vars:
            return ds[name], name

    data_vars = list(ds.data_vars)
    if not data_vars:
        raise RuntimeError("Decoded GRIB dataset has no data variables.")
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


def get_cached_dataset():
    now_utc = datetime.now(timezone.utc)
    product, s3_key, checked = pick_best_available_key(now_utc)
    file_ts = parse_timestamp_from_key(s3_key)

    with CACHE_LOCK:
        if CACHE["selectedKey"] == s3_key and CACHE["dataArray"] is not None:
            return {
                "selectedProduct": CACHE["selectedProduct"],
                "selectedKey": CACHE["selectedKey"],
                "fileTimestampUtc": CACHE["fileTimestampUtc"],
                "variableName": CACHE["variableName"],
                "dataArray": CACHE["dataArray"],
                "io": CACHE["io"],
                "cacheHit": True,
                "checkedProducts": checked,
            }

        ds, io_info = open_dataset_from_s3_key(s3_key)
        da, variable_name = get_data_var(ds)
        da = normalize_data_array(da)

        CACHE["selectedKey"] = s3_key
        CACHE["selectedProduct"] = product
        CACHE["fileTimestampUtc"] = iso_utc(file_ts) if file_ts else None
        CACHE["variableName"] = variable_name
        CACHE["dataArray"] = da
        CACHE["io"] = io_info

        return {
            "selectedProduct": product,
            "selectedKey": s3_key,
            "fileTimestampUtc": iso_utc(file_ts) if file_ts else None,
            "variableName": variable_name,
            "dataArray": da,
            "io": io_info,
            "cacheHit": False,
            "checkedProducts": checked,
        }


def get_dataset_for_key_uncached(product, s3_key):
    file_ts = parse_timestamp_from_key(s3_key)
    ds, io_info = open_dataset_from_s3_key(s3_key)
    da, variable_name = get_data_var(ds)
    da = normalize_data_array(da)
    return {
        "selectedProduct": product,
        "selectedKey": s3_key,
        "fileTimestampUtc": iso_utc(file_ts) if file_ts else None,
        "variableName": variable_name,
        "dataArray": da,
        "io": io_info,
        "cacheHit": False,
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


def build_weighted_points(lat, lon, radius_miles):
    pts = []
    for p in SAMPLE_POINTS:
        plat, plon = offset_point(
            lat,
            lon,
            p["dxMiles"] * radius_miles,
            p["dyMiles"] * radius_miles,
        )
        pts.append({
            "key": p["key"],
            "weight": p["weight"],
            "lat": plat,
            "lon": plon,
        })
    return pts


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
    return {
        "lat": round_num(lat, 6),
        "lon": round_num(lon, 6),
        "hourlyRainMm": round_num(sampled["mm"], 4),
        "nearestGridLatitude": round_num(sampled["nearestGridLatitude"], 6),
        "nearestGridLongitude": round_num(sampled["nearestGridLongitude"], 6),
        "nearestGridLongitudeRaw": round_num(sampled["nearestGridLongitudeRaw"], 6),
        "queryLongitudeUsed": round_num(sampled["queryLongitudeUsed"], 6),
        "longitudeMode": sampled["longitudeMode"],
    }


def build_weighted_result(da, lat, lon, radius_miles):
    pts = build_weighted_points(lat, lon, radius_miles)
    samples = []
    good = []

    for p in pts:
        sampled = sample_nearest_with_grid(da, p["lat"], p["lon"])
        rec = {
            "key": p["key"],
            "weight": p["weight"],
            "lat": round_num(p["lat"], 6),
            "lon": round_num(p["lon"], 6),
            "mm": round_num(sampled["mm"], 4),
            "nearestGridLatitude": round_num(sampled["nearestGridLatitude"], 6),
            "nearestGridLongitude": round_num(sampled["nearestGridLongitude"], 6),
            "nearestGridLongitudeRaw": round_num(sampled["nearestGridLongitudeRaw"], 6),
            "queryLongitudeUsed": round_num(sampled["queryLongitudeUsed"], 6),
            "longitudeMode": sampled["longitudeMode"],
            "ok": True,
        }
        samples.append(rec)
        good.append(rec)

    used_weight = sum(s["weight"] for s in good)
    weighted_mm = sum((s["mm"] or 0.0) * s["weight"] for s in good) / used_weight

    return {
        "lat": round_num(lat, 6),
        "lon": round_num(lon, 6),
        "radiusMiles": radius_miles,
        "weightedHourlyRainMm": round_num(weighted_mm, 4),
        "attemptedPointCount": len(samples),
        "successfulPointCount": len(good),
        "samples": samples,
    }


def load_active_fields_for_batch():
    db = get_db()
    raw = []

    try:
        snap = db.collection(FIELDS_COLLECTION).where("status", "==", "active").stream()
        for doc in snap:
            raw.append({"id": doc.id, "data": doc.to_dict() or {}})
    except Exception as e:
        print(f"[Batch] fields query(status==active) failed: {e}", flush=True)

    if not raw:
        try:
            snap2 = db.collection(FIELDS_COLLECTION).stream()
            for doc in snap2:
                raw.append({"id": doc.id, "data": doc.to_dict() or {}})
        except Exception as e:
            print(f"[Batch] fields query(all) failed: {e}", flush=True)
            raw = []

    out = []
    for r in raw:
        d = r["data"] or {}
        status = str(d.get("status", "")).strip().lower()
        if status != "active":
            continue

        loc = d.get("location") or {}
        lat = num(loc.get("lat"))
        lng = num(loc.get("lng"))
        if lat is None or lng is None:
            continue

        out.append({
            "id": r["id"],
            "name": str(d.get("name") or ""),
            "farmId": d.get("farmId"),
            "farmName": d.get("farmName"),
            "lat": lat,
            "lng": lng,
        })

    return out


def get_field_by_id(field_id):
    db = get_db()
    doc = db.collection(FIELDS_COLLECTION).document(field_id).get()
    if not doc.exists:
        return None

    d = doc.to_dict() or {}
    loc = d.get("location") or {}
    lat = num(loc.get("lat"))
    lng = num(loc.get("lng"))
    if lat is None or lng is None:
        raise RuntimeError("Field exists but location.lat/lng is missing or invalid")

    return {
        "id": doc.id,
        "name": str(d.get("name") or ""),
        "farmId": d.get("farmId"),
        "farmName": d.get("farmName"),
        "lat": lat,
        "lng": lng,
    }


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
        "io": meta["io"],
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
        ts = str(d.get("fileTimestampUtc") or "")
        rain_mm = get_doc_rain_mm(d)
        if not ts or rain_mm is None:
            continue
        rows.append({
            "hourKey": doc.id,
            "fileTimestampUtc": ts,
            "rainMm": round_num(rain_mm, 4),
            "selectedProduct": d.get("selectedProduct"),
            "mode": d.get("mode"),
            "source": d.get("source"),
        })

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

        bucket["rainMm"] += float(row["rainMm"] or 0.0)
        bucket["hoursCount"] += 1

    daily30 = [
        {
            "dateISO": v["dateISO"],
            "rainMm": round_num(v["rainMm"], 4),
            "hoursCount": v["hoursCount"],
        }
        for _, v in sorted(daily_map.items())
    ][-KEEP_DAYS:]

    return last24, daily30


def write_field_hour(parent_ref, hour_ref, field, meta, mode, radius_miles, result):
    history_entry = build_hour_history_entry(meta, mode, radius_miles, result)
    hour_ref.set(history_entry, merge=True)

    last24, daily30 = rebuild_last24_and_daily30(field["id"])
    existing_state = get_field_mrms_state(field["id"])

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
    parent_ref.set(parent_payload, merge=True)


def queue_job_exists_active(doc_id):
    db = get_db()
    snap = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION).document(doc_id).get()
    if not snap.exists:
        return False
    data = snap.to_dict() or {}
    return str(data.get("status") or "").strip().lower() in {"queued", "running"}


def field_needs_full_backfill(field_id):
    state = get_field_mrms_state(field_id)
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

    doc_id = full_backfill_job_id(field_id)
    ref = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION).document(doc_id)

    if queue_job_exists_active(doc_id):
        return {
            "fieldId": field_id,
            "fieldName": field.get("name"),
            "status": "already_active",
            "days": days,
            "mode": mode,
            "radiusMiles": radius_miles,
            "jobType": "full_backfill",
        }

    anchor_hour_utc = iso_utc(floor_to_hour_utc(datetime.now(timezone.utc)))

    ref.set({
        "jobType": "full_backfill",
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "status": "queued",
        "days": days,
        "mode": mode,
        "radiusMiles": radius_miles,
        "anchorHourUtc": anchor_hour_utc,
        "hoursTotal": days * 24,
        "hoursDone": 0,
        "chunkHours": int(clamp(DEFAULT_FULL_BACKFILL_CHUNK_HOURS, 1, 168)),
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
        "days": days,
        "mode": mode,
        "radiusMiles": radius_miles,
        "jobType": "full_backfill",
        "anchorHourUtc": anchor_hour_utc,
        "hoursTotal": days * 24,
        "chunkHours": int(clamp(DEFAULT_FULL_BACKFILL_CHUNK_HOURS, 1, 168)),
    }


def force_reset_full_backfill(field_id, days=30, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES, reason="location_changed"):
    db = get_db()
    field = get_field_by_id(field_id)
    if not field:
        raise RuntimeError("Field not found")

    days = int(clamp(days, 1, 30))
    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    anchor_hour_utc = iso_utc(floor_to_hour_utc(datetime.now(timezone.utc)))
    doc_id = full_backfill_job_id(field_id)
    ref = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION).document(doc_id)

    ref.set({
        "jobType": "full_backfill",
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "status": "queued",
        "days": days,
        "mode": mode,
        "radiusMiles": radius_miles,
        "anchorHourUtc": anchor_hour_utc,
        "hoursTotal": days * 24,
        "hoursDone": 0,
        "chunkHours": int(clamp(DEFAULT_FULL_BACKFILL_CHUNK_HOURS, 1, 168)),
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
        "startedAt": None,
        "finishedAt": None,
        "error": None,
        "resetReason": reason,
        "attempts": firestore.Increment(1),
        "lastChunkSummary": None,
    }, merge=True)

    return {
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "status": "queued",
        "days": days,
        "mode": mode,
        "radiusMiles": radius_miles,
        "jobType": "full_backfill",
        "anchorHourUtc": anchor_hour_utc,
        "hoursTotal": days * 24,
        "chunkHours": int(clamp(DEFAULT_FULL_BACKFILL_CHUNK_HOURS, 1, 168)),
        "resetReason": reason,
    }


def delete_subcollection_documents(coll_ref, batch_size=400):
    deleted = 0
    while True:
        docs = list(coll_ref.limit(batch_size).stream())
        if not docs:
            break
        b = get_db().batch()
        for doc in docs:
            b.delete(doc.reference)
            deleted += 1
        b.commit()
        if len(docs) < batch_size:
            break
    return deleted


def reset_field_history_for_location_change(field, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    db = get_db()
    field_id = field["id"]
    parent_ref = db.collection(MRMS_PARENT_COLLECTION).document(field_id)
    hourly_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION)

    deleted_hours = delete_subcollection_documents(hourly_ref)
    deleted_queue_jobs = delete_queue_jobs_for_field(field_id)

    parent_ref.set({
        "fieldId": field_id,
        "fieldName": field.get("name") or None,
        "farmId": field.get("farmId"),
        "farmName": field.get("farmName"),
        "location": {"lat": field["lat"], "lng": field["lng"]},
        "mrmsHourlyLatest": {},
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
            "locationChangedAt": firestore.SERVER_TIMESTAMP,
            "locationResetReason": "field_lat_lng_changed",
        },
        "mrmsLastUpdatedAt": firestore.SERVER_TIMESTAMP,
    }, merge=True)

    queue_result = force_reset_full_backfill(
        field_id=field_id,
        days=KEEP_DAYS,
        mode=mode,
        radius_miles=radius_miles,
        reason="location_changed",
    )

    return {
        "fieldId": field_id,
        "fieldName": field.get("name"),
        "deletedHourlyDocs": deleted_hours,
        "deletedQueueJobs": deleted_queue_jobs,
        "queueResult": queue_result,
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
    if field_needs_full_backfill(field["id"]):
        return {"fieldId": field["id"], "queued": False, "reason": "full_backfill_not_complete"}

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

    meta = get_cached_dataset()
    da = meta["dataArray"]

    ok = 0
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

            result = build_single_result(da, field["lat"], field["lng"]) if mode == "single" else build_weighted_result(da, field["lat"], field["lng"], radius_miles)

            parent_ref = get_db().collection(MRMS_PARENT_COLLECTION).document(field["id"])
            hour_id = hour_doc_id_from_iso(meta["fileTimestampUtc"])
            hour_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION).document(hour_id)
            write_field_hour(parent_ref, hour_ref, field, meta, mode, radius_miles, result)
            ok += 1

            if not location_changed_this_run:
                try:
                    auto_rep = maybe_auto_enqueue_gap_repair(
                        field=field,
                        mode=mode,
                        radius_miles=radius_miles,
                        latest_hour_utc=meta["fileTimestampUtc"],
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
        "fail": fail,
        "collection": MRMS_PARENT_COLLECTION,
        "selectedProduct": meta["selectedProduct"],
        "selectedKey": meta["selectedKey"].replace(f"{AWS_BUCKET}/", ""),
        "fileTimestampUtc": meta["fileTimestampUtc"],
        "cacheHit": meta["cacheHit"],
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
            history_entry = build_hour_history_entry(meta, mode, radius_miles, result)
            hour_ref.set(history_entry, merge=True)
            ok += 1
        except Exception as e:
            fail += 1
            failures.append({
                "targetHourUtc": iso_utc(target_dt),
                "error": str(e),
            })

    finalize_field_parent_from_hourly(field["id"], mark_full_backfill_complete=True)

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "daysRequested": days,
        "hoursRequested": total_hours,
        "ok": ok,
        "skippedNoFile": skipped,
        "fail": fail,
        "historySubcollection": MRMS_HOURLY_SUBCOLLECTION,
        "failures": failures[:50],
    }


def backfill_field_chunk(field_id, anchor_hour_utc, hours_total, hours_done, chunk_hours, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES, deadline_ts=None):
    field = get_field_by_id(field_id)
    if not field:
        raise RuntimeError("Field not found")

    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    anchor_dt = floor_to_hour_utc(parse_iso_utc(anchor_hour_utc))
    hours_total = int(clamp(hours_total, 1, KEEP_HOURS))
    hours_done = int(clamp(hours_done, 0, hours_total))
    chunk_hours = int(clamp(chunk_hours, 1, 168))

    start_index = hours_done
    end_index = min(hours_total, start_index + chunk_hours)

    parent_ref = get_db().collection(MRMS_PARENT_COLLECTION).document(field["id"])

    ok = 0
    skipped = 0
    fail = 0
    failures = []
    processed_slots = 0

    for i in range(start_index, end_index):
        if deadline_ts is not None and time.time() >= deadline_ts:
            break

        target_dt = anchor_dt - timedelta(hours=i)
        product, s3_key, checked = pick_best_key_for_hour(target_dt)

        if not s3_key:
            skipped += 1
            processed_slots += 1
            continue

        try:
            meta = get_dataset_for_key_uncached(product, s3_key)
            meta["checkedProducts"] = checked
            da = meta["dataArray"]

            result = build_single_result(da, field["lat"], field["lng"]) if mode == "single" else build_weighted_result(da, field["lat"], field["lng"], radius_miles)

            hour_id = hour_doc_id_from_iso(meta["fileTimestampUtc"])
            hour_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION).document(hour_id)
            history_entry = build_hour_history_entry(meta, mode, radius_miles, result)
            hour_ref.set(history_entry, merge=True)
            ok += 1
        except Exception as e:
            fail += 1
            failures.append({
                "targetHourUtc": iso_utc(target_dt),
                "error": str(e),
            })

        processed_slots += 1

    new_hours_done = hours_done + processed_slots
    is_complete = new_hours_done >= hours_total

    finalize_field_parent_from_hourly(field["id"], mark_full_backfill_complete=is_complete)

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "jobType": "full_backfill",
        "anchorHourUtc": iso_utc(anchor_dt),
        "hoursTotal": hours_total,
        "hoursDoneBefore": hours_done,
        "hoursDoneAfter": new_hours_done,
        "chunkHoursRequested": chunk_hours,
        "chunkHoursProcessed": processed_slots,
        "isComplete": is_complete,
        "ok": ok,
        "skippedNoFile": skipped,
        "fail": fail,
        "failures": failures[:50],
    }


def repair_field_range(field_id, start_hour_utc, end_hour_utc, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES):
    field = get_field_by_id(field_id)
    if not field:
        raise RuntimeError("Field not found")

    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    start_dt = floor_to_hour_utc(parse_iso_utc(start_hour_utc))
    end_dt = floor_to_hour_utc(parse_iso_utc(end_hour_utc))

    if end_dt < start_dt:
        raise RuntimeError("endHourUtc must be >= startHourUtc")

    parent_ref = get_db().collection(MRMS_PARENT_COLLECTION).document(field["id"])

    ok = 0
    skipped = 0
    fail = 0
    failures = []

    cur = start_dt
    while cur <= end_dt:
        product, s3_key, checked = pick_best_key_for_hour(cur)

        if not s3_key:
            skipped += 1
            cur += timedelta(hours=1)
            continue

        try:
            meta = get_dataset_for_key_uncached(product, s3_key)
            meta["checkedProducts"] = checked
            da = meta["dataArray"]

            result = build_single_result(da, field["lat"], field["lng"]) if mode == "single" else build_weighted_result(da, field["lat"], field["lng"], radius_miles)

            hour_id = hour_doc_id_from_iso(meta["fileTimestampUtc"])
            hour_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION).document(hour_id)
            history_entry = build_hour_history_entry(meta, mode, radius_miles, result)
            hour_ref.set(history_entry, merge=True)
            ok += 1
        except Exception as e:
            fail += 1
            failures.append({
                "targetHourUtc": iso_utc(cur),
                "error": str(e),
            })

        cur += timedelta(hours=1)

    finalize_field_parent_from_hourly(field["id"], mark_full_backfill_complete=False)

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "jobType": "repair_gap",
        "startHourUtc": iso_utc(start_dt),
        "endHourUtc": iso_utc(end_dt),
        "ok": ok,
        "skippedNoFile": skipped,
        "fail": fail,
        "failures": failures[:50],
    }


def repair_field_range_chunk(field_id, start_hour_utc, end_hour_utc, cursor_hour_utc=None, chunk_hours=DEFAULT_REPAIR_CHUNK_HOURS, mode="weighted", radius_miles=DEFAULT_RADIUS_MILES, deadline_ts=None):
    field = get_field_by_id(field_id)
    if not field:
        raise RuntimeError("Field not found")

    if mode not in {"single", "weighted"}:
        raise RuntimeError("mode must be 'single' or 'weighted'")
    if radius_miles <= 0:
        raise RuntimeError("radiusMiles must be > 0")

    start_dt = floor_to_hour_utc(parse_iso_utc(start_hour_utc))
    end_dt = floor_to_hour_utc(parse_iso_utc(end_hour_utc))
    if end_dt < start_dt:
        raise RuntimeError("endHourUtc must be >= startHourUtc")

    cur = floor_to_hour_utc(parse_iso_utc(cursor_hour_utc)) if cursor_hour_utc else start_dt
    if cur < start_dt:
        cur = start_dt
    if cur > end_dt:
        finalize_field_parent_from_hourly(field["id"], mark_full_backfill_complete=False)
        return {
            "fieldId": field["id"],
            "fieldName": field.get("name"),
            "jobType": "repair_gap",
            "startHourUtc": iso_utc(start_dt),
            "endHourUtc": iso_utc(end_dt),
            "cursorHourUtcBefore": iso_utc(cur),
            "cursorHourUtcAfter": None,
            "isComplete": True,
            "chunkHoursRequested": int(clamp(chunk_hours, 1, 168)),
            "chunkHoursProcessed": 0,
            "ok": 0,
            "skippedNoFile": 0,
            "fail": 0,
            "failures": [],
        }

    chunk_hours = int(clamp(chunk_hours, 1, 168))
    processed_slots = 0

    parent_ref = get_db().collection(MRMS_PARENT_COLLECTION).document(field["id"])

    ok = 0
    skipped = 0
    fail = 0
    failures = []

    cursor_before = cur

    while cur <= end_dt and processed_slots < chunk_hours:
        if deadline_ts is not None and time.time() >= deadline_ts:
            break

        product, s3_key, checked = pick_best_key_for_hour(cur)

        if not s3_key:
            skipped += 1
            processed_slots += 1
            cur += timedelta(hours=1)
            continue

        try:
            meta = get_dataset_for_key_uncached(product, s3_key)
            meta["checkedProducts"] = checked
            da = meta["dataArray"]

            result = build_single_result(da, field["lat"], field["lng"]) if mode == "single" else build_weighted_result(da, field["lat"], field["lng"], radius_miles)

            hour_id = hour_doc_id_from_iso(meta["fileTimestampUtc"])
            hour_ref = parent_ref.collection(MRMS_HOURLY_SUBCOLLECTION).document(hour_id)
            history_entry = build_hour_history_entry(meta, mode, radius_miles, result)
            hour_ref.set(history_entry, merge=True)
            ok += 1
        except Exception as e:
            fail += 1
            failures.append({
                "targetHourUtc": iso_utc(cur),
                "error": str(e),
            })

        processed_slots += 1
        cur += timedelta(hours=1)

    is_complete = cur > end_dt
    finalize_field_parent_from_hourly(field["id"], mark_full_backfill_complete=False)

    return {
        "fieldId": field["id"],
        "fieldName": field.get("name"),
        "jobType": "repair_gap",
        "startHourUtc": iso_utc(start_dt),
        "endHourUtc": iso_utc(end_dt),
        "cursorHourUtcBefore": iso_utc(cursor_before),
        "cursorHourUtcAfter": None if is_complete else iso_utc(cur),
        "isComplete": is_complete,
        "chunkHoursRequested": chunk_hours,
        "chunkHoursProcessed": processed_slots,
        "ok": ok,
        "skippedNoFile": skipped,
        "fail": fail,
        "failures": failures[:50],
    }


def claim_next_backfill():
    db = get_db()
    queue_ref = db.collection(MRMS_BACKFILL_QUEUE_COLLECTION)

    candidates = list(
        queue_ref.where("status", "==", "queued").limit(200).stream()
    )

    if not candidates:
        return None, None

    def created_sort_value(doc_dict):
        created = doc_dict.get("createdAt")
        if created is None:
            return datetime.max.replace(tzinfo=timezone.utc)

        try:
            if isinstance(created, datetime):
                return created if created.tzinfo else created.replace(tzinfo=timezone.utc)

            if hasattr(created, "datetime"):
                dt = created.datetime
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

            if hasattr(created, "to_datetime"):
                dt = created.to_datetime()
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

        return datetime.max.replace(tzinfo=timezone.utc)

    def job_priority(doc_dict):
        job_type = str(doc_dict.get("jobType") or "full_backfill").strip().lower()
        if job_type == "full_backfill":
            return 0
        if job_type == "repair_gap":
            return 1
        return 9

    enriched = []
    for doc in candidates:
        d = doc.to_dict() or {}
        enriched.append((job_priority(d), created_sort_value(d), doc, d))

    enriched.sort(key=lambda x: (x[0], x[1]))

    doc = enriched[0][2]
    ref = doc.reference

    @firestore.transactional
    def txn_claim(transaction, doc_ref):
        snap = doc_ref.get(transaction=transaction)
        if not snap.exists:
            return None
        d = snap.to_dict() or {}
        if str(d.get("status") or "").strip().lower() != "queued":
            return None

        transaction.set(doc_ref, {
            "status": "running",
            "startedAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "error": None,
        }, merge=True)
        return d

    transaction = db.transaction()
    claimed_doc = txn_claim(transaction, ref)
    if not claimed_doc:
        return None, None

    return ref, claimed_doc


def process_one_backfill_job():
    ref, queued = claim_next_backfill()
    if not ref:
        return {"processed": False, "message": "No queued backfill jobs found"}

    job_type = str(queued.get("jobType") or "full_backfill").strip().lower()
    field_id = queued.get("fieldId")
    mode = str(queued.get("mode") or "weighted").strip().lower()
    radius_miles = num(queued.get("radiusMiles")) or DEFAULT_RADIUS_MILES
    deadline_ts = time.time() + (DEFAULT_BACKFILL_MAX_MINUTES_PER_RUN * 60.0) - 20.0

    try:
        if job_type == "repair_gap":
            start_hour_utc = str(queued.get("startHourUtc") or "").strip()
            end_hour_utc = str(queued.get("endHourUtc") or "").strip()
            if not start_hour_utc or not end_hour_utc:
                raise RuntimeError("repair_gap queue item missing startHourUtc/endHourUtc")

            cursor_hour_utc = str(queued.get("repairCursorHourUtc") or start_hour_utc).strip()
            chunk_hours = int(clamp(queued.get("chunkHours") or DEFAULT_REPAIR_CHUNK_HOURS, 1, 168))

            result = repair_field_range_chunk(
                field_id=field_id,
                start_hour_utc=start_hour_utc,
                end_hour_utc=end_hour_utc,
                cursor_hour_utc=cursor_hour_utc,
                chunk_hours=chunk_hours,
                mode=mode,
                radius_miles=radius_miles,
                deadline_ts=deadline_ts,
            )

            if result.get("isComplete"):
                ref.set({
                    "status": "done",
                    "finishedAt": firestore.SERVER_TIMESTAMP,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                    "hoursDone": queued.get("hoursTotal") or result.get("chunkHoursProcessed"),
                    "resultSummary": {
                        "okHours": result.get("ok"),
                        "skippedNoFile": result.get("skippedNoFile"),
                        "failHours": result.get("fail"),
                        "jobType": job_type,
                    },
                    "lastChunkSummary": {
                        "cursorHourUtcBefore": result.get("cursorHourUtcBefore"),
                        "cursorHourUtcAfter": result.get("cursorHourUtcAfter"),
                        "chunkHoursProcessed": result.get("chunkHoursProcessed"),
                        "okHours": result.get("ok"),
                        "skippedNoFile": result.get("skippedNoFile"),
                        "failHours": result.get("fail"),
                    },
                    "error": None,
                }, merge=True)

                return {"processed": True, "queueStatus": "done", "result": result}

            hours_total = int(clamp(queued.get("hoursTotal") or 1, 1, 10000))
            hours_done = int(clamp(queued.get("hoursDone") or 0, 0, hours_total))
            new_hours_done = min(hours_total, hours_done + int(result.get("chunkHoursProcessed") or 0))

            ref.set({
                "status": "queued",
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "repairCursorHourUtc": result.get("cursorHourUtcAfter"),
                "hoursDone": new_hours_done,
                "lastChunkSummary": {
                    "cursorHourUtcBefore": result.get("cursorHourUtcBefore"),
                    "cursorHourUtcAfter": result.get("cursorHourUtcAfter"),
                    "chunkHoursProcessed": result.get("chunkHoursProcessed"),
                    "okHours": result.get("ok"),
                    "skippedNoFile": result.get("skippedNoFile"),
                    "failHours": result.get("fail"),
                },
                "error": None,
            }, merge=True)

            return {"processed": True, "queueStatus": "requeued", "result": result}

        days = int(clamp(queued.get("days") or 30, 1, 30))
        hours_total = int(clamp(queued.get("hoursTotal") or (days * 24), 1, KEEP_HOURS))
        hours_done = int(clamp(queued.get("hoursDone") or 0, 0, hours_total))
        chunk_hours = int(clamp(queued.get("chunkHours") or DEFAULT_FULL_BACKFILL_CHUNK_HOURS, 1, 168))
        anchor_hour_utc = str(queued.get("anchorHourUtc") or iso_utc(floor_to_hour_utc(datetime.now(timezone.utc)))).strip()

        result = backfill_field_chunk(
            field_id=field_id,
            anchor_hour_utc=anchor_hour_utc,
            hours_total=hours_total,
            hours_done=hours_done,
            chunk_hours=chunk_hours,
            mode=mode,
            radius_miles=radius_miles,
            deadline_ts=deadline_ts,
        )

        if result.get("isComplete"):
            ref.set({
                "status": "done",
                "finishedAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                "hoursDone": result.get("hoursDoneAfter"),
                "resultSummary": {
                    "okHours": result.get("ok"),
                    "skippedNoFile": result.get("skippedNoFile"),
                    "failHours": result.get("fail"),
                    "jobType": job_type,
                },
                "lastChunkSummary": {
                    "hoursDoneBefore": result.get("hoursDoneBefore"),
                    "hoursDoneAfter": result.get("hoursDoneAfter"),
                    "chunkHoursProcessed": result.get("chunkHoursProcessed"),
                    "okHours": result.get("ok"),
                    "skippedNoFile": result.get("skippedNoFile"),
                    "failHours": result.get("fail"),
                },
                "error": None,
            }, merge=True)

            return {"processed": True, "queueStatus": "done", "result": result}

        ref.set({
            "status": "queued",
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "hoursDone": result.get("hoursDoneAfter"),
            "lastChunkSummary": {
                "hoursDoneBefore": result.get("hoursDoneBefore"),
                "hoursDoneAfter": result.get("hoursDoneAfter"),
                "chunkHoursProcessed": result.get("chunkHoursProcessed"),
                "okHours": result.get("ok"),
                "skippedNoFile": result.get("skippedNoFile"),
                "failHours": result.get("fail"),
            },
            "error": None,
        }, merge=True)

        return {"processed": True, "queueStatus": "requeued", "result": result}

    except Exception as e:
        print(f"[process_one_backfill_job] fieldId={field_id} jobType={job_type} ERROR: {e}", flush=True)
        traceback.print_exc()
        ref.set({
            "status": "failed",
            "finishedAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "error": str(e),
        }, merge=True)
        return {"processed": True, "queueStatus": "failed", "error": str(e), "fieldId": field_id, "jobType": job_type}


def process_next_backfill(max_fields=None, max_minutes=None):
    max_fields = int(clamp(max_fields if max_fields is not None else DEFAULT_BACKFILL_MAX_FIELDS_PER_RUN, 1, 100000))
    max_minutes = float(clamp(max_minutes if max_minutes is not None else DEFAULT_BACKFILL_MAX_MINUTES_PER_RUN, 1, 55))

    start = time.time()
    processed_count = 0
    done_count = 0
    failed_count = 0
    requeued_count = 0
    results = []

    while True:
        elapsed_minutes = (time.time() - start) / 60.0
        if processed_count >= max_fields:
            break
        if elapsed_minutes >= max_minutes:
            break

        one = process_one_backfill_job()
        if not one.get("processed"):
            break

        processed_count += 1
        if one.get("queueStatus") == "done":
            done_count += 1
        elif one.get("queueStatus") == "failed":
            failed_count += 1
        elif one.get("queueStatus") == "requeued":
            requeued_count += 1

        results.append(one)

    remaining = queue_counts()

    return {
        "processed": processed_count > 0,
        "processedCount": processed_count,
        "doneCount": done_count,
        "failedCount": failed_count,
        "requeuedCount": requeued_count,
        "elapsedMinutes": round_num((time.time() - start) / 60.0, 2),
        "stoppedReason": (
            "queue_empty" if remaining["queued"] == 0 else
            "max_fields_reached" if processed_count >= max_fields else
            "max_minutes_reached"
        ),
        "maxFields": max_fields,
        "maxMinutes": max_minutes,
        "remaining": remaining,
        "results": results,
    }


def queue_status(limit=50):
    db = get_db()
    docs = list(
        db.collection(MRMS_BACKFILL_QUEUE_COLLECTION)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    rows = []
    for doc in docs:
        d = doc.to_dict() or {}
        rows.append({
            "docId": doc.id,
            "jobType": d.get("jobType") or "full_backfill",
            "fieldId": d.get("fieldId") or doc.id,
            "fieldName": d.get("fieldName"),
            "status": d.get("status"),
            "days": d.get("days"),
            "mode": d.get("mode"),
            "radiusMiles": d.get("radiusMiles"),
            "startHourUtc": d.get("startHourUtc"),
            "endHourUtc": d.get("endHourUtc"),
            "anchorHourUtc": d.get("anchorHourUtc"),
            "repairCursorHourUtc": d.get("repairCursorHourUtc"),
            "hoursTotal": d.get("hoursTotal"),
            "hoursDone": d.get("hoursDone"),
            "chunkHours": d.get("chunkHours"),
            "lastChunkSummary": d.get("lastChunkSummary"),
            "error": d.get("error"),
            "resetReason": d.get("resetReason"),
        })
    return rows


def queue_counts():
    db = get_db()
    out = {"queued": 0, "running": 0, "done": 0, "failed": 0}
    for status in out.keys():
        docs = list(
            db.collection(MRMS_BACKFILL_QUEUE_COLLECTION)
            .where("status", "==", status)
            .limit(10000)
            .stream()
        )
        out[status] = len(docs)
    return out


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
        "routes": {
            "single": "/api/mrms-1h?lat=39.7898&lon=-91.2059",
            "weighted": "/api/mrms-1h?lat=39.7898&lon=-91.2059&radiusMiles=0.5&mode=weighted",
            "runBatchCache": "/run?mode=weighted&radiusMiles=0.5",
            "backfillField": "/backfill-field?fieldId=YOUR_FIELD_ID&days=30&mode=weighted&radiusMiles=0.5",
            "enqueueBackfill": "/enqueue-backfill?fieldId=YOUR_FIELD_ID&days=30&mode=weighted&radiusMiles=0.5",
            "enqueueBackfillAll": "/enqueue-backfill-all?days=30&mode=weighted&radiusMiles=0.5",
            "processNextBackfill": "/process-next-backfill?maxFields=1&maxMinutes=4",
            "queueStatus": "/queue-status",
        },
    })


@app.get("/api/mrms-1h")
def api_mrms_1h():
    try:
        lat = num(request.args.get("lat"))
        lon = num(request.args.get("lon"))
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES
        mode = (request.args.get("mode") or "single").strip().lower()

        validate_point(lat, lon)

        if radius_miles <= 0:
            return jsonify({"ok": False, "error": "radiusMiles must be > 0"}), 400
        if mode not in {"single", "weighted"}:
            return jsonify({"ok": False, "error": "mode must be 'single' or 'weighted'"}), 400

        meta = get_cached_dataset()
        da = meta["dataArray"]
        result = build_single_result(da, lat, lon) if mode == "single" else build_weighted_result(da, lat, lon, radius_miles)

        return jsonify({
            "ok": True,
            "source": "noaa-mrms-aws",
            "mode": mode,
            "units": "mm",
            "selectedProduct": meta["selectedProduct"],
            "selectedKey": meta["selectedKey"].replace(f"{AWS_BUCKET}/", ""),
            "fileTimestampUtc": meta["fileTimestampUtc"],
            "variableName": meta["variableName"],
            "cacheHit": meta["cacheHit"],
            "checkedProducts": meta["checkedProducts"],
            "io": meta["io"],
            "result": result,
        })
    except Exception as e:
        print(f"[api/mrms-1h] ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/mrms-bulk")
def api_mrms_bulk_get():
    try:
        points_str = request.args.get("points", "")
        mode = (request.args.get("mode") or "single").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        points = parse_bulk_points_from_query(points_str)
        if not points:
            return jsonify({"ok": False, "error": "No points provided"}), 400
        if len(points) > MAX_BULK_POINTS:
            return jsonify({"ok": False, "error": f"Too many points. Max {MAX_BULK_POINTS}"}), 400
        if mode not in {"single", "weighted"}:
            return jsonify({"ok": False, "error": "mode must be 'single' or 'weighted'"}), 400
        if radius_miles <= 0:
            return jsonify({"ok": False, "error": "radiusMiles must be > 0"}), 400

        meta = get_cached_dataset()
        da = meta["dataArray"]

        results = []
        total_grid_samples = 0

        for idx, p in enumerate(points):
            validate_point(p["lat"], p["lon"])
            result = build_single_result(da, p["lat"], p["lon"]) if mode == "single" else build_weighted_result(da, p["lat"], p["lon"], radius_miles)
            total_grid_samples += 1 if mode == "single" else 6
            results.append({"index": idx, **result})

        return jsonify({
            "ok": True,
            "source": "noaa-mrms-aws",
            "mode": mode,
            "units": "mm",
            "selectedProduct": meta["selectedProduct"],
            "selectedKey": meta["selectedKey"].replace(f"{AWS_BUCKET}/", ""),
            "fileTimestampUtc": meta["fileTimestampUtc"],
            "variableName": meta["variableName"],
            "cacheHit": meta["cacheHit"],
            "checkedProducts": meta["checkedProducts"],
            "io": meta["io"],
            "pointCount": len(points),
            "totalGridSamples": total_grid_samples,
            "results": results,
        })
    except Exception as e:
        print(f"[api/mrms-bulk] ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/run")
def run_batch():
    try:
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES
        out = run_batch_cache(mode=mode, radius_miles=radius_miles)
        return jsonify({
            "ok": True,
            "mode": mode,
            "radiusMiles": radius_miles if mode == "weighted" else None,
            "writesToCollection": MRMS_PARENT_COLLECTION,
            "historySubcollection": MRMS_HOURLY_SUBCOLLECTION,
            "rainUnit": "mm",
            "result": out,
        })
    except Exception as e:
        print(f"[/run] ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/backfill-field")
def backfill_field_route():
    try:
        field_id = str(request.args.get("fieldId") or "").strip()
        days = int(clamp(request.args.get("days") or 30, 1, 30))
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        if not field_id:
            return jsonify({"ok": False, "error": "Missing fieldId"}), 400

        out = backfill_field(field_id=field_id, days=days, mode=mode, radius_miles=radius_miles)
        return jsonify({"ok": True, "mode": mode, "radiusMiles": radius_miles if mode == "weighted" else None, "rainUnit": "mm", "result": out})
    except Exception as e:
        print(f"[/backfill-field] ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/enqueue-backfill")
def enqueue_backfill_route():
    try:
        field_id = str(request.args.get("fieldId") or "").strip()
        days = int(clamp(request.args.get("days") or 30, 1, 30))
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        if not field_id:
            return jsonify({"ok": False, "error": "Missing fieldId"}), 400

        out = enqueue_backfill(field_id=field_id, days=days, mode=mode, radius_miles=radius_miles)
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        print(f"[/enqueue-backfill] ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/enqueue-backfill-all")
def enqueue_backfill_all_route():
    try:
        days = int(clamp(request.args.get("days") or 30, 1, 30))
        mode = (request.args.get("mode") or "weighted").strip().lower()
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        out = enqueue_backfill_all(days=days, mode=mode, radius_miles=radius_miles)
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        print(f"[/enqueue-backfill-all] ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/process-next-backfill")
def process_next_backfill_route():
    try:
        max_fields = request.args.get("maxFields")
        max_minutes = request.args.get("maxMinutes")
        out = process_next_backfill(max_fields=max_fields, max_minutes=max_minutes)
        return jsonify({"ok": True, "result": out})
    except Exception as e:
        print(f"[process-next-backfill] ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/queue-status")
def queue_status_route():
    try:
        limit = int(clamp(request.args.get("limit") or 50, 1, 200))
        out = queue_status(limit=limit)
        return jsonify({
            "ok": True,
            "queueCollection": MRMS_BACKFILL_QUEUE_COLLECTION,
            "counts": queue_counts(),
            "items": out
        })
    except Exception as e:
        print(f"[/queue-status] ERROR: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
