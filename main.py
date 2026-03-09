import gzip
import math
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request

IMPORT_ERROR = None
try:
    import fsspec
    import numpy as np
    import xarray as xr
except Exception as e:
    IMPORT_ERROR = str(e)

app = Flask(__name__)

AWS_BUCKET = "noaa-mrms-pds"
DEFAULT_REGION = "CONUS"
DEFAULT_RADIUS_MILES = 0.5
MAX_BULK_POINTS = 1000

PRODUCT_PRIORITY = [
    "MultiSensor_QPE_01H_Pass2_00.00",
    "MultiSensor_QPE_01H_Pass1_00.00",
    "RadarOnly_QPE_01H_00.00",
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


def num(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except Exception:
        return None


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


def pick_best_product_and_key(now_utc):
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

    raise RuntimeError("No usable MRMS hourly QPE file found in NOAA AWS bucket.")


def open_dataset_from_s3_key(s3_key):
    ensure_runtime_ready()
    fs = get_fs()

    with fs.open(s3_key, "rb") as f:
        compressed = f.read()

    raw = gzip.decompress(compressed)

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()

        ds = xr.open_dataset(tmp.name, engine="cfgrib")
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


def get_cached_dataset():
    now_utc = datetime.now(timezone.utc)
    product, s3_key, checked = pick_best_product_and_key(now_utc)
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
        CACHE["fileTimestampUtc"] = file_ts.isoformat() if file_ts else None
        CACHE["variableName"] = variable_name
        CACHE["dataArray"] = da
        CACHE["io"] = io_info

        return {
            "selectedProduct": product,
            "selectedKey": s3_key,
            "fileTimestampUtc": file_ts.isoformat() if file_ts else None,
            "variableName": variable_name,
            "dataArray": da,
            "io": io_info,
            "cacheHit": False,
            "checkedProducts": checked,
        }


def sample_nearest_inches(da, lat, lon):
    sampled = da.sel(latitude=lat, longitude=lon, method="nearest")

    value = sampled.values
    if isinstance(value, np.ndarray):
        value = np.asarray(value).squeeze()
        if np.size(value) != 1:
            raise RuntimeError("Unexpected non-scalar sampled value from MRMS dataset.")
        value = float(value)
    else:
        value = float(value)

    if not math.isfinite(value):
        return 0.0
    if value < 0:
        return 0.0

    return value


def sample_nearest_with_grid(da, lat, lon):
    sampled = da.sel(latitude=lat, longitude=lon, method="nearest")

    value = sampled.values
    if isinstance(value, np.ndarray):
        value = np.asarray(value).squeeze()
        if np.size(value) != 1:
            raise RuntimeError("Unexpected non-scalar sampled value from MRMS dataset.")
        value = float(value)
    else:
        value = float(value)

    grid_lat = float(sampled["latitude"].values)
    grid_lon = float(sampled["longitude"].values)

    if not math.isfinite(value) or value < 0:
        value = 0.0

    return {
        "inches": value,
        "nearestGridLatitude": grid_lat,
        "nearestGridLongitude": grid_lon,
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
        "hourlyRainInches": round_num(sampled["inches"], 4),
        "nearestGridLatitude": round_num(sampled["nearestGridLatitude"], 6),
        "nearestGridLongitude": round_num(sampled["nearestGridLongitude"], 6),
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
            "inches": round_num(sampled["inches"], 4),
            "nearestGridLatitude": round_num(sampled["nearestGridLatitude"], 6),
            "nearestGridLongitude": round_num(sampled["nearestGridLongitude"], 6),
            "ok": True,
        }
        samples.append(rec)
        good.append(rec)

    used_weight = sum(s["weight"] for s in good)
    weighted_inches = sum((s["inches"] or 0.0) * s["weight"] for s in good) / used_weight

    return {
        "lat": round_num(lat, 6),
        "lon": round_num(lon, 6),
        "radiusMiles": radius_miles,
        "weightedHourlyRainInches": round_num(weighted_inches, 4),
        "attemptedPointCount": len(samples),
        "successfulPointCount": len(good),
        "samples": samples,
    }


@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "FarmVista NOAA MRMS rainfall bulk service",
        "routes": {
            "single": "/api/mrms-1h?lat=39.7898&lon=-91.2059",
            "weighted": "/api/mrms-1h?lat=39.7898&lon=-91.2059&radiusMiles=0.5&mode=weighted",
            "bulkGet": "/api/mrms-bulk?points=39.7898,-91.2059;39.8,-91.19&mode=weighted&radiusMiles=0.5",
            "bulkPost": "POST /api/mrms-bulk  JSON: {\"points\":[{\"lat\":39.78,\"lon\":-91.20}],\"mode\":\"weighted\",\"radiusMiles\":0.5}",
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

        if mode == "single":
            result = build_single_result(da, lat, lon)
        else:
            result = build_weighted_result(da, lat, lon, radius_miles)

        return jsonify({
            "ok": True,
            "source": "noaa-mrms-aws",
            "mode": mode,
            "units": "inches",
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
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


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
            if mode == "single":
                result = build_single_result(da, p["lat"], p["lon"])
                total_grid_samples += 1
            else:
                result = build_weighted_result(da, p["lat"], p["lon"], radius_miles)
                total_grid_samples += 6

            results.append({
                "index": idx,
                **result,
            })

        return jsonify({
            "ok": True,
            "source": "noaa-mrms-aws",
            "mode": mode,
            "units": "inches",
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
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


@app.post("/api/mrms-bulk")
def api_mrms_bulk_post():
    try:
        body = request.get_json(silent=True) or {}
        points = body.get("points") or []
        mode = str(body.get("mode") or "single").strip().lower()
        radius_miles = num(body.get("radiusMiles")) or DEFAULT_RADIUS_MILES

        if not isinstance(points, list) or not points:
            return jsonify({"ok": False, "error": "JSON body must include points array"}), 400
        if len(points) > MAX_BULK_POINTS:
            return jsonify({"ok": False, "error": f"Too many points. Max {MAX_BULK_POINTS}"}), 400
        if mode not in {"single", "weighted"}:
            return jsonify({"ok": False, "error": "mode must be 'single' or 'weighted'"}), 400
        if radius_miles <= 0:
            return jsonify({"ok": False, "error": "radiusMiles must be > 0"}), 400

        cleaned_points = []
        for idx, p in enumerate(points):
            if not isinstance(p, dict):
                return jsonify({"ok": False, "error": f"Point {idx} must be an object"}), 400
            lat = num(p.get("lat"))
            lon = num(p.get("lon"))
            validate_point(lat, lon)
            cleaned_points.append({"lat": lat, "lon": lon})

        meta = get_cached_dataset()
        da = meta["dataArray"]

        results = []
        total_grid_samples = 0

        for idx, p in enumerate(cleaned_points):
            if mode == "single":
                result = build_single_result(da, p["lat"], p["lon"])
                total_grid_samples += 1
            else:
                result = build_weighted_result(da, p["lat"], p["lon"], radius_miles)
                total_grid_samples += 6

            results.append({
                "index": idx,
                **result,
            })

        return jsonify({
            "ok": True,
            "source": "noaa-mrms-aws",
            "mode": mode,
            "units": "inches",
            "selectedProduct": meta["selectedProduct"],
            "selectedKey": meta["selectedKey"].replace(f"{AWS_BUCKET}/", ""),
            "fileTimestampUtc": meta["fileTimestampUtc"],
            "variableName": meta["variableName"],
            "cacheHit": meta["cacheHit"],
            "checkedProducts": meta["checkedProducts"],
            "io": meta["io"],
            "pointCount": len(cleaned_points),
            "totalGridSamples": total_grid_samples,
            "results": results,
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)