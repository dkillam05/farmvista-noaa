import gzip
import math
import os
import tempfile
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

        try:
            ds = xr.open_dataset(tmp.name, engine="cfgrib")
            ds.load()
            return ds, {
                "compressedBytes": len(compressed),
                "decompressedBytes": len(raw),
                "tempFileSize": os.path.getsize(tmp.name),
            }
        except Exception as e:
            raise RuntimeError(
                "MRMS file was found and downloaded, but GRIB decode failed. "
                f"Original error: {e}"
            )


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
    coords = list(da.coords)

    if "latitude" not in da.coords and "lat" in da.coords:
        rename_map["lat"] = "latitude"
    if "longitude" not in da.coords and "lon" in da.coords:
        rename_map["lon"] = "longitude"

    if rename_map:
        da = da.rename(rename_map)

    if "latitude" not in da.coords or "longitude" not in da.coords:
        raise RuntimeError(f"Dataset missing usable latitude/longitude coordinates. Found coords: {coords}")

    # squeeze singleton dimensions like time/step/heightAboveSea
    for dim in list(da.dims):
        if dim not in ("latitude", "longitude") and da.sizes.get(dim, 0) == 1:
            da = da.isel({dim: 0})

    return da


def sample_nearest_inches(da, lat, lon):
    da = normalize_data_array(da)
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


@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "FarmVista NOAA MRMS rainfall service",
        "routes": {
            "single": "/api/mrms-1h?lat=39.7898&lon=-91.2059",
            "weighted": "/api/mrms-1h?lat=39.7898&lon=-91.2059&radiusMiles=0.5&mode=weighted",
            "debug": "/api/mrms-debug?lat=39.7898&lon=-91.2059",
        },
    })


@app.get("/api/mrms-debug")
def api_mrms_debug():
    try:
        lat = num(request.args.get("lat"))
        lon = num(request.args.get("lon"))

        if lat is None or lon is None:
            return jsonify({"ok": False, "error": "Missing or invalid lat/lon"}), 400

        now_utc = datetime.now(timezone.utc)
        product, s3_key, checked = pick_best_product_and_key(now_utc)
        file_ts = parse_timestamp_from_key(s3_key)

        ds, io_info = open_dataset_from_s3_key(s3_key)
        da, variable_name = get_data_var(ds)
        da = normalize_data_array(da)

        return jsonify({
            "ok": True,
            "source": "noaa-mrms-aws-debug",
            "input": {
                "lat": lat,
                "lon": lon,
            },
            "selectedProduct": product,
            "selectedKey": s3_key.replace(f"{AWS_BUCKET}/", ""),
            "fileTimestampUtc": file_ts.isoformat() if file_ts else None,
            "checkedProducts": checked,
            "debug": {
                **io_info,
                "dataVars": list(ds.data_vars),
                "coords": list(ds.coords),
                "dims": {str(k): int(v) for k, v in da.sizes.items()},
                "variableName": variable_name,
            },
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


@app.get("/api/mrms-1h")
def api_mrms_1h():
    try:
        lat = num(request.args.get("lat"))
        lon = num(request.args.get("lon"))
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES
        mode = (request.args.get("mode") or "single").strip().lower()

        if lat is None or lon is None:
            return jsonify({"ok": False, "error": "Missing or invalid lat/lon"}), 400
        if lat < -90 or lat > 90 or lon < -180 or lon > 180:
            return jsonify({"ok": False, "error": "lat/lon out of range"}), 400
        if radius_miles <= 0:
            return jsonify({"ok": False, "error": "radiusMiles must be > 0"}), 400
        if mode not in {"single", "weighted"}:
            return jsonify({"ok": False, "error": "mode must be 'single' or 'weighted'"}), 400

        now_utc = datetime.now(timezone.utc)
        product, s3_key, checked = pick_best_product_and_key(now_utc)
        file_ts = parse_timestamp_from_key(s3_key)

        ds, io_info = open_dataset_from_s3_key(s3_key)
        da, variable_name = get_data_var(ds)

        if mode == "single":
            inches = sample_nearest_inches(da, lat, lon)

            return jsonify({
                "ok": True,
                "source": "noaa-mrms-aws",
                "mode": "single",
                "units": "inches",
                "input": {
                    "lat": lat,
                    "lon": lon,
                },
                "selectedProduct": product,
                "selectedKey": s3_key.replace(f"{AWS_BUCKET}/", ""),
                "fileTimestampUtc": file_ts.isoformat() if file_ts else None,
                "variableName": variable_name,
                "hourlyRainInches": round_num(inches, 4),
                "checkedProducts": checked,
                "io": io_info,
            })

        pts = build_weighted_points(lat, lon, radius_miles)
        samples = []
        good = []

        for p in pts:
            try:
                inches = sample_nearest_inches(da, p["lat"], p["lon"])
                rec = {
                    "key": p["key"],
                    "weight": p["weight"],
                    "lat": round_num(p["lat"], 6),
                    "lon": round_num(p["lon"], 6),
                    "inches": round_num(inches, 4),
                    "ok": True,
                }
                good.append(rec)
            except Exception as e:
                rec = {
                    "key": p["key"],
                    "weight": p["weight"],
                    "lat": round_num(p["lat"], 6),
                    "lon": round_num(p["lon"], 6),
                    "inches": None,
                    "ok": False,
                    "error": str(e),
                }
            samples.append(rec)

        if not good:
            return jsonify({
                "ok": False,
                "error": "No usable sample points from decoded MRMS dataset.",
                "selectedProduct": product,
                "selectedKey": s3_key.replace(f"{AWS_BUCKET}/", ""),
                "fileTimestampUtc": file_ts.isoformat() if file_ts else None,
                "variableName": variable_name,
                "checkedProducts": checked,
                "samples": samples,
            }), 502

        used_weight = sum(s["weight"] for s in good)
        weighted_inches = sum((s["inches"] or 0.0) * s["weight"] for s in good) / used_weight

        return jsonify({
            "ok": True,
            "source": "noaa-mrms-aws",
            "mode": "weighted",
            "units": "inches",
            "input": {
                "lat": lat,
                "lon": lon,
                "radiusMiles": radius_miles,
            },
            "selectedProduct": product,
            "selectedKey": s3_key.replace(f"{AWS_BUCKET}/", ""),
            "fileTimestampUtc": file_ts.isoformat() if file_ts else None,
            "variableName": variable_name,
            "weightedHourlyRainInches": round_num(weighted_inches, 4),
            "attemptedPointCount": len(samples),
            "successfulPointCount": len(good),
            "checkedProducts": checked,
            "samples": samples,
            "io": io_info,
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)