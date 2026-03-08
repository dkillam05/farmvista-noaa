import math
import os
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request

IMPORT_ERROR = None
try:
    import fsspec
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


def num(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except Exception:
        return None


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


@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "FarmVista NOAA MRMS metadata test",
        "routes": {
            "healthz": "/healthz",
            "mrmsTest": "/api/mrms-1h?lat=39.7898&lon=-91.2059&radiusMiles=0.5&mode=weighted",
        },
    })


@app.get("/healthz")
def healthz():
    return jsonify({
        "ok": True,
        "pythonReady": IMPORT_ERROR is None,
        "importError": IMPORT_ERROR,
        "service": "FarmVista NOAA MRMS metadata test",
    })


@app.get("/api/mrms-1h")
def api_mrms_1h():
    try:
        lat = num(request.args.get("lat"))
        lon = num(request.args.get("lon"))
        radius_miles = num(request.args.get("radiusMiles")) or DEFAULT_RADIUS_MILES
        mode = (request.args.get("mode") or "weighted").strip().lower()

        if lat is None or lon is None:
            return jsonify({"ok": False, "error": "Missing or invalid lat/lon"}), 400
        if lat < -90 or lat > 90 or lon < -180 or lon > 180:
            return jsonify({"ok": False, "error": "lat/lon out of range"}), 400
        if radius_miles <= 0:
            return jsonify({"ok": False, "error": "radiusMiles must be > 0"}), 400

        now_utc = datetime.now(timezone.utc)
        product, s3_key, checked = pick_best_product_and_key(now_utc)
        file_ts = parse_timestamp_from_key(s3_key)

        return jsonify({
            "ok": True,
            "message": "MRMS AWS listing is working.",
            "source": "noaa-mrms-aws",
            "input": {
                "lat": lat,
                "lon": lon,
                "radiusMiles": radius_miles,
                "mode": mode,
            },
            "selectedProduct": product,
            "selectedKey": s3_key.replace(f"{AWS_BUCKET}/", ""),
            "fileTimestampUtc": file_ts.isoformat() if file_ts else None,
            "checkedProducts": checked,
            "nextStep": "AWS bucket access is confirmed. Safe base is restored.",
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
