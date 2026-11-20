"""NutriScan AI web server.

Run:   python app.py      then open  http://localhost:5000
"""
import base64
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from nutriscan import advice, classifier, detector, ocr, parser, scoring

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB photos
_ai_lock = threading.Lock()


def _jpeg_b64(img, max_side=700):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    """Which AI parts are active (shown in the app so the demo is transparent)."""
    with _ai_lock:  # the first call loads the models; don't load them twice in parallel
        return jsonify({
            "detector": detector.available(),
            "ocr": ocr.available(),
            "classifier": classifier.available(),
            "llm": advice.llm_available(),
        })


@app.post("/api/scan")
def scan():
    """Photo → YOLO crop → OCR → structured nutrients."""
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image uploaded"}), 400
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read this image file"}), 400

    # One scan at a time: the models aren't thread-safe and each scan uses all CPU cores
    with _ai_lock:
        return _scan_image(img)


def _scan_image(img):
    t0 = time.time()
    crop, det = detector.detect_and_crop(img)
    t_det = time.time() - t0

    if not ocr.available():
        return jsonify({"error": "OCR engine (EasyOCR) is not installed on the server. "
                                 "You can type the values in manually.",
                        "ocr_available": False, "detector": det,
                        "crop": _jpeg_b64(crop)}), 503

    t0 = time.time()
    lines = ocr.read_lines(crop)
    nutrients, method = parser.parse_nutrients(lines, return_method=True)

    # YOLO can box the wrong area or only part of the table. If the crop gave
    # few values, also read the whole photo and keep whichever found more.
    def n_read(m):
        return sum(1 for v in m.values() if v != "estimated")

    if det.get("found") and n_read(method) < 4:
        full_lines = ocr.read_lines(img)
        full_n, full_m = parser.parse_nutrients(full_lines, return_method=True)
        if n_read(full_m) > n_read(method):
            lines, nutrients, method, crop = full_lines, full_n, full_m, img
            det = {**det, "found": False, "note": "Crop was incomplete, used full photo"}
    t_ocr = time.time() - t0

    size, unit = parser.parse_serving(lines)
    warnings = parser.validate(nutrients)
    if method.get("energy_kcal") == "estimated":
        warnings.insert(0, "Energy couldn't be read, so it was estimated from carbs, "
                           "protein and fat. Check it against the label.")
    return jsonify({
        "nutrients": nutrients,
        "serving_size": size, "serving_unit": unit,
        "warnings": warnings,
        "ocr_lines": lines,
        "detector": det,
        "crop": _jpeg_b64(crop),
        "timing": {"detect_s": round(t_det, 2), "ocr_s": round(t_ocr, 2)},
    })


@app.post("/api/analyze")
def analyze():
    """Nutrients + user profile → expert rules + ML grade + AI advice."""
    data = request.get_json(force=True, silent=True) or {}
    nutrients = data.get("nutrients") or {}
    profile = data.get("profile") or {}
    product = {"name": (data.get("product_name") or "").strip()[:80],
               "serving_size": data.get("serving_size"),
               "serving_unit": data.get("serving_unit") or "g"}

    result = scoring.analyze(nutrients, profile)
    ml = classifier.predict(nutrients, product["serving_size"], product["serving_unit"])
    grade = scoring.combine(result["rules_grade"], (ml or {}).get("grade"))
    adv = advice.get_advice(product, result, grade, profile)

    return jsonify({**result, "grade": grade, "ml": ml, "advice": adv,
                    "warnings": parser.validate(nutrients), "product": product})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n  NutriScan AI running → http://localhost:%d\n" % port)
    app.run(host="0.0.0.0", port=port, debug=False)
