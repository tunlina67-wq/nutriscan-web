"""Step 2 (Image preprocessing + Deep-learning OCR): read Thai and English text."""
import cv2

_reader = None
_load_error = None


def _load():
    global _reader, _load_error
    if _reader is not None or _load_error is not None:
        return _reader
    try:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _reader = easyocr.Reader(["th", "en"], gpu=gpu)
    except Exception as e:
        _load_error = str(e)
    return _reader


def available() -> bool:
    return _load() is not None


def preprocess(img):
    """Upscale small images, convert to grayscale, boost contrast with CLAHE."""
    if img.shape[1] < 1000:
        s = 1000 / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    elif img.shape[1] > 1800:  # very large photos are slow and don't help
        s = 1800 / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def group_lines(ocr_results, y_tol: float = 0.6):
    """Merge OCR word boxes that sit on the same row into text lines."""
    items = []
    for box, text, _conf in ocr_results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append(((min(ys) + max(ys)) / 2, min(xs), max(ys) - min(ys), text))
    items.sort()
    lines = []
    for yc, x, h, t in items:
        if lines and abs(yc - lines[-1]["y"]) < y_tol * max(h, lines[-1]["h"]):
            lines[-1]["parts"].append((x, t))
        else:
            lines.append({"y": yc, "h": h, "parts": [(x, t)]})
    return [" ".join(t for _, t in sorted(l["parts"])) for l in lines]


def read_lines(img):
    reader = _load()
    if reader is None:
        raise RuntimeError("EasyOCR is not installed: " + str(_load_error))
    return group_lines(reader.readtext(preprocess(img)))
