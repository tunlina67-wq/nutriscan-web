"""Step 3 (Information extraction): turn OCR text lines into structured nutrients."""
import re
from difflib import SequenceMatcher

# Keywords in English and Thai (more specific keywords first)
NUTRIENTS = {
    "energy_kcal":  ["energy", "calories", "พลังงาน"],
    "total_fat_g":  ["total fat", "ไขมันทั้งหมด"],
    "sat_fat_g":    ["saturated", "sat. fat", "sat fat", "ไขมันอิ่มตัว"],
    "protein_g":    ["protein", "โปรตีน"],
    "carbs_g":      ["carbohydrate", "total carb", "คาร์โบไฮเดรตทั้งหมด", "คาร์โบไฮเดรต", "คาร์โบ"],
    "sugar_g":      ["sugar", "น้ำตาล", "น้ำตาลทั้งหมด"],
    "sodium_mg":    ["sodium", "โซเดียม"],
    "potassium_mg": ["potassium", "โพแทสเซียม"],
}

LABELS = {
    "energy_kcal": "Energy", "total_fat_g": "Total fat", "sat_fat_g": "Saturated fat",
    "protein_g": "Protein", "carbs_g": "Carbs", "sugar_g": "Sugar",
    "sodium_mg": "Sodium", "potassium_mg": "Potassium",
}
UNITS = {k: k.rsplit("_", 1)[1] for k in NUTRIENTS}

NUM = re.compile(r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?")
SERVING = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|มล|มิลลิลิตร|g\b|กรัม|ก\.)")
SERVING_HINTS = ["serving", "per", "หนึ่งครั้ง", "ขวด", "bottle", "ซอง", "กล่อง"]
# Lines that state the serving size itself (checked before the looser hints)
SERVING_SIZE_HINTS = ["serving size", "serv size", "serv. size",
                      "หนึ่งหน่วยบริโภค", "หนึงหน่วยบริโภค", "กินได้"]


def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"(?<![a-z])o(?=\s*(?:g\b|mg\b|kcal|ก|มก))", "0", t)  # 'O g' -> '0 g'
    t = re.sub(r"(?<=\d)\s*,(?=\d(?!\d\d))", ".", t)                # '2,5' -> '2.5'
    return t


# Label words that are NOT one of our nutrients. A misread line must look more
# like a nutrient than like any of these before the fuzzy pass accepts it.
OTHER_WORDS = ["cholesterol", "คอเลสเตอรอล", "โคเลสเตอรอล", "fiber", "ใยอาหาร", "vitamin",
               "วิตามิน", "calcium", "แคลเซียม", "iron", "เหล็ก", "ฟอสฟอรัส", "หน่วยบริโภค",
               "ร้อยละ", "ข้อมูลโภชนาการ", "คุณค่าทางโภชนาการ", "ไขมันทรานส์", "trans fat",
               "cholest", "fat cal", "calories from fat", "พลังงานจากไขมัน", "serv size",
               "servings", "percent", "daily value", "dietary fiber",
               "ความต้องการพลังงานวันละ", "วันละ", "calorie diet"]
FUZZY_MIN = 0.5
ALL_KEYWORDS = [kw for kws in NUTRIENTS.values() for kw in kws]


def _chunks(line):
    """Split a line into (label, number) pairs. Labels often sit in two columns,
    e.g. 'total fat 12g 18% total carb. 23g' -> ('total fat', 12), ('total carb.', 23)."""
    prev = 0
    for m in NUM.finditer(line):
        yield line[prev:m.start()], m
        prev = m.end()


def _clean_label(seg):
    seg = re.sub(r"^\s*(?:mg|g|kcal|%|มก\.?|ก\.?)", "", seg)   # unit left over from previous number
    return re.sub(r"[\s().,:*%<>\-]", "", seg)


def _similarity(word, label):
    """Best match of `word` against the start or the end of `label` (0..1)."""
    if not label:
        return 0.0
    best = 0.0
    for e in (0, 2, 4):
        n = len(word) + e
        best = max(best, SequenceMatcher(None, word, label[:n]).ratio(),
                   SequenceMatcher(None, word, label[-n:]).ratio())
    return best


def parse_nutrients(lines, return_method=False):
    """Turn OCR lines into nutrient values.

    Pass 1: exact keyword, then the first number after it.
    Pass 2: fuzzy match for misread Thai (e.g. 'โซเคียม' for 'โซเดียม').
    Pass 3: estimate energy from carbs/protein/fat if it couldn't be read.
    With return_method=True also returns how each value was found.
    """
    out, method = {k: None for k in NUTRIENTS}, {}
    texts = [clean_text(line) for line in lines]

    for idx, l in enumerate(texts):
        for key, kws in NUTRIENTS.items():
            if out[key] is not None:
                continue
            for kw in kws:
                i = l.find(kw)
                if i == -1:
                    continue
                if key == "total_fat_g" and ("saturated" in l or "อิ่มตัว" in l):
                    continue
                m = NUM.search(l, i + len(kw))
                if m:
                    out[key] = float(m.group().replace(",", ""))
                    method[key] = "exact"
                    break

    for l in texts:
        for seg, m in _chunks(l):
            if any(kw in seg for kw in ALL_KEYWORDS):
                continue                      # pass 1 already handled this label
            label = _clean_label(seg)
            if len(label) < 3:
                continue
            best_key, best = None, 0.0
            for key, kws in NUTRIENTS.items():
                for kw in kws:
                    s = _similarity(kw.replace(" ", ""), label)
                    if s > best:
                        best_key, best = key, s
            other = max(_similarity(w.replace(" ", ""), label) for w in OTHER_WORDS)
            if best >= FUZZY_MIN and best > other and out[best_key] is None:
                out[best_key] = float(m.group().replace(",", ""))
                method[best_key] = f"fuzzy {best:.2f}"

    c, p, f = out["carbs_g"], out["protein_g"], out["total_fat_g"]
    if out["energy_kcal"] is None and None not in (c, p, f):
        out["energy_kcal"] = float(round(4 * c + 4 * p + 9 * f))
        method["energy_kcal"] = "estimated"

    return (out, method) if return_method else out


def parse_serving(lines):
    """Serving size, e.g. 'Amount per serving: 1 Bottle (330 ml)' -> (330, 'ml')."""
    candidates = [clean_text(l) for l in lines]
    sized = [l for l in candidates if any(h in l for h in SERVING_SIZE_HINTS)]
    hinted = [l for l in candidates if any(h in l for h in SERVING_HINTS)]
    for l in sized + hinted + candidates:
        for m in SERVING.finditer(l):
            if float(m.group(1)) == 0:        # e.g. 'trans fat 0g' is not a serving size
                continue
            unit = "ml" if m.group(2) in ("ml", "มล", "มิลลิลิตร") else "g"
            return float(m.group(1)), unit
    return None, None


def validate(n):
    """Sanity checks that catch OCR mistakes."""
    warnings = []
    e, c, p, f = (n.get(k) for k in ("energy_kcal", "carbs_g", "protein_g", "total_fat_g"))
    if None not in (e, c, p, f) and e > 0:
        calc = 4 * c + 4 * p + 9 * f
        if abs(calc - e) > max(25, 0.25 * e):
            warnings.append(f"Energy {e:g} kcal doesn't match 4×carbs + 4×protein + 9×fat "
                            f"≈ {calc:g} kcal. Please check the values.")
    s = n.get("sugar_g")
    if s is not None and c is not None and s > c + 0.5:
        warnings.append("Sugar is larger than total carbohydrate. One of them is probably misread.")
    sf = n.get("sat_fat_g")
    if sf is not None and f is not None and sf > f + 0.5:
        warnings.append("Saturated fat is larger than total fat. One of them is probably misread.")
    missing = [LABELS[k] for k in ("energy_kcal", "sugar_g", "sodium_mg") if n.get(k) is None]
    if missing:
        warnings.append("Could not read: " + ", ".join(missing) + ". Please type them in.")
    return warnings
