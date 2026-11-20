"""Step 4 (Rule-based expert system): compare the product with the user's daily needs
and produce an A–E health grade.

References used for the rules:
- Thai RDI (Thai FDA), based on 2,000 kcal: fat 65 g, saturated fat 20 g, carbohydrate 300 g,
  sodium 2,000 mg
- Thai Department of Health / WHO: free sugar at most 6 teaspoons (24 g) per day
- Mifflin–St Jeor equation for daily energy need
"""
import math

GRADES = "ABCDE"
ACTIVITY = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very_active": 1.9}
GOALS = {
    "general": "General health",
    "lose_weight": "Lose weight",
    "build_muscle": "Build muscle",
    "reduce_sugar": "Reduce sugar",
    "reduce_salt": "Reduce salt",
}

# nutrient -> (label, unit, kind)  kind: "limit" = eat less, "target" = good to get
ITEMS = {
    "energy_kcal":  ("Energy", "kcal", "limit"),
    "sugar_g":      ("Sugar", "g", "limit"),
    "sodium_mg":    ("Sodium", "mg", "limit"),
    "total_fat_g":  ("Total fat", "g", "limit"),
    "sat_fat_g":    ("Saturated fat", "g", "limit"),
    "carbs_g":      ("Carbs", "g", "limit"),
    "protein_g":    ("Protein", "g", "target"),
    "potassium_mg": ("Potassium", "mg", "target"),
}

# Weights of the expert-system risk score (sum = 1.0)
WEIGHTS = {"sugar_g": 0.45, "sodium_mg": 0.25, "sat_fat_g": 0.15, "energy_kcal": 0.15}
THRESHOLDS = [(8, "A"), (18, "B"), (40, "C"), (65, "D")]  # risk < x -> grade, else E


def _num(v, default=None):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def daily_needs(profile: dict | None):
    """Personal daily energy need and nutrient limits/targets."""
    p = profile or {}
    age, h, w = _num(p.get("age")), _num(p.get("height_cm")), _num(p.get("weight_kg"))
    goal = p.get("goal") if p.get("goal") in GOALS else "general"
    personal = None not in (age, h, w)

    if personal:
        bmr = 10 * w + 6.25 * h - 5 * age + (5 if p.get("sex") == "male" else -161)
        energy = bmr * ACTIVITY.get(p.get("activity"), 1.375)
        if goal == "lose_weight":
            energy = max(1200, energy - 500)
        elif goal == "build_muscle":
            energy += 250
        energy = round(energy / 10) * 10
    else:
        energy = 2000

    scale = energy / 2000
    limits = {
        "energy_kcal": energy,
        "sugar_g": round(24 * scale * (0.75 if goal == "reduce_sugar" else 1), 1),
        "sodium_mg": 1500 if goal == "reduce_salt" else 2000,
        "total_fat_g": round(65 * scale),
        "sat_fat_g": round(20 * scale),
        "carbs_g": round(300 * scale),
        "protein_g": round((1.6 if goal == "build_muscle" else 0.8) * w) if personal else 50,
        "potassium_mg": 3500,
    }
    return {"energy_need": energy, "limits": limits, "personal": personal, "goal": goal,
            "bmi": round(w / (h / 100) ** 2, 1) if personal and h > 0 else None}


def _level(pct, kind):
    if kind == "target":
        return "good" if pct >= 20 else "neutral"
    return "low" if pct < 10 else ("medium" if pct <= 30 else "high")


def analyze(nutrients: dict, profile: dict | None = None):
    needs = daily_needs(profile)
    limits = needs["limits"]

    items, pct = [], {}
    for key, (label, unit, kind) in ITEMS.items():
        v = _num(nutrients.get(key))
        if v is None:
            continue
        p = round(100 * v / limits[key], 1) if limits[key] else 0
        pct[key] = p
        items.append({"key": key, "label": label, "value": v, "unit": unit, "kind": kind,
                      "limit": limits[key], "percent": p, "level": _level(p, kind)})

    # ---- expert-system rules ----
    rules = []
    risk = sum(w * min(pct.get(k, 0), 150) for k, w in WEIGHTS.items())
    for k, w in WEIGHTS.items():
        if k in pct:
            rules.append(f"{ITEMS[k][0]}: {pct[k]:g}% of daily limit × weight {w} "
                         f"= {w * min(pct[k], 150):.1f} risk points")

    prot = pct.get("protein_g", 0)
    if prot >= 20:
        risk -= 4
        rules.append("IF protein ≥ 20% of daily target THEN −4 risk points")
    elif prot >= 10:
        risk -= 2
        rules.append("IF protein ≥ 10% of daily target THEN −2 risk points")
    risk = max(0.0, risk)

    grade = next((g for limit, g in THRESHOLDS if risk < limit), "E")
    rules.append(f"Total risk {risk:.1f} → grade {grade} "
                 f"(A<8, B<18, C<40, D<65, E≥65)")

    over = [ITEMS[k][0] for k in WEIGHTS if pct.get(k, 0) >= 100]
    if over and GRADES.index(grade) < GRADES.index("D"):
        grade = "D"
        rules.append(f"IF any nutrient ≥ 100% of daily limit ({', '.join(over)}) THEN grade at least D")

    flags = []
    for it in items:
        if it["kind"] == "limit" and it["level"] == "high" and it["key"] != "carbs_g":
            flags.append(f"High {it['label'].lower()}: {it['percent']:g}% of your daily limit")
        if it["kind"] == "target" and it["level"] == "good":
            flags.append(f"Good source of {it['label'].lower()}: {it['percent']:g}% of your daily target")

    return {"rules_grade": grade, "risk_score": round(risk, 1), "rules_fired": rules,
            "items": items, "flags": flags, "needs": needs}


def combine(rules_grade: str, ml_grade: str | None):
    """Final grade = average of expert rules and ML model, rounded to the more cautious side."""
    if not ml_grade or ml_grade not in GRADES:
        return rules_grade
    return GRADES[math.ceil((GRADES.index(rules_grade) + GRADES.index(ml_grade)) / 2)]
