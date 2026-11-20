"""Step 6 (Large language model): explain the result in friendly, personalized language.

Set the environment variable ANTHROPIC_API_KEY to use Claude. Without a key, the app
falls back to template advice built from the expert-system flags, so it always works.
"""
import json
import os

import requests

MODEL = os.environ.get("NUTRISCAN_LLM_MODEL", "claude-haiku-5-5")
API_URL = "https://api.anthropic.com/v1/messages"

GOAL_TEXT = {
    "general": "general health", "lose_weight": "losing weight", "build_muscle": "building muscle",
    "reduce_sugar": "cutting down on sugar", "reduce_salt": "cutting down on salt",
}


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _template(product, analysis, grade, profile):
    items = {i["key"]: i for i in analysis["items"]}
    goal = analysis["needs"]["goal"]
    is_drink = product.get("serving_unit") == "ml"
    name = product.get("name") or ("This drink" if is_drink else "This product")
    high = [i for i in analysis["items"]
            if i["kind"] == "limit" and i["level"] == "high" and i["key"] != "carbs_g"]
    parts = []

    if high:
        worst = max(high, key=lambda i: i["percent"])
        parts.append(f"{name} gives you {worst['percent']:.0f}% of your daily "
                     f"{worst['label'].lower()} limit in one serving.")
    else:
        parts.append(f"{name} is fairly moderate. No nutrient goes above 30% of your daily limit.")

    sodium = items.get("sodium_mg")
    sugar = items.get("sugar_g")
    if is_drink and sodium and sodium["value"] >= 150 and sugar and sugar["value"] >= 8:
        parts.append("The salt and sugar make it useful after heavy exercise or sweating, "
                     "but it's too sweet as an everyday drink. Water is better for normal thirst.")
    elif grade in "DE":
        parts.append("Treat it as an occasional item rather than an everyday choice.")
    elif grade == "C":
        parts.append("It's okay sometimes. Balance it with less-processed foods for the rest of the day.")
    else:
        parts.append("It's a reasonable choice for everyday eating.")

    if goal == "reduce_sugar" and sugar and sugar["percent"] >= 20:
        parts.append("Since you're cutting down on sugar, look for an unsweetened version.")
    elif goal == "reduce_salt" and sodium and sodium["percent"] >= 15:
        parts.append("Since you're reducing salt, keep the rest of today's meals low in sodium.")
    elif goal == "lose_weight" and items.get("energy_kcal") and items["energy_kcal"]["percent"] >= 15:
        parts.append("For weight loss, watch the calories. This is a big part of your daily budget.")
    elif goal == "build_muscle" and items.get("protein_g"):
        p = items["protein_g"]
        parts.append(f"For muscle building it adds {p['percent']:.0f}% of your protein target"
                     + (", which is helpful." if p["percent"] >= 20 else ", which is low."))
    return " ".join(parts)


def _llm(product, analysis, grade, profile):
    facts = {
        "product": product,
        "grade": grade,
        "per_serving": [{k: i[k] for k in ("label", "value", "unit", "percent", "kind")}
                        for i in analysis["items"]],
        "user": {"goal": GOAL_TEXT.get(analysis["needs"]["goal"]),
                 "daily_energy_need_kcal": analysis["needs"]["energy_need"],
                 "age": profile.get("age"), "sex": profile.get("sex"),
                 "activity": profile.get("activity")},
        "flags": analysis["flags"],
    }
    prompt = (
        "You are a friendly nutrition assistant inside a label-scanning app. Using ONLY the data "
        "below, write 2–3 short sentences (max 60 words) for the user: the main thing to notice, "
        "when this product is a good or poor choice, and one tip related to their goal. "
        "Percent values are % of the user's personal daily limit (or target, for kind=target). "
        "No diagnosis, no medical claims, no markdown.\n\n" + json.dumps(facts, ensure_ascii=False)
    )
    r = requests.post(API_URL, timeout=20, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01", "content-type": "application/json",
    }, json={"model": MODEL, "max_tokens": 200, "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"]).strip()


def get_advice(product, analysis, grade, profile):
    if llm_available():
        try:
            return {"text": _llm(product, analysis, grade, profile or {}), "source": "llm",
                    "model": MODEL}
        except Exception as e:
            print("LLM advice failed, using template:", e)
    return {"text": _template(product, analysis, grade, profile or {}), "source": "template"}
