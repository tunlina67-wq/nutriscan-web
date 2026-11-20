/* NutriScan AI frontend: Scan → Nutrients Found → Your Result */
const $ = (s) => document.querySelector(s);

const FIELDS = [
  ["energy_kcal", "Energy", "kcal"], ["total_fat_g", "Total fat", "g"],
  ["sat_fat_g", "Saturated fat", "g"], ["carbs_g", "Carbs", "g"],
  ["sugar_g", "Sugar", "g"], ["protein_g", "Protein", "g"],
  ["sodium_mg", "Sodium", "mg"], ["potassium_mg", "Potassium", "mg"],
];
// Values from the grape-drink label used in the proposal (fallback if OCR isn't installed)
const DEMO = {
  name: "Royal-D Grape", serving_size: 330, serving_unit: "ml",
  nutrients: { energy_kcal: 90, total_fat_g: 0, sat_fat_g: 0, carbs_g: 23, sugar_g: 16,
               protein_g: 0, sodium_mg: 280, potassium_mg: 75 },
};
const TITLES = { scan: "NutriScan", loading: "Scanning…", nutrients: "Nutrients Found",
                 result: "Your Result", profile: "My Profile" };
const GOALS = { general: "General health", lose_weight: "Lose weight", build_muscle: "Build muscle",
                reduce_sugar: "Reduce sugar", reduce_salt: "Reduce salt" };

const state = { screen: "scan", history: [], profile: loadProfile(), lastResult: null };

/* ---------------- helpers ---------------- */
function loadProfile() {
  try { return JSON.parse(localStorage.getItem("nutriscan_profile")) || {}; } catch { return {}; }
}
function saveProfile(p) {
  state.profile = p;
  try { localStorage.setItem("nutriscan_profile", JSON.stringify(p)); } catch {}
}
function toast(msg, ms = 3200) {
  const t = $("#toast"); t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), ms);
}
const fmt = (v) => (v == null || v === "" ? "" : Number(v).toLocaleString("en-US", { maximumFractionDigits: 1 }));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function show(screen, push = true) {
  if (push && state.screen !== screen && state.screen !== "loading") state.history.push(state.screen);
  state.screen = screen;
  document.querySelectorAll(".screen").forEach((s) => s.classList.toggle("active", s.id === "scr-" + screen));
  $("#barTitle").textContent = TITLES[screen];
  $("#backBtn").hidden = screen === "scan" || screen === "loading";
  $("#profileBtn").hidden = screen === "profile";
  window.scrollTo({ top: 0 });
}
function back() {
  const prev = state.history.pop() || "scan";
  show(prev, false);
}

/* ---------------- status + profile card ---------------- */
async function loadStatus() {
  const names = { detector: "YOLO table detector", ocr: "OCR (Thai + English)",
                  classifier: "ML grade classifier", llm: "LLM advice" };
  let st = {};
  try { st = await (await fetch("/api/status")).json(); } catch {}
  $("#aiChips").innerHTML = Object.entries(names)
    .map(([k, n]) => `<span class="chip ${st[k] ? "on" : ""}" title="${st[k] ? "Active" : "Not set up yet (see README)"}">${n}</span>`)
    .join("");
}
function renderProfileCard() {
  const p = state.profile;
  if (p.age && p.height_cm && p.weight_kg) {
    $("#pcTitle").textContent = `${p.sex === "male" ? "Male" : "Female"}, ${p.age} y · ${p.weight_kg} kg`;
    $("#pcSub").textContent = `Goal: ${GOALS[p.goal] || GOALS.general}. Tap to edit`;
  } else {
    $("#pcTitle").textContent = "Daily needs: 2,000 kcal";
    $("#pcSub").textContent = "Standard Thai RDI. Set up your profile for personal advice";
  }
}

/* ---------------- profile form ---------------- */
function openProfile() {
  const f = $("#profileForm"), p = state.profile;
  f.reset();
  if (p.sex) f.querySelector(`input[name=sex][value=${p.sex}]`).checked = true;
  ["age", "height_cm", "weight_kg"].forEach((k) => { if (p[k]) f.elements[k].value = p[k]; });
  if (p.activity) f.elements.activity.value = p.activity;
  f.querySelector(`input[name=goal][value=${p.goal || "general"}]`).checked = true;
  show("profile");
}
$("#profileForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const f = e.target, d = Object.fromEntries(new FormData(f));
  const p = { sex: d.sex || "female", activity: d.activity, goal: d.goal || "general" };
  for (const k of ["age", "height_cm", "weight_kg"]) if (d[k]) p[k] = Number(d[k]);
  if (!(p.age && p.height_cm && p.weight_kg)) { toast("Please fill in age, height and weight."); return; }
  saveProfile(p); renderProfileCard(); toast("Profile saved ✓");
  if (state.lastResult) runAnalyze(); else back();
});
$("#clearProfile").addEventListener("click", () => {
  saveProfile({}); renderProfileCard(); toast("Profile cleared"); back();
});

/* ---------------- scanning ---------------- */
function setStep(i) {
  document.querySelectorAll("#loadSteps li").forEach((li) => {
    const n = +li.dataset.step;
    li.className = n < i ? "done" : n === i ? "doing" : "";
  });
}

async function scanFile(file, { demo = false } = {}) {
  if (!file) return;
  if (!file.type.startsWith("image/")) { toast("Please choose an image file."); return; }
  $("#loadingImg").src = URL.createObjectURL(file);
  show("loading"); setStep(0);

  // Advance the step list while the server works (the server does all 3 steps in one call)
  const ticker = (async () => { await sleep(900); setStep(1); await sleep(1600); setStep(2); })();
  const fd = new FormData(); fd.append("image", file);
  let data, ok = false;
  try {
    const res = await fetch("/api/scan", { method: "POST", body: fd });
    data = await res.json(); ok = res.ok;
  } catch { data = { error: "Cannot reach the server. Is app.py running?" }; }
  await ticker; setStep(3); await sleep(350);

  if (ok) {
    fillNutrients({ nutrients: data.nutrients, serving_size: data.serving_size,
                    serving_unit: data.serving_unit, name: demo ? DEMO.name : "" },
                  data);
  } else if (demo) {
    fillNutrients(DEMO, { crop: data.crop, detector: data.detector,
                          warnings: ["OCR isn't installed on this server, so these are the demo label's real values."] });
  } else {
    fillNutrients({ nutrients: {}, serving_unit: "g" },
                  { crop: data.crop, detector: data.detector, warnings: [data.error || "Scan failed."] });
  }
  show("nutrients");
}

function fillNutrients(values, scan = {}) {
  const n = values.nutrients || {};
  $("#productName").value = values.name || "";
  $("#servingSize").value = values.serving_size ?? "";
  $("#servingUnit").value = values.serving_unit || "g";

  $("#nutrientTable").innerHTML = FIELDS.map(([k, label, unit]) => `
    <div class="nrow">
      <label for="n_${k}">${label}</label>
      <div class="val">
        <input id="n_${k}" data-key="${k}" type="number" inputmode="decimal" min="0" step="any"
               value="${n[k] ?? ""}" class="${n[k] == null ? "missing" : ""}" placeholder="–">
        <span class="unit">${unit}</span>
      </div>
    </div>`).join("");
  $("#nutrientTable").querySelectorAll("input").forEach((i) =>
    i.addEventListener("input", () => i.classList.toggle("missing", i.value === "")));

  const img = $("#cropImg");
  if (scan.crop) img.src = scan.crop; else img.removeAttribute("src");
  const det = scan.detector, badge = $("#detBadge");
  if (det && det.found) { badge.textContent = `Table detected · ${Math.round(det.confidence * 100)}%`; badge.className = "det-badge"; }
  else if (det) { badge.textContent = det.used ? "No table found · full photo" : "Full photo used"; badge.className = "det-badge warn"; }
  else { badge.textContent = "Manual entry"; badge.className = "det-badge warn"; }

  const w = scan.warnings || [];
  $("#scanWarnings").hidden = !w.length;
  $("#scanWarnings").innerHTML = w.map((x) => `<p>⚠️ ${esc(x)}</p>`).join("");
  $("#rawBox").hidden = !(scan.ocr_lines && scan.ocr_lines.length);
  $("#rawText").textContent = (scan.ocr_lines || []).join("\n");
}

function readNutrients() {
  const n = {};
  document.querySelectorAll("#nutrientTable input").forEach((i) => {
    n[i.dataset.key] = i.value === "" ? null : Number(i.value);
  });
  return n;
}

/* ---------------- analysis ---------------- */
async function runAnalyze() {
  const nutrients = readNutrients();
  if (["energy_kcal", "sugar_g", "sodium_mg"].every((k) => nutrients[k] == null)) {
    toast("Enter at least energy, sugar or sodium first."); return;
  }
  const btn = $("#analyzeBtn"); btn.disabled = true; btn.textContent = "Analyzing…";
  const body = {
    nutrients, profile: state.profile,
    product_name: $("#productName").value,
    serving_size: Number($("#servingSize").value) || null,
    serving_unit: $("#servingUnit").value,
  };
  try {
    const res = await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error();
    state.lastResult = await res.json();
    renderResult(state.lastResult);
    if (state.screen === "profile") { state.history = ["scan", "nutrients"]; show("result", false); }
    else show("result");
  } catch { toast("Analysis failed. Is the server running?"); }
  finally { btn.disabled = false; btn.innerHTML = 'Analyze <span aria-hidden="true">→</span>'; }
}

function barHTML(it) {
  const cls = it.kind === "target" ? it.level : (it.percent >= 100 ? "over" : it.level);
  return `<div class="bar">
    <div class="bar-top"><b>${it.label}</b>
      <span>${fmt(it.value)} ${it.unit} / ${fmt(it.limit)} ${it.unit}<span class="pct">${fmt(it.percent)}%</span></span></div>
    <div class="track"><div class="fill ${cls}" data-w="${Math.min(it.percent, 100)}"></div></div>
  </div>`;
}

function renderResult(r) {
  const g = $("#gradeBadge");
  g.textContent = r.grade; g.className = "grade " + r.grade;
  g.style.animation = "none"; void g.offsetWidth; g.style.animation = "";
  const p = r.product;
  $("#resProduct").textContent = p.name || "This product";
  $("#gradeSub").textContent = p.serving_size ? `Per serving · ${fmt(p.serving_size)} ${p.serving_unit}` : "Per serving";

  $("#adviceText").textContent = r.advice.text;
  $("#adviceTag").textContent = r.advice.source === "llm" ? "LLM" : "Smart template";

  const order = ["sugar_g", "sodium_mg", "energy_kcal", "sat_fat_g", "total_fat_g", "carbs_g"];
  const limits = r.items.filter((i) => i.kind === "limit").sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key));
  const targets = r.items.filter((i) => i.kind === "target");
  $("#limitBars").innerHTML = limits.map(barHTML).join("");
  $("#targetBars").innerHTML = targets.map(barHTML).join("");
  $("#targetWrap").hidden = !targets.length;
  requestAnimationFrame(() => setTimeout(() =>
    document.querySelectorAll("#scr-result .fill[data-w]").forEach((f) => (f.style.width = f.dataset.w + "%")), 60));

  $("#rulesGrade").textContent = r.rules_grade;
  $("#rulesList").innerHTML = r.rules_fired.map((x) => `<li>${esc(x)}</li>`).join("");
  if (r.ml && r.ml.grade) {
    $("#mlGrade").textContent = r.ml.grade;
    $("#mlBody").innerHTML =
      `<p>Random Forest trained on Open Food Facts${r.ml.test_accuracy ? ` (test accuracy ${(r.ml.test_accuracy * 100).toFixed(1)}%)` : ""}.
       Final grade = average of rules and ML, rounded to the cautious side.</p>` +
      Object.entries(r.ml.probabilities).map(([k, v]) =>
        `<div class="proba"><b>${k}</b><div class="track"><div class="fill" style="width:${v * 100}%"></div></div><span>${(v * 100).toFixed(0)}%</span></div>`).join("");
  } else {
    $("#mlGrade").textContent = "off";
    $("#mlBody").innerHTML = `<p>${r.ml && r.ml.note ? esc(r.ml.note) + ". " : ""}${
      r.ml ? "" : "Model not trained yet: run <code>python train_classifier.py</code>. "}The grade uses the expert rules only.</p>`;
  }
  const nd = r.needs;
  $("#needsText").textContent = nd.personal
    ? `${fmt(nd.energy_need)} kcal/day (Mifflin–St Jeor × activity${nd.goal !== "general" ? ", goal: " + GOALS[nd.goal].toLowerCase() : ""})` +
      `${nd.bmi ? ` · BMI ${nd.bmi}` : ""} · sugar ≤ ${fmt(nd.limits.sugar_g)} g · sodium ≤ ${fmt(nd.limits.sodium_mg)} mg`
    : "Standard 2,000 kcal (Thai RDI). Set up your profile for personal limits.";
}

/* ---------------- events ---------------- */
$("#cameraInput").addEventListener("change", (e) => { scanFile(e.target.files[0]); e.target.value = ""; });
$("#uploadInput").addEventListener("change", (e) => { scanFile(e.target.files[0]); e.target.value = ""; });
$("#demoBtn").addEventListener("click", async () => {
  try {
    const blob = await (await fetch("/static/sample-label.jpg")).blob();
    scanFile(new File([blob], "sample-label.jpg", { type: "image/jpeg" }), { demo: true });
  } catch { fillNutrients(DEMO, {}); show("nutrients"); }
});
$("#manualBtn").addEventListener("click", () => { fillNutrients({ nutrients: {}, serving_unit: "g" }); show("nutrients"); });
$("#analyzeBtn").addEventListener("click", runAnalyze);
$("#editBtn").addEventListener("click", () => show("nutrients"));
$("#againBtn").addEventListener("click", () => { state.history = []; state.lastResult = null; show("scan", false); });
$("#backBtn").addEventListener("click", back);
$("#profileBtn").addEventListener("click", openProfile);
$("#profileCard").addEventListener("click", openProfile);
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  if (state.screen === "scan" && e.dataTransfer.files[0]) scanFile(e.dataTransfer.files[0]);
});

renderProfileCard();
loadStatus();
