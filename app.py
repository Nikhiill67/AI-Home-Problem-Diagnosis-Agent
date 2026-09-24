import os, re, json
from typing import TypedDict
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "troubleshooter-dev-key")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CATEGORIES = {
    "wifi": ["wifi", "wi-fi", "internet", "router", "network", "disconnect"],
    "laptop": ["laptop", "computer", "pc", "slow", "freezing", "lag", "overheating"],
    "printer": ["printer", "printing", "print", "paper jam", "ink"],
    "phone": ["phone", "mobile", "charging", "charger", "battery", "usb"],
    "vehicle": ["bike", "car", "vehicle", "brake", "braking", "steering", "engine", "starting"],
    "fan": ["fan", "ceiling fan", "clicking", "buzzing", "noise"],
    "washing_machine": ["washing machine", "washer", "drain", "draining", "spin cycle"],
    "study": ["study", "studying", "concentrate", "focus", "forgetting", "tasks"],
    "general": []
}

FALLBACK = {
    "wifi": {
        "questions": ["Does the problem happen only on one device or on several devices?",
                      "Can the affected device connect to Wi-Fi but not reach websites, or does it fail to connect at all?",
                      "Have you restarted the affected device and the router since the problem started?"],
        "causes": [("Device-specific network settings or adapter issue", "The phone works, so the connection may be healthy while the laptop has a local issue.", "common"),
                   ("Driver or network configuration problem", "A laptop-only failure can be related to its Wi-Fi adapter software or saved network settings.", "possible"),
                   ("Router compatibility or access-control issue", "A router setting can affect one device differently, but there is not enough evidence yet.", "uncommon")],
        "checks": ["Restart the laptop and reconnect to the same Wi-Fi network.", "Forget the Wi-Fi network on the laptop and reconnect using the correct password.", "Check whether another laptop or computer can use the same Wi-Fi."],
        "avoid": ["Do not change advanced router settings or reset the router unless you know the current configuration."],
        "urgency": "low", "help": "If multiple devices lose connectivity or the router repeatedly fails, contact your internet provider or a network technician."
    },
    "laptop": {
        "questions": ["Does the slowdown happen with a particular app or with the whole system?", "Does the laptop become hot or does its fan become unusually loud?", "How much free storage and RAM does the laptop have, if you know?"],
        "causes": [("Background processes or high resource use", "Performance that degrades after several minutes can occur when a process consumes CPU or memory.", "common"),
                   ("Heat-related performance reduction", "If the system becomes hot before slowing, thermal limits are a plausible explanation.", "possible"),
                   ("Storage or hardware issue", "Low free space or aging hardware can contribute, but more information is needed.", "possible")],
        "checks": ["Open the system task manager and observe CPU, memory, disk and temperature information if available.", "Close unused applications and browser tabs and see whether performance changes.", "Check available storage space."],
        "avoid": ["Do not open the laptop or remove internal cooling parts unless you are qualified to do so."],
        "urgency": "medium", "help": "Seek a technician if the laptop becomes extremely hot, shuts down unexpectedly, smells burnt, or performance remains poor after basic checks."
    },
    "printer": {
        "questions": ["Does the printer power on normally?", "Is there an error message or blinking warning light?", "Is the problem from one device or every device that tries to print?"],
        "causes": [("Connection or queue problem", "A stuck print queue or lost connection can prevent jobs from reaching the printer.", "common"),
                   ("Paper, ink or toner condition", "Consumables or a paper path issue can stop printing.", "possible"),
                   ("Driver/software mismatch", "A device-specific failure can be caused by its printer software.", "possible")],
        "checks": ["Check the printer display or indicator lights for a message.", "Confirm paper is loaded correctly and the printer reports adequate ink/toner.", "Cancel stuck print jobs and retry a small test page."],
        "avoid": ["Do not force paper out if it is tightly jammed; follow the printer's manual procedure."],
        "urgency": "low", "help": "Contact the printer manufacturer or a technician if it reports an internal hardware fault."
    },
    "phone": {
        "questions": ["Does the phone show that it is charging at all?", "Have you tried a different known-good cable and charger?", "Is the charging port visibly blocked or damaged?"],
        "causes": [("Cable or charger problem", "A different known-good charging setup can quickly distinguish an accessory issue.", "common"),
                   ("Debris or damage at the charging port", "A blocked or damaged port can prevent a reliable connection.", "possible"),
                   ("Battery or charging-circuit issue", "Persistent failure with known-good accessories can point to an internal issue.", "possible")],
        "checks": ["Try a known-good compatible cable and charger.", "Inspect the port visually using normal lighting without inserting objects.", "Check whether the phone shows any charging symbol or warning."],
        "avoid": ["Do not insert metal objects, liquids or force anything into the charging port."],
        "urgency": "medium", "help": "Stop charging and seek professional service if the phone or battery becomes unusually hot, swollen, leaks, or smells burnt."
    },
    "vehicle": {
        "questions": ["What part of the vehicle is affected and when does the sound or failure happen?", "Is braking or steering performance reduced, or are warning lights showing?", "Did the problem start suddenly or after maintenance or an impact?"],
        "causes": [("Normal wear or a consumable component", "Vehicle noises can come from components that wear over time, but the exact source is not confirmed.", "common"),
                   ("Loose, worn or misaligned component", "A change after an impact or maintenance can make this plausible.", "possible"),
                   ("Safety-critical brake or steering fault", "Reduced braking/steering or warning indicators can signal a serious issue.", "uncommon")],
        "checks": ["Observe warning lights and changes in braking or steering without taking risks.", "If safe, note whether the symptom occurs while stationary, moving slowly, braking, accelerating or turning.", "Arrange inspection by a qualified mechanic when the cause is unclear."],
        "avoid": ["Do not continue riding/driving if braking or steering is impaired, or if there is a major warning sign.", "Do not attempt brake or steering repairs without appropriate training."],
        "urgency": "high", "help": "If braking or steering is impaired, stop using the vehicle when safe and contact a qualified mechanic or appropriate roadside/emergency service."
    },
    "fan": {
        "questions": ["Does the clicking change with fan speed?", "Is the fan wobbling, rubbing, or making any burning smell?", "Does the sound occur continuously or only when starting/stopping?"],
        "causes": [("Loose external part or cover", "A repeating click can come from a component moving slightly during rotation.", "common"),
                   ("Blade or motor-area contact", "A sound that follows fan speed may indicate contact or alignment, but this is not confirmed.", "possible"),
                   ("Electrical or motor fault", "Burning smell, sparking or abnormal heat would make an electrical fault more concerning.", "uncommon")],
        "checks": ["Turn the fan off and observe whether any visible part appears loose or misaligned.", "Note whether the sound follows the rotation speed.", "If there is burning smell, smoke, sparking or unusual heat, switch it off if safe and leave it isolated."],
        "avoid": ["Do not open the motor housing or work on wiring while the fan is connected to power."],
        "urgency": "medium", "help": "Use a qualified electrician or appliance technician if the noise persists, especially with heat, smell, sparks or electrical symptoms."
    },
    "washing_machine": {
        "questions": ["Does the machine complete the wash cycle but leave water inside?", "Can you hear the drain pump running?", "Is an error code displayed?"],
        "causes": [("Blocked drain path or filter", "A blocked path can prevent water from leaving the machine.", "common"),
                   ("Drain hose positioning or blockage", "A kink or blockage can restrict drainage.", "possible"),
                   ("Drain pump fault", "If the pump does not operate or only hums, a pump issue is possible.", "possible")],
        "checks": ["Switch the machine off and consult its manual for the location and safe procedure for checking the drain filter.", "Visually check the external drain hose for obvious kinks.", "Record any displayed error code before contacting service."],
        "avoid": ["Do not reach into the machine or remove panels while it is powered.", "Do not force open a locked door containing water."],
        "urgency": "medium", "help": "Contact appliance service if safe basic checks do not restore drainage or if there is leakage, electrical smell or unusual noise."
    },
    "study": {
        "questions": ["When you try to study, what usually interrupts you first: phone, tiredness, unclear material, or something else?", "How long can you normally focus before the problem starts?", "Does this difficulty happen only for studying or across other daily activities too?"],
        "causes": [("Distractions or task design", "Frequent interruptions or an unclear task can make sustained attention harder.", "common"),
                   ("Fatigue or inconsistent routine", "Sleep, breaks and workload can affect concentration.", "possible"),
                   ("A broader attention or wellbeing concern", "If the difficulty is persistent across many activities, professional guidance may be useful; this does not establish a diagnosis.", "possible")],
        "checks": ["Try one clearly defined 20–25 minute task with notifications disabled.", "Write down the next single action before starting rather than studying an entire topic at once.", "Track when concentration drops and what was happening immediately beforehand."],
        "avoid": ["Do not assume a medical condition from concentration difficulties alone."],
        "urgency": "low", "help": "If concentration problems are persistent, significantly affect daily life, or come with distressing symptoms, consider speaking with a qualified health professional."
    },
    "general": {
        "questions": ["What exactly happens, and what would you expect to happen instead?", "When did the problem start, and is it constant or intermittent?", "What changed just before the problem began, such as a setting, update, repair, impact or new equipment?"],
        "causes": [("A configuration, connection or usage issue", "Many everyday problems are caused by a recent setting or connection change.", "common"),
                   ("A worn or failing component", "Physical or repeated-use problems can come from component wear.", "possible"),
                   ("A less common underlying fault", "Other causes remain possible until more specific evidence is available.", "uncommon")],
        "checks": ["Describe the symptom precisely and note when it happens.", "Check for visible error messages, warning indicators, unusual heat, smell, smoke or leaks.", "Compare with a known-good cable, device, setting or operating condition when safe."],
        "avoid": ["Do not open powered equipment or attempt repairs involving electricity, gas, vehicles, chemicals or structural damage without appropriate expertise."],
        "urgency": "low", "help": "If there is a safety risk or the problem persists after safe checks, contact a qualified professional."
    }
}

class State(TypedDict):
    messages: list
    category: str
    missing: list
    questions: list
    analysis: dict
    needs_questions: bool

def clean_json(raw):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I | re.M)
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}

def llm_json(prompt):
    if not API_KEY:
        return {}
    try:
        llm = ChatGoogleGenerativeAI(model=MODEL, google_api_key=API_KEY, temperature=0.1)
        return clean_json(llm.invoke([HumanMessage(content=prompt)]).content)
    except Exception:
        return {}

def classify_problem(state):
    text = " ".join(m["content"] for m in state["messages"][-6:])
    if API_KEY:
        data = llm_json(f"""Classify this everyday troubleshooting conversation into exactly one category:
wifi,laptop,printer,phone,vehicle,fan,washing_machine,study,general.
Return JSON only: {{"category":"..."}}
Conversation:
{text}""")
        if data.get("category") in FALLBACK:
            return {"category": data["category"]}
    low = text.lower()
    scores = {c: sum(1 for k in keys if k in low) for c, keys in CATEGORIES.items()}
    return {"category": max(scores, key=scores.get)}

def determine_missing_information(state):
    cat, msgs = state["category"], state["messages"]
    if API_KEY:
        data = llm_json(f"""You are a cautious troubleshooting assistant. Category: {cat}.
Conversation: {json.dumps(msgs[-8:])}
Return JSON only with:
{{"missing":["short missing facts"],"questions":["up to 3 targeted questions"],"needs_questions":true_or_false}}
Set needs_questions true only when answers would materially change the safe diagnosis. Never ask for secrets or unnecessary personal data.""")
        if isinstance(data.get("questions"), list) and data.get("questions"):
            return {"missing": data.get("missing", [])[:3], "questions": data["questions"][:3],
                    "needs_questions": bool(data.get("needs_questions", True))}
    kb = FALLBACK[cat]
    return {"missing": kb["questions"][:3], "questions": kb["questions"][:3], "needs_questions": len(msgs) <= 1}

def analyze_problem(state):
    cat, msgs = state["category"], state["messages"]
    text = " ".join(m["content"] for m in msgs).lower()
    danger_terms = [
        "electric shock", "sparking", "smoke", "fire", "gas leak", "gas smell",
        "chemical spill", "toxic chemical", "structural crack", "ceiling collapse",
        "brake failure", "brakes failed", "steering failure", "can't steer",
        "unconscious", "not breathing", "severe bleeding", "chest pain"
    ]
    if any(term in text for term in danger_terms):
        return {"analysis": {
            "problem_type": cat if cat != "general" else "safety-critical",
            "summary": "The information includes a potentially unsafe condition, so troubleshooting should stop at safe observation.",
            "possible_causes": [{"cause": "A safety-critical fault may be present", "reason": "The reported symptom can involve a risk that cannot be safely assessed remotely.", "likelihood": "possible"}],
            "safe_checks": ["Move away from the immediate hazard when safe to do so.", "Observe and report warning signs, error messages or visible damage without touching the hazard."],
            "avoid": ["Do not attempt repair, open equipment, handle leaking chemicals/gas, or continue using a vehicle with impaired braking/steering.", "Use emergency services when there is an immediate threat to life or safety."],
            "urgency": "high",
            "professional_help": "Contact the appropriate qualified professional or emergency service now when there is an immediate danger.",
            "next_question": ""
        }}
    if API_KEY:
        data = llm_json(f"""Analyze this troubleshooting conversation. Do not claim certainty.
Category: {cat}
Conversation: {json.dumps(msgs[-10:])}
Return JSON only:
{{"problem_type":"{cat}","summary":"brief evidence-based summary","possible_causes":[{{"cause":"","reason":"","likelihood":"common|possible|uncommon"}}],
"safe_checks":["only low-risk observational/basic checks"],"avoid":["unsafe actions"],"urgency":"low|medium|high",
"professional_help":"when qualified help is appropriate","next_question":""}}
Safety: no medical diagnosis. For electricity, gas, fire, structural damage, serious vehicle braking/steering, hazardous chemicals or medical emergencies, do not give repair instructions; recommend stopping unsafe activity and qualified/emergency help. Use only facts provided.""")
        if data.get("possible_causes"):
            data["problem_type"] = cat
            data["possible_causes"] = data["possible_causes"][:4]
            return {"analysis": data}
    kb = FALLBACK[cat]
    return {"analysis": {"problem_type": cat, "summary": "Based on the information provided, these are plausible explanations; more evidence may be needed.",
                         "possible_causes": [{"cause": a, "reason": b, "likelihood": c} for a,b,c in kb["causes"]],
                         "safe_checks": kb["checks"], "avoid": kb["avoid"], "urgency": kb["urgency"],
                         "professional_help": kb["help"], "next_question": ""}}

def generate_safe_solution(state):
    a = state["analysis"]
    if a.get("urgency") not in ("low", "medium", "high"):
        a["urgency"] = "medium"
    if not a.get("avoid"):
        a["avoid"] = ["Do not attempt risky repairs without appropriate expertise."]
    return {"analysis": a}

graph = StateGraph(State)
graph.add_node("classify_problem", classify_problem)
graph.add_node("determine_missing_information", determine_missing_information)
graph.add_node("analyze_problem", analyze_problem)
graph.add_node("generate_safe_solution", generate_safe_solution)
graph.add_edge(START, "classify_problem")
graph.add_edge("classify_problem", "determine_missing_information")
graph.add_conditional_edges("determine_missing_information", lambda s: "ask" if s["needs_questions"] else "analyze",
                            {"ask": END, "analyze": "analyze_problem"})
graph.add_edge("analyze_problem", "generate_safe_solution")
graph.add_edge("generate_safe_solution", END)
agent_graph = graph.compile()

HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Troubleshooter</title><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#080b14;--card:rgba(18,25,43,.82);--border:rgba(255,255,255,.09);--p:#6366f1;--text:#f8fafc;--muted:#94a3b8;--ok:#10b981;--warn:#f59e0b;--danger:#ef4444}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#20204b 0,#080b14 58%,#05070c 100%);color:var(--text);font:14px 'Plus Jakarta Sans',system-ui;padding:24px 14px;min-height:100vh}.wrap{max-width:900px;margin:auto;display:grid;gap:15px}.card{background:var(--card);backdrop-filter:blur(15px);border:1px solid var(--border);border-radius:18px;padding:20px;box-shadow:0 12px 35px #0005}.head{display:flex;justify-content:space-between;align-items:center;gap:10px}.brand{display:flex;gap:12px;align-items:center}.logo{font-size:23px;background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:11px;border-radius:13px}h1{margin:0;font-size:1.35rem}.sub,.muted{color:var(--muted);font-size:.82rem}.badge{border:1px solid #6366f155;background:#6366f11c;color:#c7d2fe;border-radius:999px;padding:6px 10px;font-size:.73rem}.chat{height:420px;overflow:auto;display:flex;flex-direction:column;gap:10px;background:#080d18aa;border:1px solid var(--border);border-radius:14px;padding:14px}.bubble{max-width:86%;padding:11px 14px;border-radius:15px;line-height:1.5;white-space:pre-wrap}.ai{align-self:flex-start;background:#1e293bcc;border:1px solid var(--border)}.user{align-self:flex-end;background:linear-gradient(135deg,#6366f1,#4f46e5)}.composer{display:flex;gap:8px;margin-top:10px}.composer input{flex:1;background:#0b1120;border:1px solid var(--border);color:white;border-radius:11px;padding:12px;outline:0}.btn{border:0;border-radius:11px;padding:11px 17px;background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;font-weight:700;cursor:pointer}.result{display:grid;gap:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.section{border:1px solid var(--border);background:#0c1321aa;border-radius:13px;padding:14px}.section h3{margin:0 0 9px;font-size:.95rem}.cause{border:1px solid var(--border);border-radius:11px;padding:11px;margin:8px 0}.cause b{display:block}.pill{display:inline-block;font-size:.7rem;padding:3px 7px;border-radius:999px;margin-top:6px;background:#6366f122;color:#c7d2fe}.urgency{font-weight:800}.low{color:#34d399}.medium{color:#fbbf24}.high{color:#f87171}.checks,.avoid{margin:0;padding-left:20px;line-height:1.7}.clear{float:right;background:none;border:0;color:var(--muted);cursor:pointer}.typing{opacity:.65}@media(max-width:650px){.grid{grid-template-columns:1fr}.head{align-items:flex-start}.chat{height:390px}}
</style></head><body><main class="wrap">
<section class="card head"><div class="brand"><div class="logo">🛠️</div><div><h1>AI Troubleshooter</h1><div class="sub">Practical diagnosis for everyday problems</div></div></div><span class="badge">LangGraph • Gemini</span></section>
<section class="card"><div id="chat" class="chat"><div class="bubble ai">Tell me what is going wrong. I’ll ask a few targeted questions before suggesting safe checks.</div></div>
<div class="composer"><input id="input" placeholder="Example: My Wi-Fi works on my phone but not my laptop..." onkeydown="if(event.key==='Enter')send()"><button class="btn" onclick="send()">Send</button></div>
<div class="muted" style="margin-top:8px">For safety-critical issues, use a qualified professional or emergency service when appropriate.</div></section>
<section id="result" class="card" style="display:none"></section></main>
<script>
let history=[];
const chat=document.getElementById('chat'), input=document.getElementById('input'), result=document.getElementById('result');
function add(text,who){const d=document.createElement('div');d.className='bubble '+who;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
function esc(s){return String(s??'').replace(/[&<>"]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[x]))}
function render(a){
 if(!a)return;
 const causes=(a.possible_causes||[]).map(c=>`<div class="cause"><b>${esc(c.cause)}</b><span>${esc(c.reason)}</span><div class="pill">${esc(c.likelihood)}</div></div>`).join('');
 result.style.display='block';result.innerHTML=`<button class="clear" onclick="result.style.display='none'">hide</button><div class="result">
 <div><span class="badge">${esc(a.problem_type)}</span><h2 style="margin:9px 0 4px">${esc(a.summary)}</h2></div>
 <div class="section"><h3>LIKELY CAUSES</h3>${causes}</div><div class="grid">
 <div class="section"><h3>✓ SAFE CHECKS</h3><ul class="checks">${(a.safe_checks||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div>
 <div class="section"><h3>⚠ AVOID</h3><ul class="avoid">${(a.avoid||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div></div>
 <div class="section"><h3>URGENCY</h3><div class="urgency ${esc(a.urgency)}">${esc(a.urgency).toUpperCase()}</div></div>
 <div class="section"><h3>WHEN TO GET HELP</h3>${esc(a.professional_help)}</div></div>`;
 result.scrollIntoView({behavior:'smooth',block:'start'});
}
async function send(){
 const msg=input.value.trim();if(!msg)return;input.value='';add(msg,'user');history.push({role:'user',content:msg});
 const t=add('Thinking...','ai');t.classList.add('typing');
 try{const r=await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history.slice(-8)})});const d=await r.json();t.remove();
 if(d.questions?.length){const q=d.questions.map((x,i)=>`${i+1}. ${x}`).join('\\n');add(q,'ai');history.push({role:'assistant',content:q});}
 else if(d.analysis){const text=d.analysis.next_question||'I have enough information for a cautious first assessment. See the diagnosis below.';add(text,'ai');history.push({role:'assistant',content:text});render(d.analysis);}
 else add(d.error||'I could not process that. Please describe the problem again.','ai');
 }catch(e){t.remove();add('The assistant could not connect. You can retry; the server also has rule-based fallback mode.','ai')}
}
</script></body></html>"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/api/message", methods=["POST"])
def message():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    messages = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")]
    if not messages or messages[-1]["role"] != "user":
        return jsonify({"error": "Please send a problem description."}), 400
    try:
        state = {"messages": messages[-8:], "category": "general", "missing": [], "questions": [], "analysis": {}, "needs_questions": False}
        out = agent_graph.invoke(state)
        if out.get("needs_questions"):
            return jsonify({"questions": out.get("questions", [])[:3], "category": out.get("category", "general")})
        return jsonify({"analysis": out.get("analysis", {})})
    except Exception:
        return jsonify({"error": "Troubleshooting is temporarily unavailable. Try again with a shorter problem description."}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "gemini": bool(API_KEY), "model": MODEL})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
