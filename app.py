import os
import re
import json
from typing import TypedDict
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-development-key")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Lightweight local knowledge base used whenever Gemini is unavailable.
KB = {
    "wifi": {
        "keywords": ["wifi", "wi-fi", "internet", "router", "network", "disconnect"],
        "questions": [
            "Does the laptop connect to Wi-Fi but show 'No internet', or can it not connect at all?",
            "Can another laptop or computer use this same Wi-Fi network?",
            "Have you restarted the laptop and tried reconnecting?"
        ],
        "causes": [
            ("Laptop-specific network settings or adapter issue", "Your phone works, so the connection may be healthy while the laptop has a local issue.", "common"),
            ("Saved network or driver configuration problem", "A laptop-only failure can be related to saved Wi-Fi settings or adapter software.", "possible"),
            ("Router compatibility or access-control setting", "Some router settings can affect individual devices, but more evidence is needed.", "uncommon")
        ],
        "checks": ["Restart the laptop and reconnect to the same Wi-Fi network.", "Forget the network on the laptop and reconnect using the correct password.", "Check whether another computer can use the same Wi-Fi."],
        "avoid": ["Do not change advanced router settings or factory-reset the router unless you know its current configuration."],
        "urgency": "low", "help": "If multiple devices lose connectivity or the router repeatedly fails, contact your internet provider."
    },
    "laptop": {
        "keywords": ["laptop", "computer", "pc", "slow", "freezing", "lag", "overheating"],
        "questions": ["Does the slowdown affect the whole laptop or just one app?", "Does the laptop become hot or its fan unusually loud?", "Did this start after an update or installing an app?"],
        "causes": [("High CPU or memory use", "Background applications can slow down the system.", "common"), ("Heat-related performance reduction", "If the laptop becomes hot before slowing, thermal limits may be involved.", "possible"), ("Low storage or hardware issue", "Low free space or a hardware fault may contribute.", "possible")],
        "checks": ["Open Task Manager and observe CPU, memory and disk usage.", "Close unused applications and browser tabs.", "Check available storage space."],
        "avoid": ["Do not open the laptop or remove internal components unless qualified."],
        "urgency": "medium", "help": "Seek a technician if it becomes extremely hot, shuts down unexpectedly, smells burnt or remains slow."
    },
    "printer": {
        "keywords": ["printer", "printing", "print", "paper jam", "ink", "toner"],
        "questions": ["Does the printer power on normally?", "Is there an error message or blinking warning light?", "Does printing fail from every device or only one?"],
        "causes": [("Connection or print queue problem", "A stuck queue or lost connection can prevent jobs reaching the printer.", "common"), ("Paper, ink or toner issue", "Consumables or a paper-path issue can stop printing.", "possible"), ("Driver/software mismatch", "A device-specific issue can be caused by printer software.", "possible")],
        "checks": ["Read the printer display or indicator lights.", "Confirm paper is loaded and ink/toner is sufficient.", "Cancel stuck print jobs and retry a small test page."],
        "avoid": ["Do not force tightly jammed paper out; follow the manufacturer's procedure."],
        "urgency": "low", "help": "Contact the manufacturer or a technician if the printer reports an internal hardware fault."
    },
    "phone": {
        "keywords": ["phone", "mobile", "charging", "charger", "battery", "usb"],
        "questions": ["Does the phone show a charging symbol?", "Have you tried a known-good compatible cable and charger?", "Is the phone unusually hot or is the battery swollen?"],
        "causes": [("Cable or charger problem", "A known-good charging setup can help distinguish an accessory issue.", "common"), ("Blocked or damaged charging port", "Port damage or debris can prevent a reliable connection.", "possible"), ("Battery or charging-circuit issue", "Persistent failure with known-good accessories may indicate an internal issue.", "possible")],
        "checks": ["Try a known-good compatible cable and charger.", "Inspect the port visually under normal lighting without inserting objects.", "Check for charging symbols or warnings."],
        "avoid": ["Do not insert metal objects or liquids into the charging port."],
        "urgency": "medium", "help": "Stop charging and seek service if the phone is swollen, leaking, unusually hot or smells burnt."
    },
    "vehicle": {
        "keywords": ["bike", "car", "vehicle", "brake", "braking", "steering", "engine", "starting"],
        "questions": ["What symptom occurs and when?", "Are braking or steering affected, or are warning lights showing?", "Did it start suddenly or after maintenance/an impact?"],
        "causes": [("Wear or a consumable component", "Vehicle noises can come from worn components, but the source is not confirmed.", "common"), ("Loose or misaligned component", "A recent impact or maintenance can make this plausible.", "possible"), ("Safety-critical brake or steering fault", "Reduced braking or steering can signal a serious issue.", "possible")],
        "checks": ["Observe warning lights without taking risks.", "Note when the symptom occurs only if it is safe to do so.", "Arrange inspection by a qualified mechanic."],
        "avoid": ["Do not drive or ride if braking or steering is impaired. Do not attempt safety-critical repairs without training."],
        "urgency": "high", "help": "If braking or steering is impaired, stop using the vehicle when safe and contact a qualified mechanic or roadside service."
    },
    "fan": {
        "keywords": ["fan", "ceiling fan", "clicking", "buzzing", "rattling"],
        "questions": ["Does the sound change with fan speed?", "Is the fan wobbling or is there a burning smell?", "Does the sound happen continuously or only at startup?"],
        "causes": [("Loose external part", "A cover or other part may move during rotation.", "common"), ("Blade contact or alignment issue", "A speed-related sound can indicate contact, but this is not confirmed.", "possible"), ("Electrical or motor fault", "Burning smell, sparks or abnormal heat raise concern.", "possible")],
        "checks": ["Turn the fan off and visually observe for obvious looseness.", "Note whether the sound changes with speed.", "If there is smoke, sparking or a burning smell, switch it off if safe and keep away."],
        "avoid": ["Do not open the motor housing or work on wiring while connected to power."],
        "urgency": "medium", "help": "Contact a qualified electrician or appliance technician if the noise persists or electrical warning signs appear."
    },
    "washing_machine": {
        "keywords": ["washing machine", "washer", "drain", "draining", "spin cycle"],
        "questions": ["Does the cycle finish but leave water inside?", "Can you hear the drain pump running?", "Is an error code displayed?"],
        "causes": [("Blocked drain path or filter", "A blockage can prevent water from leaving the machine.", "common"), ("Kinked or blocked drain hose", "A restricted hose can reduce drainage.", "possible"), ("Drain pump fault", "A pump that does not run normally may be faulty.", "possible")],
        "checks": ["Switch the machine off and consult its manual before checking the filter.", "Visually inspect the external hose for kinks.", "Record any error code."],
        "avoid": ["Do not remove panels while powered or force open a locked door containing water."],
        "urgency": "medium", "help": "Contact appliance service if safe basic checks fail or there is leakage, electrical smell or unusual noise."
    },
    "study": {
        "keywords": ["study", "studying", "concentrate", "focus", "forgetting", "tasks", "procrastinate"],
        "questions": ["What interrupts you most: phone, tiredness, unclear material or something else?", "How long can you focus before it becomes difficult?", "Does this happen only while studying or across daily activities too?"],
        "causes": [("Distractions or unclear tasks", "Frequent interruptions or a vague task can make sustained attention harder.", "common"), ("Fatigue or inconsistent routine", "Sleep, breaks and workload can affect concentration.", "possible"), ("Broader attention or wellbeing concern", "Persistent difficulty may warrant professional guidance; this is not a diagnosis.", "possible")],
        "checks": ["Try one clearly defined 20-minute task with notifications disabled.", "Write down the next single action before starting.", "Track when concentration drops and what was happening."],
        "avoid": ["Do not infer a medical condition from concentration difficulties alone."],
        "urgency": "low", "help": "Consider qualified support if the problem persists or significantly affects daily life."
    },
    "general": {
        "keywords": [],
        "questions": ["What exactly happens, and what did you expect instead?", "When did it start, and is it constant or intermittent?", "Did anything change just before it began?"],
        "causes": [("Configuration or usage issue", "Many everyday problems follow a setting or usage change.", "common"), ("Worn or failing component", "Physical problems can result from wear, but more evidence is needed.", "possible"), ("Other underlying issue", "Additional causes remain possible until the symptom is clearer.", "uncommon")],
        "checks": ["Describe the symptom and when it occurs.", "Look for visible error messages or warning indicators without touching hazards.", "Compare with a known-good setting or device only when safe."],
        "avoid": ["Do not open powered equipment or attempt risky repairs involving electricity, gas, vehicles or chemicals."],
        "urgency": "low", "help": "Contact a qualified professional if the problem persists or a safety risk is present."
    }
}

class AgentState(TypedDict):
    messages: list
    category: str
    questions: list
    needs_questions: bool
    analysis: dict

def parse_json(text):
    if isinstance(text, list):
        text = "".join(str(x) for x in text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.M)
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                result = json.loads(match.group(0))
                return result if isinstance(result, dict) else {}
            except (ValueError, TypeError):
                pass
    return {}

def ask_gemini(prompt):
    if not API_KEY:
        return {}
    try:
        model = ChatGoogleGenerativeAI(
            model=MODEL, google_api_key=API_KEY, temperature=0.1,
            request_timeout=20, max_retries=1
        )
        response = model.invoke([HumanMessage(content=prompt)])
        return parse_json(response.content)
    except Exception:
        app.logger.exception("Gemini call failed; using local fallback")
        return {}

def classify_problem(state):
    conversation = " ".join(str(m.get("content", "")) for m in state["messages"][-8:])
    data = ask_gemini(
        "Classify the troubleshooting conversation into exactly one category: "
        "wifi,laptop,printer,phone,vehicle,fan,washing_machine,study,general. "
        "Return JSON only: {\"category\":\"wifi\"}. Conversation:\n" + conversation
    )
    category = data.get("category")
    if category in KB:
        return {"category": category}
    text = conversation.lower()
    scores = {name: sum(1 for word in item["keywords"] if word in text) for name, item in KB.items()}
    best = max(scores, key=scores.get)
    return {"category": best if scores[best] else "general"}

def determine_missing_information(state):
    messages = state["messages"]
    category = state["category"]
    # Ask targeted questions for a new problem; analyze after the user has replied.
    user_messages = [m for m in messages if m.get("role") == "user"]
    if len(user_messages) <= 1:
        data = ask_gemini(
            "You are a cautious troubleshooting assistant. Category: " + category +
            ". Conversation: " + json.dumps(messages[-8:]) +
            '. Return JSON only: {"questions":["question 1","question 2"],"needs_questions":true}. '
            "Ask at most 2 short, relevant questions. Do not ask for secrets."
        )
        questions = data.get("questions")
        if isinstance(questions, list) and questions:
            return {"questions": [str(q) for q in questions[:2]], "needs_questions": True}
        return {"questions": KB[category]["questions"][:2], "needs_questions": True}
    return {"questions": [], "needs_questions": False}

def analyze_problem(state):
    category = state["category"]
    messages = state["messages"]
    conversation = " ".join(str(m.get("content", "")) for m in messages).lower()
    dangerous = [
        "electric shock", "sparking", "smoke", "fire", "gas leak", "gas smell",
        "chemical spill", "toxic chemical", "ceiling collapse", "brake failure",
        "brakes failed", "steering failure", "can't steer", "cannot steer",
        "unconscious", "not breathing", "severe bleeding", "chest pain"
    ]
    if any(term in conversation for term in dangerous):
        return {"analysis": {
            "problem_type": category if category != "general" else "safety-critical",
            "summary": "Your description includes a potentially dangerous condition. Remote troubleshooting should stop at safe observation.",
            "possible_causes": [{"cause": "A safety-critical fault may be present", "reason": "The reported symptom could involve immediate risk and cannot be assessed safely remotely.", "likelihood": "possible"}],
            "safe_checks": ["Move away from the hazard if safe.", "Observe warning signs from a safe distance without touching the hazard."],
            "avoid": ["Do not attempt repairs or continue using unsafe equipment or a vehicle with impaired braking/steering.", "Contact emergency services if there is immediate danger."],
            "urgency": "high",
            "professional_help": "Contact the appropriate qualified professional or emergency service when there is immediate danger.",
            "next_question": ""
        }}
    data = ask_gemini(
        "You are a cautious everyday troubleshooting assistant. Do not claim certainty. "
        "Category: " + category + ". Conversation: " + json.dumps(messages[-10:]) +
        '. Return JSON only with this schema: {"summary":"...","possible_causes":[{"cause":"...","reason":"...","likelihood":"common|possible|uncommon"}],'
        '"safe_checks":["..."],"avoid":["..."],"urgency":"low|medium|high","professional_help":"...","next_question":""}. '
        "Use only provided facts. No medical diagnosis. Do not give dangerous repair instructions for electricity, gas, fire, structural damage, serious vehicle braking/steering or hazardous chemicals."
    )
    if isinstance(data.get("possible_causes"), list) and data["possible_causes"]:
        data["problem_type"] = category
        data["possible_causes"] = data["possible_causes"][:4]
        data["safe_checks"] = data.get("safe_checks") if isinstance(data.get("safe_checks"), list) else []
        data["avoid"] = data.get("avoid") if isinstance(data.get("avoid"), list) else []
        data["urgency"] = data.get("urgency") if data.get("urgency") in ("low", "medium", "high") else "medium"
        data["professional_help"] = str(data.get("professional_help", "Contact a qualified professional if the problem persists."))
        data["summary"] = str(data.get("summary", "Possible explanations based on the details provided."))
        data["next_question"] = str(data.get("next_question", ""))
        return {"analysis": data}
    kb = KB[category]
    return {"analysis": {
        "problem_type": category,
        "summary": "Based on the information provided, these are plausible explanations; more evidence may be needed.",
        "possible_causes": [{"cause": c, "reason": r, "likelihood": level} for c, r, level in kb["causes"]],
        "safe_checks": kb["checks"], "avoid": kb["avoid"], "urgency": kb["urgency"],
        "professional_help": kb["help"], "next_question": ""
    }}

def safety_finalize(state):
    analysis = state.get("analysis") or {}
    if analysis.get("urgency") not in ("low", "medium", "high"):
        analysis["urgency"] = "medium"
    if not analysis.get("avoid"):
        analysis["avoid"] = ["Do not attempt risky repairs without appropriate expertise."]
    return {"analysis": analysis}

builder = StateGraph(AgentState)
builder.add_node("classify", classify_problem)
builder.add_node("questions", determine_missing_information)
builder.add_node("diagnose", analyze_problem)
builder.add_node("safety_finalize", safety_finalize)
builder.add_edge(START, "classify")
builder.add_edge("classify", "questions")
builder.add_conditional_edges(
    "questions",
    lambda state: "ask" if state.get("needs_questions") else "diagnose",
    {"ask": END, "diagnose": "diagnose"}
)
builder.add_edge("diagnose", "safety_finalize")
builder.add_edge("safety_finalize", END)
agent_graph = builder.compile()

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Troubleshooter</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg:#080b14;--card:rgba(18,25,43,.88);--border:rgba(255,255,255,.1);--p:#6366f1;--text:#f8fafc;--muted:#94a3b8}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#20204b 0,#080b14 58%,#05070c 100%);color:var(--text);font:14px 'Plus Jakarta Sans',system-ui;padding:24px 14px;min-height:100vh}
.wrap{max-width:1000px;margin:auto;display:grid;gap:16px}.card{background:var(--card);backdrop-filter:blur(15px);border:1px solid var(--border);border-radius:20px;padding:22px;box-shadow:0 12px 35px #0005}
.head{display:flex;justify-content:space-between;align-items:center;gap:12px}.brand{display:flex;gap:14px;align-items:center}.logo{font-size:28px;background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:15px;border-radius:16px}h1{margin:0;font-size:1.5rem}.sub,.muted{color:var(--muted)}.sub{font-size:.9rem;margin-top:3px}
.badge{border:1px solid #6366f155;background:#6366f11c;color:#c7d2fe;border-radius:999px;padding:8px 12px;font-size:.8rem;white-space:nowrap}
.chat{height:440px;overflow:auto;display:flex;flex-direction:column;gap:12px;background:#080d18aa;border:1px solid var(--border);border-radius:16px;padding:16px}
.bubble{max-width:88%;padding:13px 16px;border-radius:16px;line-height:1.55;white-space:pre-wrap;overflow-wrap:anywhere}.ai{align-self:flex-start;background:#1e293bcc;border:1px solid var(--border)}.user{align-self:flex-end;background:linear-gradient(135deg,#6366f1,#4f46e5)}
.composer{display:flex;gap:9px;margin-top:12px}.composer input{flex:1;min-width:0;background:#0b1120;border:1px solid var(--border);color:white;border-radius:12px;padding:14px;outline:0;font:inherit}.composer input:focus{border-color:#818cf8}
.btn{border:0;border-radius:12px;padding:12px 20px;background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;font-weight:800;cursor:pointer;font:inherit}.btn:disabled{opacity:.55;cursor:wait}
.result{display:grid;gap:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.section{border:1px solid var(--border);background:#0c1321aa;border-radius:14px;padding:15px}.section h3{margin:0 0 10px;font-size:.92rem}.cause{border:1px solid var(--border);border-radius:12px;padding:12px;margin:8px 0;line-height:1.5}.cause b{display:block;margin-bottom:4px}.pill{display:inline-block;font-size:.7rem;padding:4px 8px;border-radius:999px;margin-top:7px;background:#6366f122;color:#c7d2fe}
.urgency{font-weight:800}.low{color:#34d399}.medium{color:#fbbf24}.high{color:#f87171}.checks,.avoid{margin:0;padding-left:20px;line-height:1.8}.clear{float:right;background:none;border:0;color:var(--muted);cursor:pointer}.typing{opacity:.65}
@media(max-width:650px){.grid{grid-template-columns:1fr}.head{align-items:flex-start;flex-direction:column}.chat{height:390px}.bubble{max-width:95%}.card{padding:15px}}
</style></head><body><main class="wrap">
<section class="card head"><div class="brand"><div class="logo">🛠️</div><div><h1>AI Troubleshooter</h1><div class="sub">Practical diagnosis for everyday problems</div></div></div><span class="badge">LangGraph • Gemini + fallback</span></section>
<section class="card"><div id="chat" class="chat"><div class="bubble ai">Tell me what is going wrong. I’ll ask a few targeted questions before suggesting safe checks.</div></div>
<div class="composer"><input id="input" placeholder="Example: My Wi-Fi works on my phone but not my laptop..." autocomplete="off"><button id="send" class="btn" onclick="send()">Send</button></div>
<div class="muted" style="margin-top:10px">For safety-critical issues, use a qualified professional or emergency service when appropriate.</div></section>
<section id="result" class="card" style="display:none"></section></main>
<script>
let history=[],busy=false;
const chat=document.getElementById('chat'),input=document.getElementById('input'),result=document.getElementById('result'),sendButton=document.getElementById('send');
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
function add(text,who){const d=document.createElement('div');d.className='bubble '+who;d.textContent=text;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function render(a){
 if(!a)return;
 const causes=(a.possible_causes||[]).map(c=>`<div class="cause"><b>${esc(c.cause)}</b><span>${esc(c.reason)}</span><div class="pill">${esc(c.likelihood)}</div></div>`).join('');
 result.style.display='block';
 result.innerHTML=`<button class="clear" onclick="result.style.display='none'">Hide</button><div class="result">
 <div><span class="badge">${esc(a.problem_type)}</span><h2 style="margin:12px 0 5px">${esc(a.summary)}</h2></div>
 <div class="section"><h3>LIKELY CAUSES</h3>${causes||'No causes were returned.'}</div><div class="grid">
 <div class="section"><h3>✓ SAFE CHECKS</h3><ul class="checks">${(a.safe_checks||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div>
 <div class="section"><h3>⚠ AVOID</h3><ul class="avoid">${(a.avoid||[]).map(x=>`<li>${esc(x)}</li>`).join('')}</ul></div></div>
 <div class="section"><h3>URGENCY</h3><div class="urgency ${esc(a.urgency)}">${esc(a.urgency).toUpperCase()}</div></div>
 <div class="section"><h3>WHEN TO GET HELP</h3>${esc(a.professional_help)}</div></div>`;
}
async function send(){
 const msg=input.value.trim();if(!msg||busy)return;
 input.value='';add(msg,'user');history.push({role:'user',content:msg});
 const typing=add('Thinking...','ai');typing.classList.add('typing');busy=true;sendButton.disabled=true;
 try{
  const response=await fetch('/api/message',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history.slice(-10)})});
  const raw=await response.text();let data;
  try{data=JSON.parse(raw)}catch{throw new Error(`Server returned a non-JSON response (HTTP ${response.status}). Check the Flask terminal.`)}
  typing.remove();
  if(!response.ok){add(data.error||`Server error: ${response.status}`,'ai');return}
  if(Array.isArray(data.questions)&&data.questions.length){
   const q=data.questions.map((x,i)=>`${i+1}. ${x}`).join('\n');add(q,'ai');history.push({role:'assistant',content:q});
  }else if(data.analysis){
   const text=data.analysis.next_question||'I have enough information for a cautious first assessment. See the diagnosis below.';
   add(text,'ai');history.push({role:'assistant',content:text});render(data.analysis);
  }else add(data.error||'No valid response was returned. Please try again.','ai');
 }catch(e){typing.remove();console.error('Agent request failed:',e);add(`Request failed: ${e.message}`,'ai')}
 finally{busy=false;sendButton.disabled=false;input.focus()}
}
</script></body></html>"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)

@app.route("/api/message", methods=["POST"])
def message():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return jsonify({"error": "Messages must be a list."}), 400
    messages = [
        {"role": m["role"], "content": str(m["content"])[:3000]}
        for m in messages
        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str) and m["content"].strip()
    ][-10:]
    if not messages or messages[-1]["role"] != "user":
        return jsonify({"error": "Please send a problem description."}), 400
    try:
        state = {"messages": messages, "category": "general", "questions": [],
                 "needs_questions": False, "analysis": {}}
        output = agent_graph.invoke(state)
        if output.get("needs_questions"):
            return jsonify({"questions": output.get("questions", [])[:2],
                            "category": output.get("category", "general")})
        return jsonify({"analysis": output.get("analysis", {})})
    except Exception:
        app.logger.exception("Agent request failed")
        # Keep the application useful even if a graph node unexpectedly fails.
        text = " ".join(m["content"] for m in messages).lower()
        category = next((name for name, item in KB.items()
                         if name != "general" and any(k in text for k in item["keywords"])), "general")
        kb = KB[category]
        if len([m for m in messages if m["role"] == "user"]) <= 1:
            return jsonify({"questions": kb["questions"][:2], "category": category})
        return jsonify({"analysis": {
            "problem_type": category,
            "summary": "Local fallback response: based on the information provided, these are plausible explanations.",
            "possible_causes": [{"cause": c, "reason": r, "likelihood": level} for c, r, level in kb["causes"]],
            "safe_checks": kb["checks"], "avoid": kb["avoid"], "urgency": kb["urgency"],
            "professional_help": kb["help"], "next_question": ""
        }})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "gemini_configured": bool(API_KEY), "model": MODEL})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
