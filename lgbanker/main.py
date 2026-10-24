"""
Card Transaction Dispute — LangGraph version
============================================
Same scenario: a cardholder disputes a charge (unrecognized, duplicate,
or merchant problem).

LangGraph style = EXPLICIT STATE MACHINE.
You define the state, every step (node), and every branch (edge).
The LLM is used only where judgment is needed (classifying the complaint,
writing the letter). Routing, thresholds and the human-approval pause are
plain Python, so every run follows an auditable path.

Graph:
    intake (LLM classify) -> fetch_data -> score_risk -> [route]
        fraud        -> block_card -> open_case
        duplicate    ------------------> open_case
        merchant     ------------------> open_case
    open_case -> [amount > $500?] -> human_review (interrupt) -> credit
                                  `-> credit
    credit -> write_letter -> END

Setup (local model via Ollama, managed with uv):
    ollama pull llama3.2
    uv add langgraph langchain-ollama pydantic
    uv run card_dispute_langgraph.py
"""
import os
import re
from typing import Annotated, Literal, Optional, TypedDict
import operator

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Local Llama 3.2 via Ollama. Override with OLLAMA_MODEL / OLLAMA_HOST if needed.
llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    temperature=0,          # deterministic extraction
)
AUTO_CREDIT_LIMIT = 500.00

# ---------------------------------------------------------------------------
# Mock back-office systems (same data as the Strands version)
# ---------------------------------------------------------------------------
TRANSACTIONS = {
    "TXN-1001": {"card": "4111-XXXX-XXXX-1234", "merchant": "TECHGADGETS ONLINE",
                 "amount": 742.18, "date": "2026-10-19", "channel": "card-not-present",
                 "country": "RO", "3ds": False},
    "TXN-1002": {"card": "4111-XXXX-XXXX-1234", "merchant": "BLUE BOTTLE COFFEE",
                 "amount": 6.50, "date": "2026-10-20", "channel": "chip",
                 "country": "US", "3ds": None},
    "TXN-1003": {"card": "4111-XXXX-XXXX-1234", "merchant": "BLUE BOTTLE COFFEE",
                 "amount": 6.50, "date": "2026-10-20", "channel": "chip",
                 "country": "US", "3ds": None},
}
CARDHOLDER = {"4111-XXXX-XXXX-1234": {"home_country": "US", "card_in_possession": True}}


# ---------------------------------------------------------------------------
# 1. State shared by every node
# ---------------------------------------------------------------------------
class DisputeState(TypedDict, total=False):
    complaint: str
    txn_id: str
    claim: str                  # what the customer says happened
    txn: dict
    duplicates: list
    fraud_score: int
    fraud_signals: list
    dispute_type: Literal["fraud", "duplicate", "merchant"]
    case_id: str
    credit_status: str
    analyst_decision: str
    letter: str
    audit_log: Annotated[list[str], operator.add]   # append-only trail


# Structured output for the one classification step that needs an LLM
class Intake(BaseModel):
    txn_id: str = Field(description="Transaction ID mentioned, e.g. TXN-1001")
    claim: Literal["unrecognized", "duplicate", "not_received", "not_as_described"]


# ---------------------------------------------------------------------------
# 2. Nodes
# ---------------------------------------------------------------------------
def _rule_based_intake(text: str) -> Intake:
    """Fallback if the small local model returns malformed JSON."""
    m = re.search(r"TXN-\d+", text, re.I)
    t = text.lower()
    if any(w in t for w in ("twice", "double", "duplicate")):
        claim = "duplicate"
    elif any(w in t for w in ("never received", "not received", "didn't arrive")):
        claim = "not_received"
    elif any(w in t for w in ("not as described", "broken", "wrong item")):
        claim = "not_as_described"
    else:
        claim = "unrecognized"
    return Intake(txn_id=m.group(0).upper() if m else "UNKNOWN", claim=claim)


def intake(state: DisputeState):
    source = "llm"
    try:
        # Ollama supports JSON-schema structured output; Llama 3.2 handles this
        # well with temperature=0, but a 3B model can still slip occasionally.
        r: Intake = llm.with_structured_output(Intake).invoke(
            "Extract the transaction ID (format TXN-1234) and the customer's claim "
            "from this card-dispute complaint. claim must be one of: unrecognized, "
            "duplicate, not_received, not_as_described.\n\n"
            f"Complaint: {state['complaint']}")
        if r is None or r.txn_id.upper() not in TRANSACTIONS:
            raise ValueError("bad extraction")
        r.txn_id = r.txn_id.upper()
    except Exception:
        r, source = _rule_based_intake(state["complaint"]), "rules-fallback"
    return {"txn_id": r.txn_id, "claim": r.claim,
            "audit_log": [f"intake ({source}): {r.txn_id} claim={r.claim}"]}


def fetch_data(state: DisputeState):
    t = TRANSACTIONS[state["txn_id"]]
    dups = [k for k, v in TRANSACTIONS.items()
            if v["card"] == t["card"] and v["merchant"] == t["merchant"]
            and v["date"] == t["date"] and v["amount"] == t["amount"]]
    return {"txn": t, "duplicates": dups,
            "audit_log": [f"fetch: ${t['amount']} at {t['merchant']}, matches={dups}"]}


def score_risk(state: DisputeState):
    t, holder = state["txn"], CARDHOLDER[state["txn"]["card"]]
    score, signals = 10, []
    if t["channel"] == "card-not-present": score += 25; signals.append("CNP")
    if t["country"] != holder["home_country"]: score += 30; signals.append("foreign")
    if t["3ds"] is False: score += 20; signals.append("no 3DS")
    return {"fraud_score": min(score, 100), "fraud_signals": signals,
            "audit_log": [f"risk: score={score} {signals}"]}


def classify(state: DisputeState):
    # Deterministic business rules — the same input always takes the same path.
    if state["claim"] == "duplicate" and len(state["duplicates"]) > 1:
        kind = "duplicate"
    elif state["claim"] == "unrecognized" and state["fraud_score"] >= 60:
        kind = "fraud"
    else:
        kind = "merchant"
    return {"dispute_type": kind, "audit_log": [f"classify: {kind}"]}


def block_card(state: DisputeState):
    return {"audit_log": [f"block_card: {state['txn']['card']} blocked, replacement ordered"]}


def open_case(state: DisputeState):
    case_id = f"CB-{abs(hash(state['txn_id'])) % 10000:04d}"
    return {"case_id": case_id,
            "audit_log": [f"open_case: {case_id} reason={state['dispute_type']}"]}


def human_review(state: DisputeState):
    # Pauses the graph. State is saved by the checkpointer; resume later
    # with Command(resume="approve" | "deny").
    decision = interrupt({
        "case_id": state["case_id"], "amount": state["txn"]["amount"],
        "type": state["dispute_type"], "fraud_score": state["fraud_score"],
        "question": "Approve provisional credit? (approve/deny)",
    })
    return {"analyst_decision": decision,
            "audit_log": [f"human_review: analyst said {decision}"]}


def credit(state: DisputeState):
    if state.get("analyst_decision") == "deny":
        status = "denied by analyst"
    else:
        status = f"provisional credit ${state['txn']['amount']:.2f} issued"
    return {"credit_status": status, "audit_log": [f"credit: {status}"]}


BANK_NAME = "Example Bank Card Services"


def _letter_facts(state: DisputeState) -> list[str]:
    """Code decides WHAT the letter says; the LLM only decides HOW to say it."""
    t, kind = state["txn"], state["dispute_type"]
    amt, merchant, case = f"${t['amount']:.2f}", t["merchant"], state["case_id"]
    last4 = t["card"][-4:]
    facts = [f"You contacted us about a {amt} charge from {merchant} on {t['date']} "
             f"on your card ending {last4}. Your case number is {case}."]
    if kind == "fraud":
        facts += ["We believe this charge was not made by you.",
                  f"We have blocked your card ending {last4} to protect you, "
                  "and a replacement card is on its way.",
                  "You will not be responsible for this unauthorized charge."]
    elif kind == "duplicate":
        facts += [f"We confirmed that {merchant} charged you twice for the same purchase.",
                  "We are reversing the extra charge. Your card stays active."]
    else:
        facts += [f"We have opened a dispute with {merchant}.",
                  f"{merchant} has 30 days to respond. We will contact you with the outcome.",
                  "Your card stays active."]
    if state.get("analyst_decision") == "deny":
        facts.append("After review, we could not issue a temporary credit at this time.")
    else:
        facts.append(f"We have added a temporary credit of {amt} to your account "
                     "while the case is finalized.")
    return facts


def _template_letter(facts: list[str]) -> str:
    return ("Dear Cardholder,\n\n" + "\n\n".join(facts) +
            f"\n\nIf you have questions, reply to this message and quote your case number."
            f"\n\nSincerely,\n{BANK_NAME}")


def write_letter(state: DisputeState):
    facts = _letter_facts(state)
    bullets = "\n".join(f"- {f}" for f in facts)
    msg = llm.invoke(
        "Rewrite the facts below as a short, warm letter from a bank to its customer.\n"
        "Rules:\n"
        "- Use ONLY these facts. Do not add, remove or change any fact, amount, date or case number.\n"
        "- Write to the customer (the cardholder), never to the merchant.\n"
        "- Start with 'Dear Cardholder,'. "
        f"End with 'Sincerely,' and then '{BANK_NAME}' on the next line.\n"
        "- No placeholders like [Your Name]. Under 130 words. Output only the letter.\n\n"
        f"Facts:\n{bullets}")
    letter = msg.content.strip()

    # Guardrail: if the model dropped key facts or invented a placeholder,
    # send the deterministic template instead of a wrong letter.
    must_have = [state["case_id"], f"{state['txn']['amount']:.2f}"]
    ok = all(m in letter for m in must_have) and "[" not in letter
    if state["dispute_type"] != "merchant" and "30 days" in letter:
        ok = False
    if not ok:
        return {"letter": _template_letter(facts),
                "audit_log": ["letter: LLM draft failed checks -> template used"]}
    return {"letter": letter, "audit_log": ["letter: drafted by LLM, passed checks"]}


# ---------------------------------------------------------------------------
# 3. Routers (plain Python)
# ---------------------------------------------------------------------------
def route_by_type(state: DisputeState) -> str:
    return "block_card" if state["dispute_type"] == "fraud" else "open_case"


def route_by_amount(state: DisputeState) -> str:
    return "human_review" if state["txn"]["amount"] > AUTO_CREDIT_LIMIT else "credit"


# ---------------------------------------------------------------------------
# 4. Wire the graph
# ---------------------------------------------------------------------------
g = StateGraph(DisputeState)
for name, fn in [("intake", intake), ("fetch_data", fetch_data),
                 ("score_risk", score_risk), ("classify", classify),
                 ("block_card", block_card), ("open_case", open_case),
                 ("human_review", human_review), ("credit", credit),
                 ("write_letter", write_letter)]:
    g.add_node(name, fn)

g.add_edge(START, "intake")
g.add_edge("intake", "fetch_data")
g.add_edge("fetch_data", "score_risk")
g.add_edge("score_risk", "classify")
g.add_conditional_edges("classify", route_by_type, ["block_card", "open_case"])
g.add_edge("block_card", "open_case")
g.add_conditional_edges("open_case", route_by_amount, ["human_review", "credit"])
g.add_edge("human_review", "credit")
g.add_edge("credit", "write_letter")
g.add_edge("write_letter", END)

# A checkpointer is required for interrupt/resume (use Postgres/Redis in prod)
app = g.compile(checkpointer=MemorySaver())


def run(complaint: str, thread_id: str, analyst_answer: str = "approve"):
    cfg = {"configurable": {"thread_id": thread_id}}
    print("\n" + "=" * 70 + f"\nCUSTOMER: {complaint}\n" + "=" * 70)
    result = app.invoke({"complaint": complaint, "audit_log": []}, cfg)

    if "__interrupt__" in result:                       # paused for a human
        print("\n⏸  PAUSED FOR ANALYST:", result["__interrupt__"][0].value)
        print(f"   analyst answers: {analyst_answer}")
        result = app.invoke(Command(resume=analyst_answer), cfg)

    print("\nAUDIT LOG:")
    for line in result["audit_log"]:
        print("  -", line)
    print("\nLETTER:\n" + result["letter"])


if __name__ == "__main__":
    run("TXN-1001: I don't recognize this $742.18 charge from TECHGADGETS ONLINE. "
        "I still have my card.", thread_id="case-1")
    run("TXN-1003: I was charged twice for one coffee at Blue Bottle on Oct 20.",
        thread_id="case-2")