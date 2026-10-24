"""
Card Transaction Dispute — Strands Agents version (local Llama 3.2 via Ollama)
==============================================================================
Same scenario and mock data as card_dispute_langgraph.py.

Strands style = MODEL-DRIVEN.
There is no graph. One agent gets a goal, a short policy and two tools,
and the model decides what to call and with which dispute type.

Small-model adjustments (Llama 3.2 is a 3B model):
  * Few, coarse tools (2 instead of 8): small models rarely chain more than
    one or two tool calls, so the letter runs inside resolve_dispute.
  * Each tool result says exactly what to call next.
  * If the agent stops without opening a case, the code nudges it (max 2x).
  * Hard rules live INSIDE the tools, not in the prompt: the evidence must
    support the dispute type, the card is blocked only for fraud, and
    credits over $500 need analyst approval.
  * The letter uses the same "code decides the facts, LLM only words them,
    then check the draft" pattern as the LangGraph version.

Run with uv:
    ollama pull llama3.2
    uv add 'strands-agents[ollama]'
    uv run card_dispute_strands.py
"""
import os

from strands import Agent, tool
from strands.models.ollama import OllamaModel

MODEL = OllamaModel(
    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    model_id=os.getenv("OLLAMA_MODEL", "llama3.2"),
    temperature=0,
)
AUTO_CREDIT_LIMIT = 500.00
ANALYST_ANSWER = os.getenv("ANALYST_ANSWER", "approve")   # simulated human
BANK_NAME = "Example Bank Card Services"

# ---------------------------------------------------------------------------
# Mock back-office systems (identical to the LangGraph version)
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

CASES: dict[str, dict] = {}
AUDIT: list[str] = []


def _log(line: str) -> None:
    AUDIT.append(line)


def _evidence(txn_id: str) -> dict:
    t = TRANSACTIONS[txn_id]
    holder = CARDHOLDER[t["card"]]
    dups = [k for k, v in TRANSACTIONS.items()
            if v["card"] == t["card"] and v["merchant"] == t["merchant"]
            and v["date"] == t["date"] and v["amount"] == t["amount"]]
    score, signals = 10, []
    if t["channel"] == "card-not-present": score += 25; signals.append("CNP")
    if t["country"] != holder["home_country"]: score += 30; signals.append("foreign")
    if t["3ds"] is False: score += 20; signals.append("no 3DS")
    return {"txn": t, "duplicates": dups, "fraud_score": min(score, 100),
            "fraud_signals": signals}


# ---------------------------------------------------------------------------
# Tool 1: investigate (read-only)
# ---------------------------------------------------------------------------
@tool
def investigate_transaction(txn_id: str) -> dict:
    """Look up a disputed card transaction: details, duplicate charges and fraud score.
    Always call this first.

    Args:
        txn_id: Transaction ID like "TXN-1001".
    """
    txn_id = txn_id.strip().upper()
    if txn_id not in TRANSACTIONS:
        return {"error": f"{txn_id} not found"}
    ev = _evidence(txn_id)
    _log(f"investigate: {txn_id} ${ev['txn']['amount']} at {ev['txn']['merchant']}, "
         f"duplicates={ev['duplicates']}, fraud_score={ev['fraud_score']} {ev['fraud_signals']}")
    return {
        "merchant": ev["txn"]["merchant"], "amount": ev["txn"]["amount"],
        "date": ev["txn"]["date"],
        "duplicate_charges_found": len(ev["duplicates"]) > 1,
        "fraud_score": ev["fraud_score"], "fraud_signals": ev["fraud_signals"],
        "guidance": "duplicate if duplicate_charges_found; fraud if the customer does "
                    "not recognize it and fraud_score >= 60; otherwise merchant.",
        "next_step": f"You are NOT done. Now call resolve_dispute with txn_id='{txn_id}' "
                     "and dispute_type set to 'duplicate', 'fraud' or 'merchant'.",
    }


# ---------------------------------------------------------------------------
# Tool 2: resolve (all money/card actions; hard rules enforced here)
# ---------------------------------------------------------------------------
@tool
def resolve_dispute(txn_id: str, dispute_type: str) -> dict:
    """Resolve a dispute: opens the case, blocks the card if fraud, issues a
    temporary credit and sends the customer letter. Call once, after
    investigate_transaction. This is the step that actually creates the case ID.

    Args:
        txn_id: Transaction ID like "TXN-1001".
        dispute_type: One of "fraud", "duplicate", "merchant".
    """
    txn_id, kind = txn_id.strip().upper(), dispute_type.strip().lower()
    if txn_id not in TRANSACTIONS:
        return {"error": f"{txn_id} not found"}
    if kind not in ("fraud", "duplicate", "merchant"):
        return {"error": "dispute_type must be fraud, duplicate or merchant"}
    if any(c["txn_id"] == txn_id for c in CASES.values()):
        return {"error": f"{txn_id} already has a case; do not resolve twice"}

    ev = _evidence(txn_id)
    # Guardrail: the evidence must support the type the model chose.
    if kind == "duplicate" and len(ev["duplicates"]) < 2:
        return {"error": "REJECTED: no duplicate charge found. Choose another type."}
    if kind == "fraud" and ev["fraud_score"] < 60:
        return {"error": f"REJECTED: fraud score {ev['fraud_score']} is below 60. "
                         "Treat as a merchant dispute."}
    _log(f"classify (by model): {kind}")

    t = ev["txn"]
    if kind == "fraud":
        _log(f"block_card: {t['card']} blocked, replacement ordered")

    case_id = f"CB-{abs(hash(txn_id)) % 10000:04d}"
    CASES[case_id] = {"txn_id": txn_id, "type": kind, "decision": None}
    _log(f"open_case: {case_id} reason={kind}")

    decision = "approve"
    if t["amount"] > AUTO_CREDIT_LIMIT:
        print(f"\n⏸  PAUSED FOR ANALYST: case {case_id}, ${t['amount']:.2f}, {kind}, "
              f"fraud_score={ev['fraud_score']}. Approve provisional credit?")
        decision = ANALYST_ANSWER          # swap for input() or a review queue
        print(f"   analyst answers: {decision}")
        _log(f"human_review: analyst said {decision}")
    CASES[case_id]["decision"] = decision

    status = ("denied by analyst" if decision == "deny"
              else f"provisional credit ${t['amount']:.2f} issued")
    _log(f"credit: {status}")
    _write_letter(case_id)
    return {"case_id": case_id, "card_blocked": kind == "fraud", "credit": status,
            "letter": "sent",
            "next_step": f"DONE. Reply with one line: '{kind}, case {case_id}'."}


# ---------------------------------------------------------------------------
# Letter (runs inside resolve_dispute): code picks facts -> sub-agent words
# them -> checks. Not a tool, so the model can't forget to call it.
# ---------------------------------------------------------------------------
letter_writer = Agent(model=MODEL, callback_handler=None, system_prompt=(
    "You rewrite bullet-point facts as a short, warm letter from a bank to its "
    "customer. Use ONLY the facts given; never add, remove or change any fact, "
    "amount, date or case number. Write to the cardholder, never to the merchant. "
    f"Start with 'Dear Cardholder,'. End with 'Sincerely,' then '{BANK_NAME}'. "
    "No placeholders like [Your Name]. Under 130 words. Output only the letter."))


def _letter_facts(case_id: str) -> list[str]:
    case = CASES[case_id]
    t, kind = TRANSACTIONS[case["txn_id"]], case["type"]
    amt, merchant, last4 = f"${t['amount']:.2f}", t["merchant"], t["card"][-4:]
    facts = [f"You contacted us about a {amt} charge from {merchant} on {t['date']} "
             f"on your card ending {last4}. Your case number is {case_id}."]
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
    if case["decision"] == "deny":
        facts.append("After review, we could not issue a temporary credit at this time.")
    else:
        facts.append(f"We have added a temporary credit of {amt} to your account "
                     "while the case is finalized.")
    return facts


def _write_letter(case_id: str) -> None:
    case, facts = CASES[case_id], _letter_facts(case_id)

    letter_writer.messages.clear()
    draft = str(letter_writer("Facts:\n" + "\n".join(f"- {f}" for f in facts))).strip()

    amount = f"{TRANSACTIONS[case['txn_id']]['amount']:.2f}"
    problems = []
    if case_id not in draft: problems.append("missing case number")
    if amount not in draft: problems.append("missing amount")
    if "[" in draft: problems.append("placeholder")
    if case["type"] != "merchant" and "30 days" in draft:
        problems.append("30-day line in non-merchant letter")

    if problems:
        letter = ("Dear Cardholder,\n\n" + "\n\n".join(facts) +
                  "\n\nIf you have questions, reply to this message and quote your "
                  f"case number.\n\nSincerely,\n{BANK_NAME}")
        _log(f"letter: LLM draft failed checks ({', '.join(problems)}) -> template used")
    else:
        letter = draft
        _log("letter: drafted by LLM, passed checks")
    case["letter"] = letter


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------
dispute_agent = Agent(
    model=MODEL,
    callback_handler=None,          # quiet; we print the audit log instead
    system_prompt=(
        "You are a bank card-dispute analyst. You MUST use tools; never invent "
        "a case ID. For each customer complaint:\n"
        "1. Call investigate_transaction with the transaction ID.\n"
        "2. Choose the dispute type from the result: 'duplicate' if duplicate charges "
        "were found; 'fraud' if the customer does not recognize the charge and "
        "fraud_score is 60 or more; otherwise 'merchant'.\n"
        "3. Call resolve_dispute with the transaction ID and that type. If it returns "
        "REJECTED, follow its advice and call it again.\n"
        "Only after resolve_dispute returns a case_id, reply with one line: "
        "the dispute type and case ID."),
    tools=[investigate_transaction, resolve_dispute],
)

MAX_NUDGES = 2   # how many times we remind the model if it stops early


def run(complaint: str):
    print("\n" + "=" * 70 + f"\nCUSTOMER: {complaint}\n" + "=" * 70)
    AUDIT.clear()
    _log(f"complaint received: {complaint[:60]}...")
    dispute_agent.messages.clear()            # fresh conversation per complaint
    summary = str(dispute_agent(complaint)).strip()

    # Small models often stop after one tool call and invent an answer.
    # Check the real outcome (a case in CASES), not the model's claim, and nudge.
    txn_id = next((t for t in TRANSACTIONS if t in complaint.upper()), None)
    for attempt in range(1, MAX_NUDGES + 1):
        if txn_id is None or any(c["txn_id"] == txn_id for c in CASES.values()):
            break
        _log(f"nudge {attempt}: agent stopped without resolving; reminded it")
        summary = str(dispute_agent(
            f"You have NOT finished: no case exists yet and '{summary[:40]}' is not a "
            f"real case. Call the resolve_dispute tool now with txn_id='{txn_id}' "
            "and the correct dispute_type.")).strip()

    print("\nAUDIT LOG:")
    for line in AUDIT:
        print("  -", line)
    print("\nAGENT SUMMARY:", summary)
    letters = [c["letter"] for c in CASES.values()
               if c.get("letter") and c["txn_id"] in complaint.upper()]
    print("\nLETTER:\n" + (letters[-1] if letters
                           else "(none — the agent did not finish the steps)"))


if __name__ == "__main__":
    run("TXN-1001: I don't recognize this $742.18 charge from TECHGADGETS ONLINE. "
        "I still have my card.")
    run("TXN-1003: I was charged twice for one coffee at Blue Bottle on Oct 20.")