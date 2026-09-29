"""
ISDO Lab C5 - SLA & Escalation Agent
Monitors SLA deadlines, predicts breach risk, escalates CRITICAL/BREACHED P1/P2
tickets, and pauses at a human-in-the-loop (HITL) gate before any P1 escalation.

Run from the project root (start mcp_server/snow_shim.py first for live updates):
    python agents/sla_agent.py

Changes from the lab sample (see comments marked FIX):
  - Model ID: claude-opus-5-5 (override with ANTHROPIC_MODEL in .env, same as C4)
  - No temperature parameter: current models return a 400 error for any
    non-default temperature, so the prompt's temperature=0.0 cannot be sent.
    The escalation RULES are enforced in code, so results are deterministic anyway.
  - HITL gate is driven by the ticket's priority, not a manual True/False flag,
    so a P1 can never be escalated without a human decision
  - Escalation policy is enforced in code as well as in the prompt
  - update_ticket PATCHes the C2 ServiceNow shim; falls back to simulation if it's down
  - HITL decisions are logged to logs/hitl_decisions.jsonl
  - Bounded agent loop that exits on every stop_reason (sample could loop forever)
  - Test due dates adjusted so all 4 risk levels really occur (see test_tickets)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

# ── SETUP ─────────────────────────────────────────────────────────────────────

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
load_dotenv(ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set - add it to the .env file in the project root.")

client = anthropic.Anthropic()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")   # FIX: was claude-opus-5
MAX_TOKENS = 3000        # FIX: was 500 - Opus 5.5 always thinks; thinking counts here
MAX_TURNS = 6            # FIX: hard cap on model round-trips per ticket
SNOW_URL = os.environ.get("SNOW_URL", "http://127.0.0.1:5001")   # C2 snow_shim
HITL_LOG = ROOT / "logs" / "hitl_decisions.jsonl"

# Simulated "now" for consistent, reproducible demo results
SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)

# SLA targets in minutes
SLA_MINUTES = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}

ESCALATION_TEAMS = {
    "Network": "L2-Network-Ops",
    "Application": "L2-App-Support",
    "Server": "L2-Server-Ops",
    "Access": "L2-Security-Ops",
}
DEFAULT_TEAM = "L2-Service-Desk"          # everything else ("General")
VALID_TEAMS = set(ESCALATION_TEAMS.values()) | {DEFAULT_TEAM}

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

tools = [
    {
        "name": "get_sla_status",
        "description": "Check the SLA status of a ticket. Returns minutes remaining, breach risk "
                       "level (BREACHED/CRITICAL/AT_RISK/ON_TRACK), and whether escalation is required.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {"type": "string",
                            "description": "SLA due datetime in format YYYY-MM-DD HH:MM:SS"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            },
            "required": ["ticket_number", "sla_due", "priority"],
        },
    },
    {
        "name": "update_ticket",
        "description": "Update a ticket in ServiceNow: escalate it to a team, add a work note, "
                       "or change its state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {"type": "string", "enum": ["escalate", "add_note", "update_state"],
                           "description": "Action to perform on the ticket"},
                "escalation_team": {"type": "string",
                                    "description": "Team to escalate to, e.g. L2-Network-Ops"},
                "note": {"type": "string", "description": "Work note to add to the ticket"},
                "new_state": {"type": "string",
                              "description": "New state, e.g. In Progress, Escalated, Resolved"},
            },
            "required": ["ticket_number", "action"],
        },
    },
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────

def get_sla_status(ticket_number, sla_due, priority):
    """Calculate minutes remaining and breach risk against the priority's SLA target."""
    if priority not in SLA_MINUTES:
        return {"error": f"Unknown priority: {priority}"}
    try:
        due_dt = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due format (need YYYY-MM-DD HH:MM:SS): {sla_due}"}

    minutes_remaining = int((due_dt - SIMULATED_NOW).total_seconds() // 60)
    target = SLA_MINUTES[priority]
    pct_left = round(100 * minutes_remaining / target, 1)

    if minutes_remaining < 0:
        risk, msg = "BREACHED", f"SLA BREACHED by {abs(minutes_remaining)} minutes"
    elif minutes_remaining < target * 0.2:
        risk, msg = "CRITICAL", f"Only {minutes_remaining} minutes remaining - breach imminent"
    elif minutes_remaining < target * 0.5:
        risk, msg = "AT_RISK", f"{minutes_remaining} minutes remaining - at risk"
    else:
        risk, msg = "ON_TRACK", f"{minutes_remaining} minutes remaining - on track"

    # Policy: only CRITICAL/BREACHED tickets at P1/P2 escalate; P3/P4 are monitored only
    requires_escalation = risk in ("BREACHED", "CRITICAL") and priority in ("P1", "P2")
    return {
        "ticket_number": ticket_number,
        "sla_due": sla_due,
        "priority": priority,
        "sla_target_minutes": target,
        "minutes_remaining": minutes_remaining,
        "percent_remaining": pct_left,
        "breach_risk": risk,
        "status_message": msg,
        "requires_escalation": requires_escalation,
        "requires_hitl": requires_escalation and priority == "P1",
    }


def patch_servicenow(ticket_number, fields):
    """PATCH the C2 ServiceNow shim. Returns (mode, detail)."""
    url = f"{SNOW_URL}/api/now/table/incident/{ticket_number}"
    try:
        r = requests.patch(url, json=fields, timeout=3)
    except requests.RequestException:
        return "simulated", "snow_shim not reachable - update simulated locally"
    if r.ok:
        return "servicenow_shim", f"HTTP {r.status_code}"
    return "servicenow_shim_error", f"HTTP {r.status_code}: {r.text[:120]}"


def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None):
    """Apply an update to ServiceNow (via the C2 shim, or simulated)."""
    stamp = datetime.now().isoformat(timespec="seconds")
    result = {"ticket_number": ticket_number, "action": action, "timestamp": stamp}

    if action == "escalate":
        mode, detail = patch_servicenow(
            ticket_number, {"assignment_group": escalation_team, "state": "Escalated"})
        result["message"] = f"Ticket {ticket_number} escalated to {escalation_team}"
        print(f"  [ServiceNow Mock] ESCALATED {ticket_number} -> {escalation_team}")
    elif action == "add_note":
        note = note or ""                            # FIX: sample crashed on a missing note
        mode, detail = "simulated", "the shim has no work-notes field - note recorded locally"
        result["message"] = f"Work note added to {ticket_number}: {note[:60]}"
        print(f"  [ServiceNow Mock] NOTE ADDED to {ticket_number}: {note[:60]}")
    elif action == "update_state":
        mode, detail = patch_servicenow(ticket_number, {"state": new_state})
        result["message"] = f"Ticket {ticket_number} state changed to: {new_state}"
        print(f"  [ServiceNow Mock] STATE CHANGED {ticket_number} -> {new_state}")
    else:
        return {"success": False, "error": f"Unknown action: {action}"}

    if action != "add_note" and mode != "servicenow_shim":
        print(f"  [ServiceNow Mock] ({detail})")
    result.update(success=not mode.endswith("error"), mode=mode, detail=detail)
    return result

# ── HITL GATE ─────────────────────────────────────────────────────────────────

def log_hitl(record):
    HITL_LOG.parent.mkdir(exist_ok=True)
    with open(HITL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def hitl_approve(ticket_number, action, detail):
    """Pause and ask a human to approve the action. Every decision is logged."""
    print("\n  " + "!!! " * 10)
    print("  HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}")
    print(f"  Action:  {action}")
    print(f"  Detail:  {detail}")
    print("  " + "!!! " * 10)
    try:
        answer = input("  Approve escalation? [y/n]: ").strip().lower()
    except EOFError:                                 # no terminal attached -> fail safe
        answer = ""
    approved = answer in ("y", "yes")
    print(f"  Decision: {'APPROVED' if approved else 'REJECTED'}")
    log_hitl({"timestamp": datetime.now().isoformat(timespec="seconds"),
              "ticket_number": ticket_number, "action": action, "detail": detail,
              "decision": "APPROVED" if approved else "REJECTED",
              "operator_input": answer})
    return approved

# ── SLA AGENT ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Call get_sla_status once to check breach risk.
2. If requires_escalation is true, call update_ticket with action "escalate" and the
   escalation team for the ticket's category (below). Do not escalate otherwise.
3. If an escalation is rejected by the human approver, do not retry it. Instead call
   update_ticket with action "add_note" recording that escalation was declined.
4. Finish with one sentence summarising the SLA status and the action taken.

Escalation teams by category:
- Network -> L2-Network-Ops
- Application -> L2-App-Support
- Server -> L2-Server-Ops
- Access -> L2-Security-Ops
- Anything else -> L2-Service-Desk"""


def run_tool(block, ticket, outcome):
    """Execute one tool call with guardrails. Returns the tool result dict."""
    name, inp = block.name, block.input

    if name == "get_sla_status":
        result = get_sla_status(inp.get("ticket_number"), inp.get("sla_due"), inp.get("priority"))
        if "error" not in result:
            outcome.update(breach_risk=result["breach_risk"],
                           minutes_remaining=result["minutes_remaining"],
                           requires_escalation=result["requires_escalation"])
            print(f"  -> Risk Level: {result['breach_risk']}")
            print(f"  -> Status:     {result['status_message']}")
        return result

    if name != "update_ticket":
        return {"error": f"Unknown tool: {name}"}

    # FIX: guardrails enforced in code - the model can only act on THIS ticket
    if inp.get("ticket_number") != ticket["number"]:
        return {"success": False, "error": f"Only {ticket['number']} may be updated in this run"}

    if inp.get("action") == "escalate":
        # Recompute from trusted ticket data rather than trusting the model's inputs
        status = get_sla_status(ticket["number"], ticket["sla_due"], ticket["priority"])
        if not status.get("requires_escalation"):
            return {"success": False, "error": "Escalation not permitted: policy requires "
                    "CRITICAL/BREACHED risk and priority P1 or P2"}
        team = inp.get("escalation_team")
        if team not in VALID_TEAMS:
            return {"success": False, "error": f"Unknown escalation team '{team}'. "
                    f"Use one of: {sorted(VALID_TEAMS)}"}
        # FIX: HITL gate keyed on the ticket's real priority, not a manual flag
        if ticket["priority"] == "P1":
            if outcome["hitl_decision"] == "REJECTED":
                return {"success": False, "error": "Escalation already rejected by human approver"}
            approved = hitl_approve(ticket["number"], "Escalate ticket", f"Escalate to {team}")
            outcome["hitl_decision"] = "APPROVED" if approved else "REJECTED"
            if not approved:
                print("  Escalation cancelled and logged.")
                return {"success": False,
                        "message": "Escalation rejected by human approver - do not retry"}
        result = update_ticket(ticket["number"], "escalate", escalation_team=team)
        outcome.update(escalated=result.get("success", False), escalation_team=team)
        return result

    return update_ticket(ticket["number"], inp.get("action"), inp.get("escalation_team"),
                         inp.get("note"), inp.get("new_state"))


def monitor_ticket(ticket):
    """Run SLA monitoring for one ticket. Returns an outcome dict (used by Lab C6)."""
    print(f"\n{'=' * 55}")
    print(f"SLA Check: {ticket['number']} | {ticket['priority']} | Category: {ticket['category']}")
    print(f"{'=' * 55}")

    outcome = {"ticket_number": ticket["number"], "priority": ticket["priority"],
               "breach_risk": None, "minutes_remaining": None, "requires_escalation": False,
               "escalated": False, "escalation_team": None, "hitl_decision": None}
    messages = [{
        "role": "user",
        "content": (f"Monitor SLA for this ticket and escalate if needed:\n\n"
                    f"Ticket: {ticket['number']}\nDescription: {ticket.get('short_description', '')}\n"
                    f"Category: {ticket['category']}\nPriority: {ticket['priority']}\n"
                    f"SLA Due: {ticket['sla_due']}"),
    }]

    for _ in range(MAX_TURNS):
        try:
            # FIX: no temperature (400 on current models); no forced tool_choice
            response = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, output_config={"effort": "medium"},
                system=SYSTEM_PROMPT, tools=tools, messages=messages)
        except anthropic.APIError as e:
            print(f"  !! API error: {e}")
            break

        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  Agent: {block.text.strip()}")
            if response.stop_reason not in ("end_turn", "stop_sequence"):
                print(f"  !! Stopped early: stop_reason={response.stop_reason}")
            break

        # Keep thinking blocks - the API requires them within a tool-use loop
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block, ticket, outcome)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(result)})
        messages.append({"role": "user", "content": results})
    else:
        print(f"  !! Gave up after {MAX_TURNS} turns")

    if outcome["requires_escalation"] and not outcome["escalated"] and not outcome["hitl_decision"]:
        print("  !! Escalation was required but the agent did not escalate")
    return outcome

# ── RUN SLA MONITORING ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Model: {MODEL} | simulated now: {SIMULATED_NOW:%Y-%m-%d %H:%M}")

    # Real tickets from incidents.csv, one per risk level.
    # FIX: the lab's due dates don't produce the risk levels it expects under its own
    # rules (e.g. INC0001002 with 30 of 60 min left is exactly 50% -> not CRITICAL).
    # Due dates marked "adjusted" were changed so each state really occurs.
    test_tickets = [
        # P1, 10 min left (17%) -> CRITICAL -> HITL prompt: type y   (CSV: 11:00, adjusted)
        {"number": "INC0001002", "short_description": "Cannot access ERP - SAP login failure",
         "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
        # P1, 60 min overdue -> BREACHED -> HITL prompt: type n     (CSV value)
        {"number": "INC0001010", "short_description": "Exchange server high CPU alert",
         "category": "Server", "priority": "P1", "sla_due": "2024-01-15 09:30:00"},
        # P2, 90 of 240 min left (37.5%) -> AT_RISK -> monitored    (CSV: 14:00, adjusted)
        # Step 5: change to '2024-01-15 10:00:00' -> BREACHED -> auto-escalates, no HITL
        {"number": "INC0001001", "short_description": "VPN not connecting after password change",
         "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00"},
        # P3, due in 2 days -> ON_TRACK -> monitored only           (CSV value)
        {"number": "INC0001003", "short_description": "Laptop running very slowly",
         "category": "Hardware", "priority": "P3", "sla_due": "2024-01-17 09:00:00"},
    ]

    outcomes = [monitor_ticket(t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSLA SUMMARY\n{'=' * 55}")
    for o in outcomes:
        action = (f"escalated -> {o['escalation_team']}" if o["escalated"]
                  else "escalation rejected (HITL)" if o["hitl_decision"] == "REJECTED"
                  else "monitored")
        print(f"  {o['ticket_number']:<11} {o['priority']}  {str(o['breach_risk']):<9} "
              f"{str(o['minutes_remaining']):>6} min  {action}")
    print(f"\nHITL decisions logged to: {HITL_LOG}")
