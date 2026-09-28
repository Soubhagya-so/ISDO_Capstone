"""
ISDO Lab C3 - Triage Agent
Reads a ticket and assigns: category, priority, assignment group, and PII flag.
Uses the Anthropic SDK with tool calling (ReAct loop: Reason -> Act -> Observe).

Run from the project root:
    python agents/triage_agent.py

Changes from the lab sample (see comments marked FIX):
  - Model ID updated to claude-opus-5-5 (override with ISDO_MODEL in .env)
  - No temperature parameter: current models reject non-default temperature
    with a 400 error, so temperature=0.0 from the lab prompt cannot be sent
  - Loop exits on every stop_reason (sample looped forever on max_tokens)
  - max_tokens raised: Opus 5.5 always thinks, and thinking counts toward it
  - get_open_tickets handles the malformed INC0001015 row and only reads data/
"""

import csv
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ── SETUP ─────────────────────────────────────────────────────────────────────

# Project root = the folder containing data/ (works from any subfolder)
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
DATA_DIR = ROOT / "data"
load_dotenv(ROOT / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set - add it to the .env file in the project root.")

client = anthropic.Anthropic()                     # reads ANTHROPIC_API_KEY
MODEL = os.environ.get("ISDO_MODEL", "claude-opus-5-5")   # FIX: was claude-opus-5
MAX_TOKENS = 2000        # FIX: was 500 - thinking + tool call must both fit
MAX_TURNS = 5            # FIX: safety cap so the loop can never run forever

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

tools = [
    {
        "name": "classify_ticket",
        "description": "Classify an IT support ticket. Returns category, priority, "
                       "assignment_group, and whether PII was detected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Network", "Application", "Hardware", "Access",
                             "Email", "Server", "Software"],
                    "description": "The ticket category",
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"],
                    "description": "P1=Critical/many users affected, P2=High/some users, "
                                   "P3=Medium/single user, P4=Low/request",
                },
                "assignment_group": {
                    "type": "string",
                    "description": "Team to assign the ticket to e.g. Network-Ops, App-Support, "
                                   "Desktop-Support, Service-Desk, Security-Ops, Server-Ops, "
                                   "Email-Support",
                },
                "pii_detected": {
                    "type": "boolean",
                    "description": "True if the ticket contains names, email addresses, "
                                   "employee IDs, or IP addresses",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the classification decision",
                },
            },
            "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"],
        },
    },
    {
        "name": "get_open_tickets",
        "description": "Get a summary count of currently open tickets by category "
                       "from the incidents CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Path to incidents.csv file, e.g. data/incidents.csv",
                }
            },
            "required": ["csv_path"],
        },
    },
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────

def read_incidents(path):
    """Read incidents.csv into dicts.

    FIX: INC0001015 has unquoted commas in its description, which shifts its
    category/priority/state into the wrong columns with a plain DictReader.
    Extra fields are joined back into the description column.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx, rows = header.index("description"), []
        for fields in reader:
            if not fields:
                continue
            extra = len(fields) - len(header)
            if extra > 0:
                fields = (fields[:idx] + [",".join(fields[idx:idx + extra + 1])]
                          + fields[idx + extra + 1:])
            rows.append({k: v.strip() for k, v in zip(header, fields)})
    return rows


def get_open_tickets(csv_path="data/incidents.csv"):
    """Return {category: open_count} from the incidents CSV."""
    # FIX: resolve relative to the project root and refuse paths outside data/,
    # since this path comes from the model
    path = (ROOT / csv_path).resolve()
    if DATA_DIR.resolve() not in path.parents:
        return {"error": f"Access denied - only files under data/ can be read: {csv_path}"}
    if not path.is_file():
        return {"error": f"File not found: {csv_path}"}
    counts = {}
    for row in read_incidents(path):
        if row.get("state") == "Open":
            cat = row.get("category") or "Unknown"
            counts[cat] = counts.get(cat, 0) + 1
    return counts


def handle_tool_call(tool_name, tool_input):
    """Route tool calls to their implementations."""
    if tool_name == "get_open_tickets":
        return get_open_tickets(tool_input.get("csv_path", "data/incidents.csv"))
    if tool_name == "classify_ticket":
        return tool_input          # the classification IS the tool's input
    return {"error": f"Unknown tool: {tool_name}"}

# ── TRIAGE AGENT ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the ISDO Triage Agent for Zensar's IT Service Desk.

Your job is to classify incoming IT support tickets. For each ticket:
1. Call the classify_ticket tool exactly once to assign category, priority, and assignment group
2. Flag if any PII (names, emails, employee IDs, IP addresses) is present

Priority rules:
- P1: Service down, many users affected, or security breach
- P2: Significant impact, single department or function affected
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment)

Be consistent and rule-based: the same ticket text must always get the same classification.
After classify_ticket returns, reply with one short confirmation sentence and stop."""


def print_classification(result):
    print(f"  -> Category:    {result.get('category')}")
    print(f"  -> Priority:    {result.get('priority')}")
    print(f"  -> Assign To:   {result.get('assignment_group')}")
    print(f"  -> PII Found:   {result.get('pii_detected')}")
    print(f"  -> Reason:      {result.get('reasoning')}")


def triage_ticket(ticket_number, short_description, description):
    """Run the triage agent on one ticket. Returns the classification dict (or None).

    The return value is what Lab C4 (Resolution Agent) and Lab C6 (Orchestrator)
    will consume.
    """
    print(f"\n{'=' * 55}")
    print(f"Triaging: {ticket_number}")
    print(f"{'=' * 55}")
    print(f"Description: {short_description}")

    messages = [{
        "role": "user",
        "content": f"Please triage this ticket:\n\nTicket: {ticket_number}\n"
                   f"Summary: {short_description}\nDetails: {description}",
    }]
    classification = None

    # Agentic loop (ReAct). FIX: bounded, and exits on every stop_reason.
    for _ in range(MAX_TURNS):
        try:
            # FIX: no temperature (400 error on current models) and no forced
            # tool_choice (Opus 5.5 rejects it) - the system prompt steers tool use
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                output_config={"effort": "low"},   # keep thinking short for triage
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
        except anthropic.APIError as e:
            print(f"  !! API error: {e}")
            return None

        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  Agent: {block.text.strip()}")
            if response.stop_reason not in ("end_turn", "stop_sequence"):
                print(f"  !! Stopped early: stop_reason={response.stop_reason}")
            break

        # Keep the full assistant turn (including thinking blocks) - the API
        # requires thinking blocks to be passed back unchanged within a tool loop
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  -> Tool called: {block.name}")
            result = handle_tool_call(block.name, block.input)
            if block.name == "classify_ticket":
                classification = result
                print_classification(result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  !! Gave up after {MAX_TURNS} turns")

    if classification is None:
        print("  !! No classification produced for this ticket")
    return classification

# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Model: {MODEL}")

    # Steps 4-5: 5 tickets from the lab + the 6th ticket from Step 5
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. "
         "Error: authentication failed."),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
         "Started 09:00 today."),
        ("INC0001008", "Network switch down - Building C",
         "Network switch in Building C server room unresponsive. 40 users in Building C affected."),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset."),
        ("REQ-1002", "VPN access for new contractor joining project Phoenix",
         "New contractor [REDACTED NAME] emp-id ZEN-9823 joining next Monday. "
         "Email: contractor@client.com"),
        # Step 5 - your own ticket. To test the priority change, swap the
        # description for: "Entire Sales team cannot access Salesforce CRM since this morning."
        ("INC-TEST-006", "Salesforce CRM not accessible",
         "User cannot access Salesforce CRM from company laptop since this morning."),
    ]

    results = {}
    for number, short_desc, desc in test_tickets:
        results[number] = triage_ticket(number, short_desc, desc)

    print("\n" + "=" * 55)
    print("TRIAGE SUMMARY")
    print("=" * 55)
    for number, r in results.items():
        if r:
            print(f"  {number:<13} {r['category']:<12} {r['priority']:<4} "
                  f"{r['assignment_group']:<16} PII={r['pii_detected']}")
        else:
            print(f"  {number:<13} (no classification)")

    print("\n" + "=" * 55)
    print("OPEN TICKET COUNTS BY CATEGORY")
    print("=" * 55)
    counts = get_open_tickets("data/incidents.csv")   # direct demo of the tool
    for cat, count in sorted(counts.items()):
        print(f"  {cat:<20} {count} open")
