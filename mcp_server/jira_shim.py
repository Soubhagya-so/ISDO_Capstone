"""Mock Jira REST API (port 5002) - serves data/requests.csv.

GET /rest/agile/1.0/board/requests   all service requests
GET /rest/api/2/issue/<key>          one request, Jira-style nested 'fields'
GET /health                          service status
"""
import csv
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

# Find the project root (the folder that contains data/) wherever this file lives
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
CSV_PATH = ROOT / "data" / "requests.csv"
OVERFLOW_COL = "summary"   # free-text column that may contain stray commas

app = Flask(__name__)


def load_csv(path, overflow_col):
    """Read a CSV into a list of dicts, re-joining unquoted commas in free text."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx, rows = header.index(overflow_col), []
        for fields in reader:
            if not fields:
                continue
            extra = len(fields) - len(header)
            if extra > 0:
                fields = (fields[:idx] + [",".join(fields[idx:idx + extra + 1])]
                          + fields[idx + extra + 1:])
            rows.append({k: v.strip() for k, v in zip(header, fields)})
    return rows


REQUESTS = {row["key"]: row for row in load_csv(CSV_PATH, OVERFLOW_COL)}


def to_jira_issue(row, issue_id):
    """Shape a flat CSV row like a Jira REST v2 issue."""
    return {
        "id": str(10000 + issue_id),
        "key": row["key"],
        "self": f"{request.host_url}rest/api/2/issue/{row['key']}",
        "fields": {
            "summary": row["summary"],
            "issuetype": {"name": row["request_type"]},
            "priority": {"name": row["priority"]},
            "assignee": {"name": row["assignee"]},
            "status": {"name": row["status"]},
            "duedate": row["sla"],
        },
    }


@app.get("/rest/agile/1.0/board/requests")
def list_requests():
    rows = list(REQUESTS.values())
    return jsonify({"startAt": 0, "maxResults": len(rows), "total": len(rows),
                    "isLast": True, "values": rows})


@app.get("/rest/api/2/issue/<key>")
def get_issue(key):
    key = key.upper()
    if key not in REQUESTS:
        return jsonify({"errorMessages": ["Issue does not exist or you do not "
                                          "have permission to see it."],
                        "errors": {}}), 404
    issue_id = list(REQUESTS).index(key) + 1
    return jsonify(to_jira_issue(REQUESTS[key], issue_id))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "jira-mock",
                    "requests": len(REQUESTS),
                    "time": datetime.now().isoformat(timespec="seconds")})


if __name__ == "__main__":
    print(f"Loaded {len(REQUESTS)} requests from {CSV_PATH}")
    app.run(host="127.0.0.1", port=5002, debug=False)
