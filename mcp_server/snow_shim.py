"""Mock ServiceNow Table API (port 5001) - serves data/incidents.csv.

GET   /api/now/table/incident            all incidents (?category= &priority=)
GET   /api/now/table/incident/<number>   one incident
PATCH /api/now/table/incident/<number>   update fields in memory (JSON body)
GET   /health                            service status
"""
import csv
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

# Find the project root (the folder that contains data/) wherever this file lives
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "data").is_dir())
CSV_PATH = ROOT / "data" / "incidents.csv"
OVERFLOW_COL = "description"   # free-text column that may contain stray commas

app = Flask(__name__)


def load_csv(path, overflow_col):
    """Read a CSV into a list of dicts.

    Rows with unquoted commas in free text have too many fields; the extra
    pieces are joined back into `overflow_col` so later columns stay aligned.
    """
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


INCIDENTS = {row["number"]: row for row in load_csv(CSV_PATH, OVERFLOW_COL)}


def not_found(number):
    body = {"error": {"message": "No Record found",
                      "detail": f"Record {number} does not exist"},
            "status": "failure"}
    return jsonify(body), 404


@app.get("/api/now/table/incident")
def list_incidents():
    rows = list(INCIDENTS.values())
    for field in ("category", "priority"):
        wanted = request.args.get(field)
        if wanted:
            rows = [r for r in rows if r[field].lower() == wanted.lower()]
    return jsonify({"result": rows})


@app.get("/api/now/table/incident/<number>")
def get_incident(number):
    row = INCIDENTS.get(number.upper())
    return jsonify({"result": row}) if row else not_found(number)


@app.patch("/api/now/table/incident/<number>")
def update_incident(number):
    row = INCIDENTS.get(number.upper())
    if not row:
        return not_found(number)
    updates = request.get_json(silent=True)
    if not isinstance(updates, dict) or not updates:
        return jsonify({"error": {"message": "Request body must be a JSON object"},
                        "status": "failure"}), 400
    unknown = sorted(set(updates) - set(row))
    if unknown:
        return jsonify({"error": {"message": f"Unknown field(s): {unknown}"},
                        "status": "failure"}), 400
    updates.pop("number", None)          # the record key can't be changed
    row.update({k: str(v) for k, v in updates.items()})
    return jsonify({"result": row})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "servicenow-mock",
                    "incidents": len(INCIDENTS),
                    "time": datetime.now().isoformat(timespec="seconds")})


if __name__ == "__main__":
    print(f"Loaded {len(INCIDENTS)} incidents from {CSV_PATH}")
    app.run(host="127.0.0.1", port=5001, debug=False)
