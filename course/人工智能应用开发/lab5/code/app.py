from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")
DATA_PATH = Path(__file__).resolve().parent / "data.json"


def read_data():
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def write_data(data):
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/dashboard")
def dashboard():
    data = read_data()
    records = data["records"]
    metrics = [
        {"label": "提交总数", "value": len(records)},
        {"label": "待评审", "value": len([x for x in records if x["status"] in {"submitted", "under_review"}])},
        {"label": "已通过", "value": len([x for x in records if x["status"] == "approved"])},
        {"label": "已驳回", "value": len([x for x in records if x["status"] == "rejected"])},
    ]
    return jsonify(
        {
            "app_name": data["app_name"],
            "entity_name": data["entity_name"],
            "fields": data["fields"],
            "create_fields": data.get("create_fields", data["fields"]),
            "statuses": data["statuses"],
            "metrics": metrics,
            "list_columns": data.get("list_columns", []),
            "filters": data.get("filters", []),
            "workflow_steps": data.get("workflow_steps", []),
            "records": records,
        }
    )


@app.post("/api/records")
def create_record():
    data = read_data()
    payload = request.get_json(force=True)
    record = {
        "id": f"{data['entity_name'].replace(' ', '_')}_{len(data['records']) + 1}",
        "status": data["statuses"][0],
        "created_at": "刚刚生成",
        "reviewed_at": "",
        "reviewer": "",
        "review_comment": "",
    }
    for field in data.get("create_fields", data["fields"]):
        value = payload.get(field["key"], "")
        record[field["key"]] = int(value) if field["type"] == "number" and str(value).strip() else str(value).strip()
    data["records"].insert(0, record)
    write_data(data)
    return jsonify({"record": record}), 201


@app.patch("/api/records/<record_id>/review")
def review_record(record_id: str):
    data = read_data()
    payload = request.get_json(force=True)
    record = next((item for item in data["records"] if item["id"] == record_id), None)
    if record is None:
        return jsonify({"error": "未找到记录。"}), 404
    record["reviewer"] = str(payload.get("reviewer", "")).strip()
    record["review_comment"] = str(payload.get("review_comment", "")).strip()
    record["status"] = str(payload.get("status", record["status"])).strip()
    record["reviewed_at"] = "刚刚生成"
    write_data(data)
    return jsonify({"record": record})


@app.get("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
