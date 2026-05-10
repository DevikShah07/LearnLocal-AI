"""
flask_app.py — LearnLocal Question Generation Web UI
Runs on http://localhost:5000
Proxies calls to the FastAPI backend at http://localhost:8000
"""

import requests
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)
API_BASE = "http://localhost:8000"


@app.route("/")
def index():
    return render_template("index.html")


# ── Proxy: PDF generate ────────────────────────────────────────────────────────
@app.route("/proxy/generate", methods=["POST"])
def proxy_generate():
    try:
        debug = request.args.get("debug", "false")
        files = {"file": (request.files["file"].filename, request.files["file"].read(), "application/pdf")}
        data  = {"config": request.form.get("config", "{}")}
        r = requests.post(f"{API_BASE}/api/v1/generate?debug={debug}", files=files, data=data, timeout=120)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.exceptions.ConnectionError:
        return jsonify({"detail": "Cannot reach FastAPI server at port 8000. Is it running?"}), 503


# ── Proxy: Text generate ───────────────────────────────────────────────────────
@app.route("/proxy/generate/text", methods=["POST"])
def proxy_generate_text():
    try:
        debug = request.args.get("debug", "false")
        data  = {"text": request.form.get("text", ""), "config": request.form.get("config", "{}")}
        r = requests.post(f"{API_BASE}/api/v1/generate/text?debug={debug}", data=data, timeout=120)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.exceptions.ConnectionError:
        return jsonify({"detail": "Cannot reach FastAPI server at port 8000. Is it running?"}), 503


# ── Proxy: Health ──────────────────────────────────────────────────────────────
@app.route("/proxy/health")
def proxy_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.exceptions.ConnectionError:
        return jsonify({"status": "offline"}), 503


# ── Proxy: FAISS — list documents ──────────────────────────────────────────────
@app.route("/proxy/docs", methods=["GET"])
def proxy_list_docs():
    try:
        r = requests.get(f"{API_BASE}/api/v1/docs", timeout=10)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.exceptions.ConnectionError:
        return jsonify({"detail": "Cannot reach FastAPI server at port 8000."}), 503


# ── Proxy: FAISS — semantic search ─────────────────────────────────────────────
@app.route("/proxy/docs/<doc_id>/search", methods=["POST"])
def proxy_search(doc_id):
    try:
        r = requests.post(
            f"{API_BASE}/api/v1/docs/{doc_id}/search",
            json=request.get_json(),
            timeout=30,
        )
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.exceptions.ConnectionError:
        return jsonify({"detail": "Cannot reach FastAPI server at port 8000."}), 503


# ── Proxy: FAISS — delete document ─────────────────────────────────────────────
@app.route("/proxy/docs/<doc_id>", methods=["DELETE"])
def proxy_delete_doc(doc_id):
    try:
        r = requests.delete(f"{API_BASE}/api/v1/docs/{doc_id}", timeout=10)
        return Response(r.content, status=r.status_code, content_type="application/json")
    except requests.exceptions.ConnectionError:
        return jsonify({"detail": "Cannot reach FastAPI server at port 8000."}), 503


if __name__ == "__main__":
    app.run(debug=True, port=5000)
