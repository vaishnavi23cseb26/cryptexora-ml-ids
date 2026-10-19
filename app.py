"""
app.py - Cryptexora ML-based Intrusion Detection System (Flask backend)

Pages:   /login  /dashboard  /overview  /detect  /logs
API:     POST /api/predict   (used by the detection form via JavaScript)
"""
import csv
import io
import ipaddress
import os
import secrets
from datetime import datetime
from functools import wraps

from flask import (Flask, Response, abort, flash, jsonify, redirect,
                   render_template, request, session, url_for)
from werkzeug.security import check_password_hash

import database
import ml_model

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-change-me-in-production")
app.config["DATABASE"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cryptexora.db")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.teardown_appcontext(database.close_db)

# Load the trained KNN model once when the server starts.
model, model_metrics = ml_model.load_model()


# ---------------------------------------------------------------- security helpers
@app.context_processor
def inject_globals():
    """Make csrf_token and the logged-in user available in every template."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(16)
    return {"csrf_token": session["csrf_token"], "current_user": session.get("username")}


@app.before_request
def check_csrf_token():
    """Block forged POST requests: every POST must carry our secret token."""
    if request.method == "POST":
        sent_token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token") or ""
        real_token = session.get("csrf_token", "")
        if not real_token or not secrets.compare_digest(sent_token, real_token):
            abort(400, "Invalid or missing security token. Please refresh the page.")


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


# ---------------------------------------------------------------- input validation
def validate_traffic_input(data):
    """Check the form values. Returns (clean_values, ip, error_message)."""
    ip_text = str(data.get("ip_address", "")).strip()
    try:
        ip = str(ipaddress.ip_address(ip_text))  # accepts IPv4 and IPv6
    except ValueError:
        return None, None, "Please enter a valid IP address (for example 192.168.1.10)."

    clean = {}
    for name in ml_model.FEATURES:
        info = ml_model.FEATURE_INFO[name]
        try:
            number = float(data.get(name, ""))
        except (TypeError, ValueError):
            return None, None, f"{info['label']} must be a number."
        if number != number or number < 0 or number > info["max"]:  # number != number -> NaN
            return None, None, f"{info['label']} must be between 0 and {info['max']:,}."
        if name != "duration":
            number = int(number)  # counts are whole numbers
        clean[name] = number
    if clean["packet_count"] < 1:
        return None, None, "Packet count must be at least 1."
    return clean, ip, None


# ---------------------------------------------------------------- login / logout
@app.route("/")
def home():
    return redirect(url_for("dashboard" if "username" in session else "login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = database.get_db().execute(
            "SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()  # start a fresh session after login
            session["username"] = user["username"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------- pages
@app.route("/dashboard")
@login_required
def dashboard():
    db = database.get_db()
    stats = db.execute(
        "SELECT COUNT(*) AS total, "
        "COALESCE(SUM(prediction = 'suspicious'), 0) AS suspicious, "
        "COALESCE(SUM(prediction = 'normal'), 0) AS normal FROM detections").fetchone()
    recent = db.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 5").fetchall()
    suspicious_percent = round(stats["suspicious"] * 100 / stats["total"]) if stats["total"] else 0
    return render_template("dashboard.html", page="dashboard", stats=stats, recent=recent,
                           suspicious_percent=suspicious_percent, metrics=model_metrics)


@app.route("/overview")
@login_required
def overview():
    return render_template("overview.html", page="overview", metrics=model_metrics,
                           feature_info=ml_model.FEATURE_INFO)


@app.route("/detect")
@login_required
def detect():
    return render_template("detect.html", page="detect", feature_info=ml_model.FEATURE_INFO)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """Validate input -> run the KNN model -> save to SQLite -> return JSON."""
    clean, ip, error = validate_traffic_input(request.get_json(silent=True) or {})
    if error:
        return jsonify({"error": error}), 400

    result = ml_model.predict_traffic(model, model_metrics, clean)
    # Short text for the history table (the long explanation is shown on the result card)
    details = result["summary"]
    if result["unusual_features"]:
        details += " Unusually high: " + ", ".join(result["unusual_features"]) + "."

    db = database.get_db()
    db.execute(
        "INSERT INTO detections (created_at, username, ip_address, duration, packet_count, "
        "bytes_transferred, failed_connections, prediction, confidence, severity, details) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session["username"], ip,
         clean["duration"], clean["packet_count"], clean["bytes_transferred"],
         clean["failed_connections"], result["prediction"], result["confidence"],
         result["severity"], details))
    db.commit()

    result["ip_address"] = ip
    result["features"] = clean
    return jsonify(result)


def fetch_logs(result_filter, search_text):
    """Read detection history from SQLite (shared by the page and the CSV export)."""
    query = "SELECT * FROM detections WHERE 1=1"
    params = []
    if result_filter in ("normal", "suspicious"):
        query += " AND prediction = ?"
        params.append(result_filter)
    if search_text:
        query += " AND (ip_address LIKE ? OR details LIKE ? OR severity LIKE ?)"
        params += [f"%{search_text}%"] * 3
    query += " ORDER BY id DESC"
    return database.get_db().execute(query, params).fetchall()


@app.route("/logs")
@login_required
def logs():
    result_filter = request.args.get("filter", "all")
    search_text = request.args.get("q", "").strip()[:50]
    rows = fetch_logs(result_filter, search_text)
    totals = database.get_db().execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(prediction = 'suspicious'), 0) AS suspicious, "
        "COALESCE(SUM(prediction = 'normal'), 0) AS normal FROM detections").fetchone()
    latest = database.get_db().execute(
        "SELECT username FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    return render_template("logs.html", page="logs", rows=rows, totals=totals,
                           result_filter=result_filter, search_text=search_text,
                           latest_user=latest["username"] if latest else "-")


@app.route("/logs/export.csv")
@login_required
def export_logs():
    rows = fetch_logs(request.args.get("filter", "all"), request.args.get("q", "").strip()[:50])
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "time", "ip", "duration_s", "packets", "bytes", "failed",
                     "result", "confidence_%", "severity", "details"])
    for r in rows:
        writer.writerow([r["id"], r["created_at"], r["ip_address"], r["duration"],
                         r["packet_count"], r["bytes_transferred"], r["failed_connections"],
                         r["prediction"], r["confidence"], r["severity"], r["details"]])
    return Response(output.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=cryptexora_logs.csv"})


@app.route("/logs/clear", methods=["POST"])
@login_required
def clear_logs():
    db = database.get_db()
    db.execute("DELETE FROM detections")
    db.commit()
    flash("All detection logs were cleared.", "success")
    return redirect(url_for("logs"))


# Create the tables (and demo admin) when the app starts.
with app.app_context():
    database.init_db()

if __name__ == "__main__":
    # debug=True is for local learning only. Never use it on a public server.
    app.run(debug=True)
