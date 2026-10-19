// detect.js - presets, form submit, and showing the prediction result

// Example traffic values for the quick-fill buttons (demo data, no real network needed)
var PRESETS = {
  normal: { ip_address: "192.168.1.10",  duration: 12.5, packet_count: 45,   bytes_transferred: 38000,  failed_connections: 0 },
  ddos:   { ip_address: "203.0.113.45",  duration: 2.1,  packet_count: 8500, bytes_transferred: 620000, failed_connections: 0 },
  probe:  { ip_address: "198.51.100.23", duration: 3.4,  packet_count: 90,   bytes_transferred: 6000,   failed_connections: 25 },
  brute:  { ip_address: "203.0.113.99",  duration: 85,   packet_count: 150,  bytes_transferred: 30000,  failed_connections: 18 }
};

var form = document.getElementById("detect-form");
var errorBox = document.getElementById("form-error");
var analyzeButton = document.getElementById("analyze-btn");

// 1) Quick-fill buttons
document.querySelectorAll("[data-preset]").forEach(function (button) {
  button.addEventListener("click", function () {
    var preset = PRESETS[button.dataset.preset];
    Object.keys(preset).forEach(function (name) {
      form.elements[name].value = preset[name];
    });
    errorBox.hidden = true;
  });
});

// 2) Submit the form to the Flask API and show the answer
form.addEventListener("submit", function (event) {
  event.preventDefault();
  errorBox.hidden = true;
  analyzeButton.disabled = true;

  var payload = {};
  Array.prototype.forEach.call(form.elements, function (field) {
    if (field.name) { payload[field.name] = field.value; }
  });

  fetch("/api/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content
    },
    body: JSON.stringify(payload)
  })
    .then(function (response) {
      return response.json().then(function (data) { return { ok: response.ok, data: data }; });
    })
    .then(function (reply) {
      if (!reply.ok) { throw new Error(reply.data.error || "Something went wrong."); }
      showResult(reply.data);
    })
    .catch(function (error) {
      errorBox.textContent = error.message;
      errorBox.hidden = false;
    })
    .then(function () { analyzeButton.disabled = false; });
});

// Add one "label ... value" row to the result list (textContent keeps it XSS-safe)
function addRow(list, label, value, pillClass) {
  var item = document.createElement("li");
  var left = document.createElement("span");
  left.textContent = label;
  var right = document.createElement("strong");
  right.textContent = value;
  if (pillClass) { right.className = "pill " + pillClass; }
  item.appendChild(left);
  item.appendChild(right);
  list.appendChild(item);
}

function showResult(result) {
  var suspicious = result.prediction === "suspicious";
  document.getElementById("result-empty").hidden = true;
  document.getElementById("result-box").hidden = false;

  document.getElementById("result-banner").className =
    "result-banner " + (suspicious ? "result-suspicious" : "result-normal");
  document.getElementById("result-emoji").textContent = suspicious ? "🚨" : "✅";
  document.getElementById("result-text").textContent =
    suspicious ? "SUSPICIOUS TRAFFIC" : "TRAFFIC NORMAL";

  var rows = document.getElementById("result-rows");
  rows.innerHTML = "";
  addRow(rows, "IP ADDRESS", result.ip_address);
  addRow(rows, "DURATION", result.features.duration + " s");
  addRow(rows, "PACKETS", result.features.packet_count);
  addRow(rows, "BYTES", result.features.bytes_transferred);
  addRow(rows, "FAILED CONNECTIONS", result.features.failed_connections);
  addRow(rows, "CONFIDENCE", result.confidence + "%");
  addRow(rows, "SEVERITY", result.severity.toUpperCase(), "pill-" + result.severity);
  addRow(rows, "SUMMARY", result.summary);

  var reasonBox = document.getElementById("reason-box");
  var reasonList = document.getElementById("reason-list");
  reasonList.innerHTML = "";
  result.reasons.forEach(function (text) {
    var item = document.createElement("li");
    item.textContent = text;
    reasonList.appendChild(item);
  });
  reasonBox.hidden = result.reasons.length === 0;
}
