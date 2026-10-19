"""
Automated tests. Run from the project folder with:
    python -m unittest discover -s tests -v
They use a temporary database, so your real detection history is never touched.
"""
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as cryptexora          # noqa: E402  (import after the path fix)
import database                   # noqa: E402
import ml_model                   # noqa: E402

NORMAL = {"ip_address": "192.168.1.10", "duration": 12.5, "packet_count": 45,
          "bytes_transferred": 38000, "failed_connections": 0}
DDOS = {"ip_address": "203.0.113.45", "duration": 2.1, "packet_count": 8500,
        "bytes_transferred": 620000, "failed_connections": 0}
PROBE = {"ip_address": "198.51.100.23", "duration": 3.4, "packet_count": 90,
         "bytes_transferred": 6000, "failed_connections": 25}
BRUTE = {"ip_address": "203.0.113.99", "duration": 85, "packet_count": 150,
         "bytes_transferred": 30000, "failed_connections": 18}


class ModelTests(unittest.TestCase):
    """Does the real KNN model give sensible answers?"""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics = ml_model.load_model()

    def predict(self, sample):
        values = {name: sample[name] for name in ml_model.FEATURES}
        return ml_model.predict_traffic(self.model, self.metrics, values)

    def test_model_is_a_real_knn(self):
        self.assertEqual(type(self.model.named_steps["knn"]).__name__, "KNeighborsClassifier")

    def test_normal_traffic(self):
        self.assertEqual(self.predict(NORMAL)["prediction"], "normal")

    def test_suspicious_presets(self):
        for sample in (DDOS, PROBE, BRUTE):
            self.assertEqual(self.predict(sample)["prediction"], "suspicious", sample)

    def test_ip_address_is_not_a_feature(self):
        self.assertNotIn("ip_address", ml_model.FEATURES)

    def test_test_accuracy_is_reasonable(self):
        self.assertGreater(self.metrics["accuracy"], 0.9)


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_file.close()
        cryptexora.app.config.update(DATABASE=self.db_file.name, TESTING=True)
        with cryptexora.app.app_context():
            database.init_db()
        self.client = cryptexora.app.test_client()

    def tearDown(self):
        os.unlink(self.db_file.name)

    def get_token(self, url="/login"):
        html = self.client.get(url).get_data(as_text=True)
        return re.search(r'name="csrf-token" content="([0-9a-f]+)"', html).group(1)

    def login(self, username="admin", password="admin123"):
        token = self.get_token()
        return self.client.post("/login", data={"username": username, "password": password,
                                                "csrf_token": token})

    def predict(self, payload):
        token = self.get_token("/detect")
        return self.client.post("/api/predict", json=payload, headers={"X-CSRF-Token": token})

    # ----- login / security
    def test_pages_require_login(self):
        for url in ("/dashboard", "/detect", "/logs", "/overview"):
            self.assertEqual(self.client.get(url).status_code, 302, url)

    def test_wrong_password_rejected(self):
        response = self.login(password="wrong")
        self.assertIn("Invalid username or password", response.get_data(as_text=True))

    def test_login_success(self):
        self.assertEqual(self.login().status_code, 302)
        self.assertEqual(self.client.get("/dashboard").status_code, 200)

    def test_password_is_hashed(self):
        with cryptexora.app.app_context():
            row = database.get_db().execute("SELECT password_hash FROM users").fetchone()
        self.assertNotIn("admin123", row["password_hash"])

    def test_post_without_csrf_token_is_blocked(self):
        response = self.client.post("/login", data={"username": "admin", "password": "admin123"})
        self.assertEqual(response.status_code, 400)

    # ----- prediction API
    def test_api_normal_and_suspicious(self):
        self.login()
        self.assertEqual(self.predict(NORMAL).get_json()["prediction"], "normal")
        self.assertEqual(self.predict(DDOS).get_json()["prediction"], "suspicious")

    def test_invalid_ip_rejected(self):
        self.login()
        response = self.predict({**NORMAL, "ip_address": "999.1.1.1"})
        self.assertEqual(response.status_code, 400)

    def test_invalid_numbers_rejected(self):
        self.login()
        for bad in ({"packet_count": -5}, {"duration": "abc"}, {"failed_connections": ""},
                    {"bytes_transferred": 10 ** 12}, {"packet_count": 0}):
            self.assertEqual(self.predict({**NORMAL, **bad}).status_code, 400, bad)

    def test_same_ip_can_be_normal_or_suspicious(self):
        """An IP alone is not proof: the features decide."""
        self.login()
        same_ip = "10.0.0.7"
        self.assertEqual(self.predict({**NORMAL, "ip_address": same_ip}).get_json()["prediction"], "normal")
        self.assertEqual(self.predict({**DDOS, "ip_address": same_ip}).get_json()["prediction"], "suspicious")

    # ----- history (SQLite)
    def test_history_filter_export_and_clear(self):
        self.login()
        self.predict(NORMAL)
        self.predict(PROBE)
        page = self.client.get("/logs").get_data(as_text=True)
        self.assertIn("192.168.1.10", page)
        self.assertIn("198.51.100.23", page)
        only_bad = self.client.get("/logs?filter=suspicious").get_data(as_text=True)
        self.assertIn("198.51.100.23", only_bad)
        self.assertNotIn("192.168.1.10", only_bad)
        csv_text = self.client.get("/logs/export.csv").get_data(as_text=True)
        self.assertEqual(len(csv_text.strip().splitlines()), 3)  # header + 2 rows
        token = self.get_token("/logs")
        self.client.post("/logs/clear", data={"csrf_token": token})
        self.assertIn("No records found", self.client.get("/logs").get_data(as_text=True))

    def test_dashboard_counts(self):
        self.login()
        self.predict(NORMAL)
        self.predict(DDOS)
        page = self.client.get("/dashboard").get_data(as_text=True)
        self.assertRegex(page, r'stat-number">2<')


if __name__ == "__main__":
    unittest.main()
