# tests/test_api.py

from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.main import app
from utils.security import API_KEY_NAME, API_KEY

client = TestClient(app)

def test_train_endpoint():
    response = client.post(
        "/train",
        headers={API_KEY_NAME: API_KEY},
        json={"ticker": "AAPL"}
    )
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_endpoint():
    response = client.get(
        "/predict",
        headers={API_KEY_NAME: API_KEY},
        params={"ticker": "AAPL"}
    )
    if response.status_code == 200:
        assert "predicted_price" in response.json()
    else:
        assert response.status_code == 404 or response.status_code == 400

def test_status_endpoint():
    response = client.get(
        "/status",
        headers={API_KEY_NAME: API_KEY},
        params={"ticker": "AAPL"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "model_exists" in data
    assert "performance_metrics" in data
    assert "system_usage" in data

def test_predict_from_file():
    # Prepare a sample CSV content with at least 60 rows
    sample_data = '\n'.join([
        "Date,Open,High,Low,Close,Volume"
    ] + [
        f"2023-01-{str(i+1).zfill(2)},100,110,90,105,1000000" for i in range(61)
    ])
    files = {'file': ('test.csv', sample_data, 'text/csv')}
    response = client.post(
        "/predict_from_file?ticker=AAPL",
        headers={API_KEY_NAME: API_KEY},
        files=files
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)