# tests/test_end_to_end.py

import requests
import os
from utils.security import API_KEY_NAME, API_KEY

BASE_URL = "http://localhost:8000"


def test_end_to_end():
    # 1: Treinar o modelo
    response = requests.post(
        f"{BASE_URL}/train",
        headers={API_KEY_NAME: API_KEY},
        json={"ticker": "AAPL"}
    )
    assert response.status_code == 200
    assert "message" in response.json()

    # Esperar o modelo terminar de ser treinado
    import time
    time.sleep(60)  # Aguarda

    # 2: Verificar o status do modelo
    response = requests.get(
        f"{BASE_URL}/status",
        headers={API_KEY_NAME: API_KEY},
        params={"ticker": "AAPL"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_exists"] is True

    # 3: Fazer uma previsão
    response = requests.get(
        f"{BASE_URL}/predict",
        headers={API_KEY_NAME: API_KEY},
        params={"ticker": "AAPL"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
