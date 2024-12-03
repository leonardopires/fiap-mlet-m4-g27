import requests
import pytest
import os

# URL base para a API em execução
BASE_URL = "http://localhost:8000"

# Chave de API para autenticação
API_KEY = os.getenv("API_KEY", "dead-beef-15-bad-f00d")
HEADERS = {"access_token": API_KEY}

DEFAULT_TICKER = "AAPL"


@pytest.fixture(scope="module")
def setup_env():
    """Configurações iniciais para os testes."""
    yield
    # Cleanup ou teardown se necessário


def test_train_endpoint_with_cleanup(setup_env):
    """Testa o treinamento de modelo com limpeza prévia."""
    # Limpa o modelo, se existir
    requests.delete(f"{BASE_URL}/delete_model", headers=HEADERS, params={"ticker": DEFAULT_TICKER})

    # Agora testa o treinamento
    payload = {"ticker": DEFAULT_TICKER}
    response = requests.post(f"{BASE_URL}/train", headers=HEADERS, json=payload)

    assert response.status_code == 202, "Treinamento deve retornar 202 após limpeza"
    assert "Treinamento iniciado" in response.json()["message"]


def test_predict_endpoint(setup_env):
    """Testa a previsão de preços após treinamento."""
    params = {"ticker": DEFAULT_TICKER}
    response = requests.get(f"{BASE_URL}/predict", headers=HEADERS, params=params)
    if response.status_code == 200:
        assert "predicted_price" in response.json(), "Resposta deve conter 'predicted_price'"
    else:
        assert response.status_code in [400, 404], "Status inválido para previsão"


def test_status_endpoint(setup_env):
    """Testa o endpoint de status para o modelo treinado."""
    params = {"ticker": DEFAULT_TICKER}
    response = requests.get(f"{BASE_URL}/status", headers=HEADERS, params=params)
    assert response.status_code == 200, "Status deve retornar 200 OK"
    data = response.json()
    assert "model_exists" in data, "Resposta deve conter 'model_exists'"
    assert "performance_metrics" in data, "Resposta deve conter 'performance_metrics'"
    assert "system_usage" in data, "Resposta deve conter 'system_usage'"


def test_predict_from_file(setup_env):
    """Testa previsões usando arquivo CSV enviado."""
    # Prepara dados CSV para o teste
    sample_data = '\n'.join(
        ["Date,Open,High,Low,Close,Volume"] +
        [f"2023-01-{str(i + 1).zfill(2)},100,110,90,105,1000000" for i in range(61)]
    )
    files = {"file": ("test.csv", sample_data)}
    response = requests.post(
        f"{BASE_URL}/predict_from_file?ticker={DEFAULT_TICKER}",
        headers=HEADERS,
        files=files
    )
    assert response.status_code == 200, "Previsão com arquivo deve retornar 200 OK"
    data = response.json()
    assert "predictions" in data, "Resposta deve conter 'predictions'"
    assert isinstance(data["predictions"], list), "'predictions' deve ser uma lista"
