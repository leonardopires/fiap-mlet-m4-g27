import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psutil
import shutil
import os
import pandas as pd
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prometheus_fastapi_instrumentator import Instrumentator

from utils.data_preprocessing import (
    get_stock_data,
    preprocess_data,
    prepare_prediction_input,
    prepare_test_data,
    preprocess_user_data,
)
from utils.model_utils import build_model, train_model, predict_price, load_trained_model
from utils.security import get_api_key
import joblib

# Verifica se o modo de depuração está habilitado
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Inicializa a aplicação FastAPI com detalhes para a documentação
app = FastAPI(
    title="API de Previsão de Ações",
    description="API para treinar modelos e prever preços de ações usando modelos LSTM.",
    version="1.0.0",
    debug=DEBUG
)
print(f"APP in main {app}")

# Configura o Prometheus para monitoramento
Instrumentator().instrument(app).expose(app)

# Define o diretório onde os modelos treinados serão salvos
MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)  # Cria o diretório se ele não existir


# Definição dos modelos de entrada e saída para os endpoints
class TrainRequest(BaseModel):
    ticker: str = Field(..., description="Código da ação a ser treinada",
                        example="AAPL")  # Código da ação a ser treinada

    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL"
            }
        }


class PredictResponse(BaseModel):
    ticker: str = Field(..., description="Código da ação prevista", example="AAPL")  # Código da ação
    predicted_price: float = Field(..., description="Preço previsto da ação", example=150.25)  # Preço previsto

    class Config:
        schema_extra = {
            "example": {
                "ticker": "AAPL",
                "predicted_price": 150.25
            }
        }


class StatusResponse(BaseModel):
    model_exists: bool = Field(..., description="Indica se o modelo existe")  # Indica se o modelo existe
    performance_metrics: dict = Field(
        ...,
        description="Métricas de desempenho do modelo",
        example={"MAE": 5.123, "RMSE": 6.789}
    )  # Métricas de desempenho do modelo
    system_usage: dict = Field(
        ...,
        description="Uso de recursos do sistema",
        example={
            "cpu_usage_percent": 15.0,
            "memory_total_mb": 8192.0,
            "memory_available_mb": 4096.0,
            "disk_total_gb": 256.0,
            "disk_used_gb": 128.0,
            "disk_free_gb": 128.0,
        }
    )  # Uso de recursos do sistema


class PredictionsResponse(BaseModel):
    predictions: list = Field(..., description="Lista de preços previstos",
                              example=[150.25, 151.30, 152.10])  # Lista de preços previstos


# Endpoint para treinar um modelo com base em um ticker
@app.post(
    "/train",
    summary="Treinar modelo para um ticker",
    description="Inicia o treinamento de um modelo LSTM para o ticker fornecido. O treinamento ocorre em segundo plano e o modelo será salvo para uso futuro.",
    responses={
        202: {
            "description": "Treinamento iniciado. O modelo estará disponível após o término do treinamento.",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Treinamento iniciado para AAPL. O modelo estará disponível após o término do treinamento."
                    }
                }
            },
        },
        400: {
            "description": "Ticker não fornecido. Informe o ticker como query string ou JSON.",
            "content": {
                "application/json": {
                    "example": {"detail": "Ticker não fornecido. Informe o ticker como query string ou JSON."}
                }
            },
        },
    },
)
async def train_endpoint(
        ticker: str = None,  # Parâmetro opcional como query string
        request: TrainRequest = None,  # Modelo opcional como corpo da requisição
        background_tasks: BackgroundTasks = None,
        api_key: str = Depends(get_api_key)
):
    """
    Endpoint para iniciar o treinamento de um modelo LSTM com base em um ticker.

    Parâmetros:
        ticker (str): Código do ticker da ação (query string ou corpo).
        request (TrainRequest): Modelo contendo o ticker (corpo da requisição).
        background_tasks (BackgroundTasks): Gerenciamento de tarefas em segundo plano.
        api_key (str): Chave de API para autenticação.

    Retorna:
        JSONResponse: Mensagem indicando o status do treinamento.
    """
    # Prioriza o ticker vindo da query string
    if ticker is None and request:
        ticker = request.ticker  # Se não houver query string, tenta pegar do corpo da requisição

    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker não fornecido. Informe o ticker como query string ou JSON.")

    ticker = ticker.upper()  # Converte o ticker para letras maiúsculas
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")

    # Verifica se o modelo já existe
    if os.path.exists(model_path):
        return {"message": f"Modelo para {ticker} já existe."}

    # Adiciona a tarefa de treinamento em segundo plano
    background_tasks.add_task(train_and_save_model, ticker)

    # Retorna o status 202 Accepted
    return JSONResponse(
        status_code=202,
        content={
            "message": f"Treinamento iniciado para {ticker}. O modelo estará disponível após o término do treinamento."}
    )


def train_and_save_model(ticker):
    """
    Realiza o treinamento do modelo e o salva no diretório especificado.

    Parâmetros:
        ticker (str): Código da ação para treinamento.
    """
    # Busca os dados históricos
    df = get_stock_data(ticker)
    if df is None or df.empty:
        print(f"Nenhum dado encontrado para o ticker {ticker}.")
        return

    # Pré-processa os dados
    X_train, y_train, scaler = preprocess_data(df)

    # Constrói o modelo
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Treina o modelo
    train_model(model, X_train, y_train)

    # Salva o modelo e o scaler
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    model.save(model_path)
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print(f"Modelo para {ticker} salvo com sucesso.")


# Endpoint para fazer previsões com base em um ticker
@app.get(
    "/predict",
    response_model=PredictResponse,
    summary="Prever o preço de fechamento para um ticker",
    description="Utiliza o modelo treinado para prever o próximo preço de fechamento da ação especificada pelo ticker."
)
async def predict_endpoint(ticker: str):
    """
    Endpoint para prever o preço de fechamento de uma ação.

    Parâmetros:
        ticker (str): Código da ação.

    Retorna:
        PredictResponse: Resposta contendo o ticker e o preço previsto.
    """
    ticker = ticker.upper()
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    # Verifica se o modelo existe
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail=f"Modelo para {ticker} não encontrado. Treine o modelo primeiro.")

    # Carrega o modelo e o scaler
    model = load_trained_model(model_path)
    scaler = joblib.load(scaler_path)

    # Obtém os dados mais recentes
    df = get_stock_data(ticker)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"Nenhum dado encontrado para o ticker {ticker}.")

    # Prepara os dados para previsão
    X_input = prepare_prediction_input(df, scaler)
    if X_input is None:
        raise HTTPException(status_code=400, detail="Dados insuficientes para previsão.")

    # Realiza a previsão
    predicted_price = predict_price(model, X_input, scaler)

    return {"ticker": ticker, "predicted_price": predicted_price}


# Endpoint para verificar o status do modelo e os recursos do sistema
@app.get(
    "/status",
    response_model=StatusResponse,
    summary="Obter o status do modelo e uso do sistema",
    description="Fornece informações sobre a existência do modelo, métricas de desempenho e uso atual de recursos do sistema."
)
async def status_endpoint(ticker: str, api_key: str = Depends(get_api_key)):
    """
    Endpoint para verificar o status do modelo e o uso de recursos do sistema.

    Parâmetros:
        ticker (str): Código da ação.
        api_key (str): Chave de API para autenticação.

    Retorna:
        StatusResponse: Informações sobre o modelo e o sistema.
    """
    ticker = ticker.upper()
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    model_exists = os.path.exists(model_path) and os.path.exists(scaler_path)
    performance_metrics = {}

    # Calcula as métricas de desempenho se o modelo existir
    if model_exists:
        model = load_trained_model(model_path)
        scaler = joblib.load(scaler_path)
        df = get_stock_data(ticker)
        X_test, y_test = prepare_test_data(df, scaler)
        if X_test is not None and y_test is not None:
            predictions = model.predict(X_test)
            predicted_prices = scaler.inverse_transform(
                np.concatenate([predictions, np.zeros((predictions.shape[0], 4))], axis=1)
            )[:, 0]
            real_prices = scaler.inverse_transform(
                np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 4))], axis=1)
            )[:, 0]
            mae = mean_absolute_error(real_prices, predicted_prices)
            rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
            performance_metrics = {"MAE": mae, "RMSE": rmse}
    else:
        performance_metrics = {"MAE": None, "RMSE": None}

    # Obtém o uso de recursos do sistema
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage("/")

    system_usage = {
        "cpu_usage_percent": cpu_usage,
        "memory_total_mb": memory.total / (1024 * 1024),
        "memory_available_mb": memory.available / (1024 * 1024),
        "disk_total_gb": disk.total / (1024 * 1024 * 1024),
        "disk_used_gb": disk.used / (1024 * 1024 * 1024),
        "disk_free_gb": disk.free / (1024 * 1024 * 1024),
    }

    return {
        "model_exists": model_exists,
        "performance_metrics": performance_metrics,
        "system_usage": system_usage,
    }


# Endpoint para prever preços com base em um arquivo enviado
@app.post(
    "/predict_from_file",
    response_model=PredictionsResponse,
    summary="Prever preços a partir de um arquivo CSV enviado",
    description="Permite ao usuário enviar um arquivo CSV com dados históricos para gerar previsões personalizadas usando o modelo treinado."
)
async def predict_from_file(
        ticker: str, file: UploadFile = File(...), api_key: str = Depends(get_api_key)
):
    """
    Endpoint para realizar previsões com base em dados enviados via arquivo CSV.

    Parâmetros:
        ticker (str): Código da ação.
        file (UploadFile): Arquivo CSV contendo os dados históricos.
        api_key (str): Chave de API para autenticação.

    Retorna:
        PredictionsResponse: Lista de preços previstos.
    """
    ticker = ticker.upper()
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail=f"Modelo para {ticker} não encontrado. Treine o modelo primeiro.")

    # Carrega o modelo e o scaler
    model = load_trained_model(model_path)
    scaler = joblib.load(scaler_path)

    # Lê o arquivo enviado
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler o arquivo enviado: {e}")

    # Pré-processa os dados
    try:
        X_input = preprocess_user_data(df, scaler)
        if X_input is None:
            raise HTTPException(status_code=400, detail="Dados insuficientes para previsão após o pré-processamento.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao pré-processar os dados: {e}")

    # Faz as previsões
    try:
        predictions = model.predict(X_input)
        predictions_full = np.concatenate([predictions, np.zeros((predictions.shape[0], 4))], axis=1)
        predicted_prices = scaler.inverse_transform(predictions_full)[:, 0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao fazer as previsões: {e}")

    return {"predictions": [float(price) for price in predicted_prices]}


@app.delete("/delete_model", summary="Deletar modelo", description="Remove o modelo treinado para um ticker.")
async def delete_model_endpoint(ticker: str, api_key: str = Depends(get_api_key)):
    """Endpoint para deletar o modelo treinado de um ticker."""
    ticker = ticker.upper()
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(scaler_path):
        os.remove(scaler_path)

    return {"message": f"Modelo para {ticker} removido com sucesso."}


# Executa a aplicação se o script for executado diretamente
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
