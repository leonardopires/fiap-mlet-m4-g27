import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


# Função para obter os dados históricos de ações usando o Yahoo Finance
def get_stock_data(ticker):
    """
    Obtém os dados históricos de ações para um ticker específico.

    Parâmetros:
        ticker (str): Código do ativo (ex.: 'AAPL', 'GOOG').

    Retorna:
        pd.DataFrame ou None: Um DataFrame com os dados históricos da ação
        ou None se ocorrer um erro ou os dados estiverem indisponíveis.
    """
    try:
        # Baixa os dados usando o yfinance
        df = yf.download(ticker, start='2010-01-01', end='2024-01-01')
        if df.empty:
            return None  # Retorna None se o DataFrame estiver vazio
        df.reset_index(inplace=True)  # Reseta o índice para incluir a coluna 'Date'
        return df
    except Exception as e:
        # Log de erro se algo der errado durante o download
        print(f"Erro ao buscar os dados para o ticker {ticker}: {e}")
        return None


# Função para pré-processar os dados históricos para treinamento
def preprocess_data(df):
    """
    Pré-processa os dados históricos para uso no treinamento do modelo.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados históricos de ações.

    Retorna:
        tuple: X (entradas), y (saídas) e o scaler usado para normalização.
    """
    # Seleciona as colunas de interesse
    features = df[['Close', 'High', 'Low', 'Open', 'Volume']]
    # Garante que os dados sejam numéricos, removendo valores ausentes
    features = features.apply(pd.to_numeric, errors='coerce').dropna()
    # Normaliza os dados para o intervalo [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    # Define o tamanho das sequências (janelas) temporais
    sequence_length = 60
    X, y = [], []
    # Cria as sequências para treinamento
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])  # 60 dias anteriores como entrada
        y.append(scaled_data[i, 0])  # Preço de fechamento como saída
    # Converte para arrays numpy
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler


# Função para preparar os dados para previsões futuras
def prepare_prediction_input(df, scaler):
    """
    Prepara os dados mais recentes para realizar previsões.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados históricos de ações.
        scaler (MinMaxScaler): Scaler usado para normalizar os dados.

    Retorna:
        np.ndarray ou None: Dados processados prontos para previsão ou None
        se os dados forem insuficientes.
    """
    sequence_length = 60
    features = df[['Close', 'High', 'Low', 'Open', 'Volume']]
    features = features.apply(pd.to_numeric, errors='coerce').dropna()

    # Verifica se há dados suficientes para criar uma sequência
    if len(features) < sequence_length:
        return None

    # Normaliza os dados e seleciona os últimos 60 dias
    scaled_features = scaler.transform(features)
    X_input = scaled_features[-sequence_length:]
    X_input = np.expand_dims(X_input, axis=0)  # Expande a dimensão para (1, 60, 5)
    return X_input


# Função para preparar os dados para teste
def prepare_test_data(df, scaler):
    """
    Pré-processa os dados históricos para criar conjuntos de teste.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados históricos de ações.
        scaler (MinMaxScaler): Scaler usado para normalizar os dados.

    Retorna:
        tuple: X_test (dados de entrada para teste) e y_test (valores reais).
    """
    # Pré-processa as colunas de interesse
    features = df[['Close', 'High', 'Low', 'Open', 'Volume']]
    features = features.apply(pd.to_numeric, errors='coerce').dropna()
    scaled_data = scaler.transform(features)

    sequence_length = 60
    X, y = [], []
    # Cria sequências temporais para o conjunto de teste
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])  # 60 dias anteriores como entrada
        y.append(scaled_data[i, 0])  # Preço de fechamento como saída

    if len(X) == 0 or len(y) == 0:
        return None, None

    # Seleciona os últimos 20% dos dados como conjunto de teste
    X = np.array(X)
    y = np.array(y)
    test_size = int(len(X) * 0.2)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    return X_test, y_test


# Função para pré-processar os dados enviados pelo usuário
def preprocess_user_data(df, scaler):
    """
    Pré-processa os dados enviados pelo usuário para previsões personalizadas.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo os dados históricos enviados pelo usuário.
        scaler (MinMaxScaler): Scaler usado para normalizar os dados.

    Retorna:
        np.ndarray ou None: Dados processados prontos para previsão ou None
        se os dados forem insuficientes.
    """
    # Verifica se as colunas necessárias estão presentes
    required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"Os dados devem conter as seguintes colunas: {required_columns}")

    # Seleciona e organiza as colunas necessárias
    features = df[required_columns]
    features = features.apply(pd.to_numeric, errors='coerce').dropna()
    # Normaliza os dados usando o scaler existente
    scaled_data = scaler.transform(features)

    # Cria sequências temporais
    sequence_length = 60
    X_input = []
    for i in range(sequence_length, len(scaled_data)):
        X_input.append(scaled_data[i - sequence_length:i])

    if len(X_input) == 0:
        return None

    X_input = np.array(X_input)
    return X_input
