from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Função para construir o modelo LSTM
def build_model(input_shape):
    """
    Constrói e compila um modelo LSTM para previsão de séries temporais.

    Parâmetros:
        input_shape (tuple): Formato dos dados de entrada (número de timesteps, número de features).

    Retorna:
        Sequential: O modelo LSTM compilado.
    """
    from tensorflow.keras.layers import Input  # Importa a camada Input

    # Inicializa o modelo sequencial
    model = Sequential()

    # Adiciona as camadas ao modelo
    model.add(Input(shape=input_shape))  # Define a forma da entrada
    model.add(LSTM(50, return_sequences=True))  # Primeira camada LSTM com 50 unidades
    model.add(Dropout(0.2))  # Dropout de 20% para evitar overfitting
    model.add(LSTM(50, return_sequences=False))  # Segunda camada LSTM sem retornar sequências
    model.add(Dropout(0.2))  # Dropout de 20% novamente
    model.add(Dense(25))  # Camada totalmente conectada com 25 neurônios
    model.add(Dense(1))  # Camada de saída para prever um único valor (preço de fechamento)

    # Compila o modelo com o otimizador Adam e a função de perda mean squared error
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# Função para treinar o modelo
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """
    Treina o modelo LSTM nos dados fornecidos.

    Parâmetros:
        model (Sequential): O modelo LSTM a ser treinado.
        X_train (np.ndarray): Dados de entrada para treinamento.
        y_train (np.ndarray): Valores reais (saídas) para treinamento.
        epochs (int): Número de épocas para o treinamento.
        batch_size (int): Tamanho do batch usado no treinamento.

    Retorna:
        None
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


# Função para salvar o modelo treinado
def save_model(model, model_path):
    """
    Salva o modelo treinado em um arquivo.

    Parâmetros:
        model (Sequential): O modelo LSTM a ser salvo.
        model_path (str): Caminho do arquivo onde o modelo será salvo.

    Retorna:
        None
    """
    model.save(model_path)


# Função para carregar um modelo treinado
def load_trained_model(model_path):
    """
    Carrega um modelo treinado a partir de um arquivo.

    Parâmetros:
        model_path (str): Caminho do arquivo onde o modelo está salvo.

    Retorna:
        Sequential: O modelo LSTM carregado.
    """
    model = load_model(model_path)
    return model


# Função para fazer previsões usando o modelo treinado
def predict_price(model, X_input, scaler):
    """
    Faz previsões de preço usando o modelo LSTM treinado.

    Parâmetros:
        model (Sequential): O modelo LSTM treinado.
        X_input (np.ndarray): Dados de entrada para a previsão (normalizados).
        scaler (MinMaxScaler): Scaler usado para normalizar os dados, necessário para a transformação inversa.

    Retorna:
        float: O preço previsto no formato original (desnormalizado).
    """
    import numpy as np

    # Faz a previsão
    prediction = model.predict(X_input)

    # Adiciona zeros para preencher os valores restantes nas colunas desnormalizadas
    prediction_full = np.concatenate([prediction, np.zeros((prediction.shape[0], 4))], axis=1)

    # Realiza a transformação inversa para retornar o preço original
    predicted_price = scaler.inverse_transform(prediction_full)[:, 0]

    return float(predicted_price[0])  # Retorna o primeiro valor previsto