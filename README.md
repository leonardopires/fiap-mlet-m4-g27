
# API de Previsão de Preços de Ações

## Autores

- [Felipe de Paula Gomes](https://github.com/Felipe-DePaula) (RM355402)
- [Jorge Guilherme Dalcorso Wald](https://github.com/JorgeWald) (355849)
- [Leonardo Petersen Thomé Pires](https://github.com/leonardopires) (RM355401)

## Índice

- [Características Principais](#características-principais)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Executando a Aplicação](#executando-a-aplicação)
- [Acessando a Documentação Swagger](#acessando-a-documentação-swagger)
- [Utilizando a API](#utilizando-a-api)
  - [Autenticação](#autenticação)
  - [Endpoints Disponíveis](#endpoints-disponíveis)
    - [/train](#train)
    - [/predict](#predict)
    - [/status](#status)
    - [/predict_from_file](#predict_from_file)
- [Monitoramento com Grafana e Prometheus](#monitoramento-com-grafana-e-prometheus)
  - [Acessando o Dashboard do Grafana](#acessando-o-dashboard-do-grafana)
- [Executando os Testes](#executando-os-testes)
- [Notas Adicionais](#notas-adicionais)

---

## Características Principais

- **Treinamento de Modelos**: Treine modelos LSTM personalizados para diferentes ações usando dados históricos.
- **Previsão de Preços**: Faça previsões de preços de fechamento para ações específicas.
- **Previsões Personalizadas**: Envie seus próprios dados históricos em formato CSV para gerar previsões personalizadas.
- **Monitoramento**: Utilize Prometheus e Grafana para monitorar a API e visualizar métricas importantes.
- **Documentação Interativa**: Acesse a documentação interativa da API via Swagger UI e ReDoc.
- **Segurança**: A API é protegida por uma chave de API para garantir acesso seguro.

## Tecnologias Utilizadas

- **FastAPI**: Framework web moderno e de alto desempenho para construção de APIs.
- **TensorFlow e Keras**: Bibliotecas para construção e treinamento de modelos de aprendizado profundo.
- **scikit-learn**: Biblioteca para pré-processamento de dados e cálculo de métricas.
- **yfinance**: Biblioteca para obtenção de dados históricos de ações.
- **Prometheus e Grafana**: Ferramentas para monitoramento e visualização de métricas.
- **Docker e Docker Compose**: Para containerização e fácil implantação da aplicação.

## Pré-requisitos

Certifique-se de ter as seguintes ferramentas instaladas em seu ambiente:

- **Docker**: [Instalação do Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Instalação do Docker Compose](https://docs.docker.com/compose/install/)

## Instalação

1. **Clone o repositório**:

   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. **Configuração da Chave de API**:

   - A chave de API é usada para autenticar as requisições.
   - Crie um arquivo `.env` na raiz do projeto e adicione a seguinte linha:

     ```env
     API_KEY=dead-beef-15-bad-f00d
     ```

   - **Nota**: Certifique-se de que esta chave corresponda à usada em suas configurações.

## Configuração

1. **Arquivo `.env`**:

   - O arquivo `.env` deve conter as configurações de ambiente necessárias.
   - Exemplo de conteúdo:

     ```env
     API_KEY=dead-beef-15-bad-f00d
     DEBUG=true
     ```

2. **Docker Compose**:

   - O arquivo `docker-compose.yml` está configurado para levantar os seguintes serviços:

     - **app**: A aplicação FastAPI.
     - **prometheus**: Serviço de monitoramento.
     - **grafana**: Interface de visualização de métricas.

   - Certifique-se de que as portas necessárias estão disponíveis em seu ambiente:

     - **8000**: Porta da aplicação FastAPI.
     - **9090**: Porta do Prometheus.
     - **3000**: Porta do Grafana.

## Executando a Aplicação

Para iniciar a aplicação e todos os serviços associados, execute:

```bash
docker-compose up
```

Este comando irá:

- Construir a imagem Docker da aplicação.
- Iniciar os serviços da API, Prometheus e Grafana.

## Acessando a Documentação Swagger

A FastAPI fornece automaticamente uma documentação interativa via Swagger UI.

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

Aqui você pode:

- Explorar os endpoints disponíveis.
- Ler as descrições detalhadas de cada endpoint.
- Testar os endpoints diretamente pelo navegador.


## Utilizando a API

### Autenticação

Alguns endpoints exigem autenticação via chave de API.

- **Cabeçalho de Autenticação**:

  ```
  access_token: dead-beef-15-bad-f00d
  ```

- No Swagger UI, clique em **Authorize** e insira a chave de API no campo apropriado.

### Endpoints Disponíveis

#### **/train**

- **Método**: `POST`
- **Descrição**: Inicia o treinamento de um modelo LSTM para o ticker fornecido.
- **Parâmetros**:
  - `ticker` (query string ou JSON): Código da ação a ser treinada (por exemplo, `AAPL`).
- **Autenticação**: Necessária.
- **Exemplo de Requisição**:

  - **Usando Query String**:

    ```
    POST /train?ticker=AAPL
    ```

  - **Usando JSON no Corpo**:

    ```json
    {
      "ticker": "AAPL"
    }
    ```

- **Resposta**:

  - **Status Code 202 Accepted**: Indica que o treinamento foi iniciado com sucesso.
  - **Mensagem**: Confirmação do início do treinamento.

#### **/predict**

- **Método**: `GET`
- **Descrição**: Utiliza o modelo treinado para prever o próximo preço de fechamento da ação especificada pelo ticker.
- **Parâmetros**:
  - `ticker` (query string): Código da ação (padrão: `AAPL`).
- **Autenticação**: Não necessária.
- **Exemplo de Requisição**:

  ```
  GET /predict?ticker=AAPL
  ```

- **Resposta**:

  ```json
  {
    "ticker": "AAPL",
    "predicted_price": 150.25
  }
  ```
  

#### **/status**

- **Método**: `GET`
- **Descrição**: Fornece informações sobre a existência do modelo, métricas de desempenho e uso atual de recursos do sistema.
- **Parâmetros**:
  - `ticker` (query string): Código da ação.
- **Autenticação**: Necessária.
- **Exemplo de Requisição**:

  ```
  GET /status?ticker=AAPL
  ```

- **Resposta**:

  ```json
  {
    "model_exists": true,
    "performance_metrics": {
      "MAE": 5.123,
      "RMSE": 6.789
    },
    "system_usage": {
      "cpu_usage_percent": 15.0,
      "memory_total_mb": 8192.0,
      "memory_available_mb": 4096.0,
      "disk_total_gb": 256.0,
      "disk_used_gb": 128.0,
      "disk_free_gb": 128.0
    }
  }
  ```

#### **/predict_from_file**

- **Método**: `POST`
- **Descrição**: Permite ao usuário enviar um arquivo CSV com dados históricos para gerar previsões personalizadas usando o modelo treinado.
- **Parâmetros**:
  - `ticker` (query string): Código da ação.
  - `file` (form-data): Arquivo CSV contendo os dados históricos.
- **Autenticação**: Necessária.
- **Exemplo de Requisição**:

  - **Usando cURL**:

    ```bash
    curl -X POST "http://localhost:8000/predict_from_file?ticker=AAPL"       -H "access_token: dead-beef-15-bad-f00d"       -F "file=@/caminho/para/seu/arquivo.csv"
    ```

- **Resposta**:

  ```json
  {
    "predictions": [150.25, 151.30, 152.10]
  }
  ```

### Observações Importantes

- **Formato do Arquivo CSV**: O arquivo enviado para `/predict_from_file` deve conter as colunas:

  ```
  Date,Open,High,Low,Close,Volume
  ```
  
## Monitoramento com Grafana e Prometheus

### Acessando o Dashboard do Grafana

Após executar a aplicação, o Grafana estará disponível no seguinte endereço:

- [http://localhost:3000](http://localhost:3000)

#### Login no Grafana

- **Usuário**: `admin`
- **Senha**: `zorzi`

#### Dashboards Disponíveis

O projeto inclui dashboards pré-configurados para monitorar:

- Métricas de desempenho da API.
- Utilização de recursos do sistema (CPU, memória, disco).

Os dashboards estão localizados na aba "Dashboards" após o login no Grafana.

---

## Executando os Testes

### Testes Unitários

O projeto inclui testes unitários para verificar a funcionalidade de cada componente.

1. **Executar os testes**:

   ```bash
   docker-compose exec app pytest
   ```

2. **Gerar relatório de cobertura**:

   ```bash
   docker-compose exec app pytest --cov=.
   ```

### Testes End-to-End

Os testes E2E verificam o comportamento da API em um ambiente realista.

1. **Executar os testes E2E**:

   ```bash
   docker-compose exec app pytest tests/e2e
   ```

---

## Notas Adicionais

- Certifique-se de que os dados históricos fornecidos para `/predict_from_file` estejam no formato correto.
- A API está configurada para funcionar com dados diários e não suporta previsões intradiárias.
- Utilize os dashboards no Grafana para monitorar a saúde do sistema e identificar possíveis gargalos.

---

Esperamos que este README seja útil para configurar e utilizar a API. Para dúvidas ou contribuições, sinta-se à vontade para abrir uma issue no repositório.