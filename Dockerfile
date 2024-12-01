# Usar a imagem oficial do Python como base
FROM python:3.8-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar o requirements.txt para a imagem
COPY requirements.txt /app/requirements.txt

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o restante do código da aplicação (isso será sobrescrito pelo volume)
COPY api /app/api
COPY utils /app/utils

# Expor a porta (se necessário)
EXPOSE 8000

# Copiar o entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Definir o entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]