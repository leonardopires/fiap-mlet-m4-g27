# utils/security.py

import os
from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

# Configuração da chave de API
# Obtém a chave de API a partir de uma variável de ambiente.
# Isso é útil para manter a segurança, pois a chave não é exposta diretamente no código.
API_KEY = os.getenv("API_KEY")

# Nome do cabeçalho que será usado para enviar a chave de API nas requisições.
API_KEY_NAME = "access_token"

# Configura o cabeçalho de autenticação para a FastAPI
# `auto_error=False` significa que não será lançado um erro automaticamente
# caso a chave esteja ausente ou inválida. Isso será tratado manualmente.
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# Função para validar a chave de API
async def get_api_key(
        api_key_header: str = Security(api_key_header),
):
    """
    Verifica se a chave de API fornecida na requisição é válida.

    Parâmetros:
        api_key_header (str): A chave de API recebida no cabeçalho da requisição.

    Retorna:
        str: A chave de API validada, se for válida.

    Lança:
        HTTPException: Se a chave de API for inválida ou ausente.
    """
    # Compara a chave enviada pelo cliente com a chave armazenada
    if api_key_header == API_KEY:
        return api_key_header
    else:
        # Lança um erro HTTP 403 se a chave for inválida
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Chave de API inválida"  # Mensagem de erro retornada ao cliente
        )
