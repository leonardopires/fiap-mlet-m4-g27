#!/bin/bash

if [ "$DEBUG" = "true" ] ; then
    echo "Iniciando em modo de depuração"
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Iniciando em modo de produção"
    uvicorn api.main:app --host 0.0.0.0 --port 8000
fi