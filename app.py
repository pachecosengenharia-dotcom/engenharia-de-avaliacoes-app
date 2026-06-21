import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Engenharia de Avaliações")

st.write("Se o app carregar esta mensagem, o código está a funcionar.")

# Teste simples de leitura
try:
    st.write("Verificando ficheiros na pasta...")
    import os
    arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
    st.write(f"Ficheiros encontrados: {arquivos}")
except Exception as e:
    st.error(f"Erro ao ler pasta: {e}")
