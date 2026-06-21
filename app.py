import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Modelo Dinâmico")

# 1. Seletor de região
arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        # 2. DETECÇÃO DINÂMICA
        # Procura por colunas que contenham 'Valor' (unidade ou total)
        col_alvo = next((c for c in df.columns if 'valor' in c.lower()), df.columns[-1])
        
        # Seleciona todas as outras colunas numéricas como variáveis de entrada
        features = []
        for col in df.columns:
            if col != col_alvo:
                # Tenta converter para numérico
                col_num = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
                # Se for maior que 50% numérica, consideramos feature
                if col_num.notna().sum() > (len(df) * 0.5):
                    features.append(col)

        st.info(f"Modelo detectou automaticamente:")
        st.write(f"- Alvo: {col_alvo}")
        st.write(f"- Variáveis de entrada: {features}")

        # 3. Processamento e Regressão
        df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce'))
        df_modelo = df_modelo.dropna()

        # ... (segue o cálculo do modelo, interface e gráficos como antes) ...
        
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
