import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Identificar e Limpar colunas
    # Transformamos tudo o que for numérico e descartamos o que for texto puro (endereços)
    df_limpo = pd.DataFrame()
    col_alvo = 'Valor Total'
    
    for col in df.columns:
        # Converter para numérico, tratando vírgulas e pontos
        col_convertida = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        
        # Manter se for o alvo ou se tiver dados numéricos válidos
        if col == col_alvo or col_convertida.notna().sum() > (len(df) * 0.8):
            df_limpo[col] = col_convertida

    df_limpo = df_limpo.dropna()
    
    st.write("Variáveis independentes detectadas:", [c for c in df_limpo.columns if c != col_alvo])
    st.write("Linhas válidas para o treino:", len(df_limpo))

    # 3. Regressão (Separando o Alvo)
    X = df_limpo.drop(columns=[col_alvo])
    y = df_limpo[col_alvo]
    
    modelo = LinearRegression().fit(X, y)
    
    # 4. Interface
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_limpo[col].median()))

    # 5. Predição
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Total Estimado", f"R$ {pred[0]:,.2f}")

except Exception as e:
    st.error(f"Erro na execução: {e}")
