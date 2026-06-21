import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Diagnóstico de Dados - Engenharia de Avaliações")

try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    
    st.write("### 1. Colunas Detectadas:", df.columns.tolist())
    
    # 2. Conversão manual e transparente
    # Vamos pegar apenas colunas que parecem ser numéricas
    df_clean = pd.DataFrame()
    for col in df.columns:
        # Tenta converter, vira NaN se não for número
        clean_col = pd.to_numeric(df[col].astype(str).str.replace('R$', '').str.replace('.', '').str.replace(',', '.'), errors='coerce')
        
        # Só incluímos se tivermos dados reais
        if clean_col.notna().sum() > 0:
            df_clean[col] = clean_col

    st.write("### 2. Colunas após limpeza numérica:", df_clean.columns.tolist())
    st.write("### 3. Total de linhas válidas:", len(df_clean.dropna()))
    
    if len(df_clean.dropna()) < 5:
        st.error("Ainda não restam linhas suficientes. Verifique o mapeamento das colunas abaixo.")
        st.dataframe(df.head(10)) # Mostra os primeiros dados para você ver a bagunça
    else:
        st.success("Dados prontos! O modelo agora pode calcular.")
        # ... (seu código de regressão aqui)

except Exception as e:
    st.error(f"Erro: {e}")
