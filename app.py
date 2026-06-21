import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura do ficheiro com codificação correta
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Identificação automática de colunas numéricas
    # Vamos criar uma lista de colunas que conseguimos converter
    colunas_numericas = []
    for col in df.columns:
        # Tenta converter para número, se falhar vira NaN
        col_convertida = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        # Se a coluna tiver dados numéricos, guardamos
        if col_convertida.notna().sum() > 0:
            df[col] = col_convertida
            colunas_numericas.append(col)
    
    # 3. Filtrar apenas colunas numéricas e remover linhas vazias
    df_limpo = df[colunas_numericas].dropna()
    
    st.write("Colunas numéricas detectadas:", colunas_numericas)

    # 4. Seleção do alvo (assumindo que existe uma coluna com 'Valor' no nome)
    col_valor = [c for c in colunas_numericas if 'Valor' in c or 'Preco' in c][0]
    features = [c for c in colunas_numericas if c != col_valor]

    # 5. Regressão Linear
    X = df_limpo[features]
    y = df_limpo[col_valor]
    modelo = LinearRegression().fit(X, y)

    # 6. Interface (Dinâmica)
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in features:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_limpo[col].median()))

    # 7. Cálculo
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")

except Exception as e:
    st.error(f"Erro ao processar o CSV: {e}")
