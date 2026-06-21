import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. SEPARAÇÃO INTELIGENTE: Só processamos colunas que podem ser números
    # Criamos um DataFrame apenas com colunas numéricas para o cálculo
    df_numerico = pd.DataFrame()
    for col in df.columns:
        # Tenta converter para número, o que for texto vira NaN
        col_convertida = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        if col_convertida.notna().sum() > 0: # Se tiver pelo menos um número
            df_numerico[col] = col_convertida

    df_numerico = df_numerico.dropna()
    
    # 3. Definição do alvo (assumindo que a coluna de valor contém 'Valor' ou 'Preco')
    col_valor = [c for c in df_numerico.columns if 'Valor' in c or 'Preco' in c][0]
    features = [c for c in df_numerico.columns if c != col_valor]
    
    st.write("Colunas numéricas processadas:", features)

    # 4. Regressão Linear
    X = df_numerico[features]
    y = df_numerico[col_valor]
    modelo = LinearRegression().fit(X, y)

    # 5. Interface
    st.sidebar.header("Dados do Imóvel")
    inputs = {}
    for col in features:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_numerico[col].median()))

    # 6. Cálculo
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")

except Exception as e:
    st.error(f"Erro no processamento: {e}")
    st.info("O sistema tentou processar apenas colunas numéricas, mas algo correu mal.")
