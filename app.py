import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura forçada
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    
    # 2. Limpeza brutal: transforma TUDO em numérico, o que for texto vira NaN
    df_clean = pd.DataFrame()
    col_alvo = 'Valor Unitário' # O alvo que você definiu

    for col in df.columns:
        # Remove caracteres indesejados, troca vírgula por ponto
        col_str = df[col].astype(str).str.replace('R$', '', regex=False).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        # Força numérico
        df_clean[col] = pd.to_numeric(col_str, errors='coerce')

    # 3. Remover linhas onde o alvo (Valor Unitário) ou os dados são nulos
    df_clean = df_clean.dropna(subset=[col_alvo])
    df_clean = df_clean.fillna(0) # Substitui o resto por 0 para não ter erro de None
    
    st.write(f"Linhas carregadas: {len(df_clean)}")

    # 4. Regressão
    y = df_clean[col_alvo]
    X = df_clean.drop(columns=[col_alvo])
    
    modelo = LinearRegression().fit(X, y)
    
    # 5. Interface
    st.sidebar.header("⚙️ Parâmetros")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(X[col].median()))
    
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Unitário Estimado", f"R$ {pred[0]:,.2f} / m²")

except Exception as e:
    st.error(f"Erro Crítico: {e}")
