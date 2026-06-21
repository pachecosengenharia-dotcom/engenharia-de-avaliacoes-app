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

    # 2. Definição do alvo (O que você quer prever)
    col_alvo = 'Valor Unitário'
    
    # 3. Limpeza Seletiva: apenas colunas que conseguimos transformar em números
    df_modelo = pd.DataFrame()
    df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('R$', '').str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    # Adicionamos colunas que queremos como variáveis (Área, Quartos, etc.)
    features = ['Área Privativa', 'Quartos', 'Idade Aparente']
    for col in features:
        if col in df.columns:
            df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    df_modelo = df_modelo.dropna()
    
    st.write("Variáveis de entrada utilizadas:", features)
    st.write(f"Linhas válidas para o treino: {len(df_modelo)}")

    # 4. Regressão
    X = df_modelo[features]
    y = df_modelo[col_alvo]
    
    modelo = LinearRegression().fit(X, y)
    
    # 5. Interface
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in features:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median()))
    
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Unitário Estimado", f"R$ {pred[0]:,.2f} / m²")

except Exception as e:
    st.error(f"Erro: {e}")
    st.info("Dica: Verifique se as colunas 'Valor Unitário', 'Área Privativa' e 'Quartos' existem exatamente com estes nomes no seu arquivo.")
