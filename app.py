import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Modelo Final")

try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Definição Exata das Variáveis
    col_alvo = 'Valor Unitário'
    features = [
        'Área Construída', 'Área do Terreno', 'Evento', 
        'Padrão de Acabamento', 'Estado de Conservação', 
        'Setor urbano', 'Data do Evento', 'Quartos', 'Suite'
    ]

    # 3. Limpeza Robusta
    # Criamos um DataFrame apenas com colunas que existem no CSV
    df_modelo = pd.DataFrame()
    
    # Processar o alvo
    if col_alvo in df.columns:
        df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    # Processar as features
    for col in features:
        if col in df.columns:
            df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    df_modelo = df_modelo.dropna()
    
    st.write(f"Variáveis utilizadas: {list(df_modelo.columns)}")
    st.write(f"Linhas válidas para o treino: {len(df_modelo)}")

    # 4. Regressão
    X = df_modelo.drop(columns=[col_alvo])
    y = df_modelo[col_alvo]
    modelo = LinearRegression().fit(X, y)
    
    # 5. Interface
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median()))
    
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Unitário Estimado", f"R$ {pred[0]:,.2f} / m²")

except Exception as e:
    st.error(f"Erro: {e}")
    st.info("Verifique se os nomes das colunas no seu CSV são idênticos aos listados no código.")
