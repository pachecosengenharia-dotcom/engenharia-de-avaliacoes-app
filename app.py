import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura robusta com tratamento de acentos
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    
    # 2. Limpeza de colunas (retirar espaços extras)
    df.columns = [c.strip() for c in df.columns]
    
    # 3. Definição das colunas de trabalho
    # Verifique se os nomes abaixo coincidem exatamente com o seu ficheiro
    features = ['Área Privativa', 'Quartos', 'Suite']
    target = 'Valor Total'
    
    # 4. Limpeza de dados (Converter texto com vírgulas para números decimais)
    for col in features + [target]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('.', '').str.replace(',', '.').astype(float)
    
    df = df.dropna()
    
    # 5. Regressão Linear Pura
    X = df[features]
    y = df[target]
    modelo = LinearRegression().fit(X, y)
    
    # 6. Interface de input
    st.sidebar.header("Características do Imóvel")
    area = st.sidebar.number_input("Área Privativa (m²)", value=float(df['Área Privativa'].mean()))
    quartos = st.sidebar.slider("Quartos", 1, 5, 2)
    suites = st.sidebar.slider("Suítes", 0, 5, 1)
    
    # 7. Cálculo e Exibição
    pred = modelo.predict([[area, quartos, suites]])
    
    st.metric("Valor de Mercado Estimado", f"R$ {pred[0]:,.2f}")
    
    st.success("Modelo carregado com sucesso!")
    
except Exception as e:
    st.error(f"Erro no processamento: {e}")
    st.info("Dica: Verifique se o ficheiro 'Goiânia - GO.csv' está na pasta raiz do repositório.")
