import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

# Carregamento robusto
try:
    df = pd.read_csv("Goiânia - GO.csv", sep=";")
    df.columns = [c.strip() for c in df.columns]
    
    # Seleção de colunas (Ajuste se o CSV tiver nomes ligeiramente diferentes)
    features = ['Área Privativa', 'Quartos', 'Suite']
    target = 'Valor Total'
    
    # Limpeza de dados (conversão segura)
    for col in features + [target]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    df = df.dropna()
    
    # Modelo matemático puro
    X = df[features]
    y = df[target]
    modelo = LinearRegression().fit(X, y)
    
    # Interface
    st.sidebar.header("Dados do Imóvel")
    area = st.sidebar.number_input("Área Privativa (m²)", value=float(df['Área Privativa'].mean()))
    quartos = st.sidebar.slider("Quartos", 1, 5, 2)
    suites = st.sidebar.slider("Suítes", 0, 5, 1)
    
    pred = modelo.predict([[area, quartos, suites]])
    st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")

except Exception as e:
    st.error(f"Erro no processamento: {e}")
