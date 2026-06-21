import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📊 Engenharia de Avaliações - Goiânia")

# 1. Carregamento Direto
try:
    df = pd.read_csv("Goiânia - GO.csv", sep=";")
    df.columns = [c.strip() for c in df.columns]
    
    # Validação de colunas
    cols_necessarias = ['Área Privativa', 'Quartos', 'Suite', 'Valor Total']
    if all(c in df.columns for c in cols_necessarias):
        
        # Limpeza
        for col in cols_necessarias:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        df = df.dropna()
        
        # Modelo Puro
        X = df[['Área Privativa', 'Quartos', 'Suite']]
        y = df['Valor Total']
        modelo = LinearRegression().fit(X, y)
        
        # UI
        area = st.sidebar.number_input("Área (m²)", value=float(df['Área Privativa'].mean()))
        quartos = st.sidebar.slider("Quartos", 1, 5, 2)
        suites = st.sidebar.slider("Suítes", 0, 5, 1)
        
        pred = modelo.predict([[area, quartos, suites]])
        st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")
        
    else:
        st.error(f"Colunas não encontradas. Temos apenas: {df.columns.tolist()}")

except Exception as e:
    st.error(f"Erro no processamento: {e}")
