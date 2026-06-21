import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📊 Engenharia de Avaliações - Goiânia")

# 1. Carregamento Robusto
try:
    # Lendo o arquivo diretamente
    df = pd.read_csv("Goiânia - GO.csv", sep=";")
    
    # Debug: Mostrar colunas encontradas para verificarmos o erro
    st.write("Colunas detectadas:", df.columns.tolist())
    
    # Limpeza de nomes
    df.columns = [c.strip() for c in df.columns]
    
    # Mapeamento dinâmico (ajuste conforme o print das colunas que aparecer na tela)
    # Se o nome no CSV for 'Área Privativa', ele usará esse nome.
    features = ['Área Privativa', 'Quartos', 'Suite']
    target = 'Valor Total'
    
    # Validação de existência de colunas
    if all(col in df.columns for col in features + [target]):
        
        # Limpeza de dados (conversão de moeda/texto para float)
        for col in features + [target]:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
        
        df = df.dropna()
        
        # Treino
        X = df[features]
        y = df[target]
        modelo = LinearRegression().fit(X, y)
        
        # Interface
        st.sidebar.header("Dados do Imóvel")
        area = st.sidebar.number_input("Área Privativa", value=float(df['Área Privativa'].mean()))
        quartos = st.sidebar.slider("Quartos", 1, 5, 2)
        suites = st.sidebar.slider("Suítes", 0, 5, 1)
        
        pred = modelo.predict([[area, quartos, suites]])
        st.metric("Valor de Mercado Estimado", f"R$ {pred[0]:,.2f}")
        
    else:
        st.error("As colunas não coincidem com o esperado. Verifique o nome das colunas no CSV.")

except Exception as e:
    st.error(f"Erro crítico: {e}")
