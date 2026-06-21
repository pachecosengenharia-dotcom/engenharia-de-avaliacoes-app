import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io

# Configuração Base
st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

# Carregar arquivo (usando o seu CSV de Goiânia)
try:
    df = pd.read_csv("Goiânia - GO.csv", sep=";")
    # Limpeza básica de nomes de colunas
    df.columns = [c.strip() for c in df.columns]
    
    # Seleção de colunas para o modelo
    # Área Privativa, Quartos, Suítes, Valor Total
    features = ['Área Privativa', 'Quartos', 'Suite']
    target = 'Valor Total'
    
    # Limpeza de dados (remover sujeira de texto)
    for col in features + [target]:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    df = df.dropna()

    # Treino do Modelo (Puro)
    X = df[features]
    y = df[target]
    
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Inputs do usuário
    st.sidebar.header("Dados do Imóvel")
    area = st.sidebar.number_input("Área Privativa (m²)", value=float(df['Área Privativa'].mean()))
    quartos = st.sidebar.slider("Quartos", 1, 5, 3)
    suites = st.sidebar.slider("Suítes", 0, 5, 1)
    
    # Predição
    pred = modelo.predict([[area, quartos, suites]])
    
    st.metric("Valor de Mercado Estimado", f"R$ {pred[0]:,.2f}")
    
    # Gráfico simples
    fig, ax = plt.subplots()
    ax.scatter(df['Área Privativa'], df[target], color='blue', alpha=0.5)
    ax.set_xlabel("Área")
    ax.set_ylabel("Valor")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erro ao processar: {e}")
    st.info("Verifique se o arquivo 'Goiânia - GO.csv' está na mesma pasta do app.py")
