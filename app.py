import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📊 Engenharia de Avaliações - Goiânia")

# Carregar o seu arquivo de mercado
try:
    # Usamos o separador ';' que é o padrão do Excel brasileiro
    df = pd.read_csv("Goiânia - GO.csv", sep=";")
    
    # Limpeza básica dos nomes das colunas (para evitar erros com espaços)
    df.columns = [col.strip() for col in df.columns]
    
    # Exibir colunas encontradas para debug na tela
    st.write("Colunas encontradas:", df.columns.tolist())
    
    # Seleção de variáveis (ajuste os nomes conforme as colunas que aparecerem acima)
    features = ['Área Privativa', 'Quartos', 'Suite']
    target = 'Valor Total'
    
    # Converter colunas para numérico, tratando erros
    for col in features + [target]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            
    df = df.dropna()

    # Criar e treinar modelo
    X = df[features]
    y = df[target]
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Entradas do usuário
    st.sidebar.header("Dados do Imóvel")
    area = st.sidebar.number_input("Área Privativa (m²)", value=float(df['Área Privativa'].mean()))
    quartos = st.sidebar.slider("Quartos", 1, 5, 2)
    suites = st.sidebar.slider("Suítes", 0, 5, 1)

    # Predição
    valor_estimado = modelo.predict([[area, quartos, suites]])
    st.metric("Valor Estimado", f"R$ {valor_estimado[0]:,.2f}")

except Exception as e:
    st.error(f"Erro ao processar arquivo: {e}")
