import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Carregamento Seguro
    df = pd.read_csv("Goiânia - GO.csv", sep=";")
    df.columns = [c.strip() for c in df.columns]
    
    st.write("Colunas encontradas no arquivo:", df.columns.tolist()) # Debug na tela
    
    # 2. Seleção de variáveis
    # Ajuste estes nomes se os que apareceram acima forem diferentes
    features = ['Área Privativa', 'Quartos', 'Suite']
    target = 'Valor Total'
    
    # 3. Limpeza de dados (conversão de vírgula para ponto)
    for col in features + [target]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    df = df.dropna()
    
    # 4. Cálculo do Modelo
    X = df[features]
    y = df[target]
    modelo = LinearRegression().fit(X, y)
    
    # 5. Interface
    st.sidebar.header("Dados do Imóvel")
    area = st.sidebar.number_input("Área Privativa (m²)", value=float(df['Área Privativa'].mean()))
    quartos = st.sidebar.slider("Quartos", 1, 5, 2)
    suites = st.sidebar.slider("Suítes", 0, 5, 1)
    
    pred = modelo.predict([[area, quartos, suites]])
    st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")

except Exception as e:
    st.error(f"Erro identificado pelo sistema: {e}")
    st.info("Verifique se o nome do arquivo no GitHub é exatamente 'Goiânia - GO.csv'")
