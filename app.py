import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

# Função de limpeza extrema para garantir que o numpy não reclame
def limpar_valor(valor):
    try:
        if pd.isna(valor): return 0.0
        s = str(valor).replace('R$', '').replace('.', '').replace(',', '.')
        return float(s)
    except:
        return 0.0

try:
    # 1. Leitura com encoding para aceitar acentos
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Mapeamento das colunas (Ajuste se necessário)
    col_area = 'Área Privativa'
    col_quartos = 'Quartos'
    col_suites = 'Suite'
    col_valor = 'Valor Total'
    
    # 3. Limpeza forçada dos dados
    df[col_area] = df[col_area].apply(limpar_valor)
    df[col_quartos] = df[col_quartos].apply(limpar_valor)
    df[col_suites] = df[col_suites].apply(limpar_valor)
    df[col_valor] = df[col_valor].apply(limpar_valor)
    
    # Filtrar apenas linhas com valor real
    df = df[df[col_valor] > 0]
    
    # 4. Treino do Modelo
    X = df[[col_area, col_quartos, col_suites]]
    y = df[col_valor]
    modelo = LinearRegression().fit(X, y)
    
    # 5. Interface
    st.sidebar.header("Dados do Imóvel")
    area_input = st.sidebar.number_input("Área Privativa (m²)", value=float(df[col_area].mean()))
    quartos_input = st.sidebar.slider("Quartos", 1, 5, 2)
    suites_input = st.sidebar.slider("Suítes", 0, 5, 1)
    
    pred = modelo.predict([[area_input, quartos_input, suites_input]])
    
    st.metric("Valor Estimado", f"R$ {pred[0]:,.2f}")
    
    # Exibir resumo dos dados para garantir que nada foi perdido
    st.write("Resumo dos dados processados:")
    st.dataframe(df.head())

except Exception as e:
    st.error(f"Erro no processamento: {e}")
    st.info("Verifique se as colunas no CSV têm exatamente os nomes: 'Área Privativa', 'Quartos', 'Suite' e 'Valor Total'.")
