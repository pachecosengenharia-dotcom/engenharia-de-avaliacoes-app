import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

# --- LISTA DE COLUNAS QUE QUERES NOS PARÂMETROS ---
# Adiciona aqui apenas o nome das colunas que queres que apareçam no input
COLUNAS_RELEVANTES = ['Área Privativa', 'Quartos', 'Suítes', 'Vagas', 'Idade Aparente']

try:
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # Função de limpeza
    def limpar_num(val):
        s = str(val).replace('R$', '').replace('.', '').replace(',', '.')
        try: return float(s)
        except: return np.nan

    # Processamento dos dados
    df_clean = pd.DataFrame()
    for col in df.columns:
        df_clean[col] = df[col].apply(limpar_num)
    df_clean = df_clean.dropna()

    # Definir alvo e features
    col_alvo = 'Valor Unitário'
    # Filtra as features: só pegamos o que estiver na nossa lista e existir no CSV
    features = [c for c in COLUNAS_RELEVANTES if c in df_clean.columns]
    
    st.write("Variáveis de entrada utilizadas:", features)

    # Regressão
    X = df_clean[features]
    y = df_clean[col_alvo]
    modelo = LinearRegression().fit(X, y)

    # Interface - APENAS COLUNAS RELEVANTES
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in features:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_clean[col].median()))

    # Predição
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Unitário Estimado", f"R$ {pred[0]:,.2f} / m²")

except Exception as e:
    st.error(f"Erro: {e}")
