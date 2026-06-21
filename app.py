import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Mapeamento Inteligente
    def encontrar_coluna(lista_possiveis):
        for nome in lista_possiveis:
            if nome in df.columns: return nome
        return None

    alvo_unit = encontrar_coluna(['Valor Unitário', 'Valor Unitario'])
    area = encontrar_coluna(['Área Construída', 'Area Construida', 'Área Privativa', 'Area Privativa'])
    
    features_base = [
        area, 'Área do Terreno', 'Evento', 
        'Padrão de Acabamento', 'Estado de Conservação', 
        'Setor urbano', 'Data do Evento', 'Quartos', 'Suite'
    ]
    features = [f for f in features_base if f is not None]

    # 3. Limpeza
    df_modelo = pd.DataFrame()
    df_modelo[alvo_unit] = pd.to_numeric(df[alvo_unit].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    df_modelo[area] = pd.to_numeric(df[area].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    for col in features:
        if col in df.columns and col != area:
            df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    df_modelo = df_modelo.dropna()
    
    # 4. Regressão
    X = df_modelo.drop(columns=[alvo_unit])
    y = df_modelo[alvo_unit]
    modelo = LinearRegression().fit(X, y)
    
    # 5. Interface
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median()))
    
    # 6. Cálculo e Exibição
    pred_unit = modelo.predict([list(inputs.values())])[0]
    pred_total = pred_unit * inputs[area] # Cálculo do Valor Total baseado na área
    
    col1, col2 = st.columns(2)
    col1.metric("Valor Unitário Estimado", f"R$ {pred_unit:,.2f} / m²")
    col2.metric("Valor Total Estimado", f"R$ {pred_total:,.2f}")
    
except Exception as e:
    st.error(f"Erro: {e}")
