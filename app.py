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

    alvo = encontrar_coluna(['Valor Unitário', 'Valor Unitario'])
    area_construida = encontrar_coluna(['Área Construída', 'Area Construida', 'Área Privativa', 'Area Privativa'])
    
    features_base = [
        area_construida, 'Área do Terreno', 'Evento', 
        'Padrão de Acabamento', 'Estado de Conservação', 
        'Setor urbano', 'Data do Evento', 'Quartos', 'Suite'
    ]
    features = [f for f in features_base if f is not None]

    # 3. Limpeza
    df_modelo = pd.DataFrame()
    df_modelo[alvo] = pd.to_numeric(df[alvo].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    for col in features:
        if col in df.columns:
            df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    df_modelo = df_modelo.dropna()
    
    # 4. Regressão
    X = df_modelo.drop(columns=[alvo])
    y = df_modelo[alvo]
    modelo = LinearRegression().fit(X, y)
    
    # 5. Interface
    st.sidebar.header("⚙️ Parâmetros do Imóvel")
    inputs = {}
    for col in X.columns:
        inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median()))
    
    pred = modelo.predict([list(inputs.values())])
    st.metric("Valor Unitário Estimado", f"R$ {pred[0]:,.2f} / m²")
    
except Exception as e:
    st.error(f"Erro: {e}")
