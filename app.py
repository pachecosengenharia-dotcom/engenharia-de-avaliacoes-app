import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Modelo Final")

try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Mapeamento Inteligente: Procuramos os nomes corretos no seu arquivo
    def encontrar_coluna(lista_possiveis):
        for nome in lista_possiveis:
            if nome in df.columns: return nome
        return None

    # Mapear colunas críticas (se o seu CSV tiver nomes diferentes, adicione aqui)
    alvo = encontrar_coluna(['Valor Unitário', 'Valor Unitario'])
    area_construida = encontrar_coluna(['Área Construída', 'Area Construida', 'Área Privativa', 'Area Privativa'])
    
    # Lista completa de features (colunas de entrada)
    features_base = [
        area_construida, 'Área do Terreno', 'Evento', 
        'Padrão de Acabamento', 'Estado de Conservação', 
        'Setor urbano', 'Data do Evento', 'Quartos', 'Suite'
    ]
    features = [f for f in features_base if f is not None]

    # 3. Limpeza e Seleção
    df_modelo = pd.DataFrame()
    df_modelo[alvo] = pd.to_numeric(df[alvo].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    for col in features:
        df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    df_modelo = df_modelo.dropna()
    
    st.write(f"Variável principal detectada (Área): **{area_construida}**")
    st.write(f"Linhas válidas para o treino: **{len(df_modelo)}**")

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
    
    # Gráfico de diagnóstico para entender a relação entre área e valor
    st.subheader("Análise de Aderência")
    [Image of scatter plot showing relationship between built area and unit value]
    
except Exception as e:
    st.error(f"Erro: {e}")
    st.info("O modelo não encontrou a coluna de Área Construída. Verifique se o nome no seu CSV é exatamente igual a um dos listados no código.")
