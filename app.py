import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

try:
    # 1. Leitura do arquivo
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Mapeamento Automático
    def encontrar_coluna(lista_possiveis):
        for nome in lista_possiveis:
            if nome in df.columns: return nome
        return None

    col_alvo = encontrar_coluna(['Valor Unitário', 'Valor Unitario'])
    # Adicionamos todas as colunas possíveis de entrada
    colunas_entrada = [
        'Área Construída', 'Area Construida', 'Área Privativa', 'Area Privativa',
        'Área do Terreno', 'Area do Terreno', 'Evento', 'Padrão de Acabamento', 
        'Estado de Conservação', 'Setor urbano', 'Data do Evento', 'Quartos', 'Suite'
    ]
    
    # 3. Processamento Limpo
    df_modelo = pd.DataFrame()
    if col_alvo:
        df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    features = []
    for col in colunas_entrada:
        if col in df.columns and col != col_alvo:
            df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
            features.append(col)
    
    df_modelo = df_modelo.dropna()

    if len(df_modelo) > 0:
        # 4. Regressão
        X = df_modelo[features]
        y = df_modelo[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        # 5. Interface
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = {}
        for col in features:
            inputs[col] = st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median()))
        
        pred_unit = modelo.predict([list(inputs.values())])[0]
        area_ref = inputs.get('Área Construída') or inputs.get('Area Construida') or inputs.get('Área Privativa') or inputs.get('Area Privativa') or 1
        pred_total = pred_unit * area_ref
        
        col1, col2 = st.columns(2)
        col1.metric("Valor Unitário Estimado", f"R$ {pred_unit:,.2f} / m²")
        col2.metric("Valor Total Estimado", f"R$ {pred_total:,.2f}")
        
        st.success("Modelo treinado com sucesso!")
        st.dataframe(df_modelo)
    else:
        st.error("Não foram encontrados dados válidos. Verifique se o seu CSV possui as colunas necessárias.")

except Exception as e:
    st.error(f"Erro: {e}")
