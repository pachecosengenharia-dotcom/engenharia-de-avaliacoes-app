import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

# 1. Tentar ler o arquivo de forma robusta
try:
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Criar a variável df_modelo desde o início (evita NameError)
    df_modelo = pd.DataFrame()
    
    # Lista de colunas esperadas (ajuste se o nome no CSV for diferente)
    col_alvo = 'Valor Unitário'
    cols_input = ['Área Privativa', 'Quartos', 'Idade Aparente']
    
    # 3. Limpeza de dados (transforma texto em número, erros viram NaN)
    for col in [col_alvo] + cols_input:
        if col in df.columns:
            # Remove pontos de milhar, troca vírgula por ponto
            s = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_modelo[col] = pd.to_numeric(s, errors='coerce')
    
    df_modelo = df_modelo.dropna()

    # 4. Verificar se temos dados antes de treinar
    if not df_modelo.empty:
        # Treino
        X = df_modelo[cols_input]
        y = df_modelo[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        # 5. Interface
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = [st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median())) for col in cols_input]
        
        # Cálculo
        pred_unit = modelo.predict([inputs])[0]
        area_ref = inputs[0] # Área é a primeira input
        pred_total = pred_unit * area_ref
        
        # Exibição
        st.metric("Valor Unitário Estimado", f"R$ {pred_unit:,.2f} / m²")
        st.metric("Valor Total Estimado", f"R$ {pred_total:,.2f}")
    else:
        st.warning("Dados insuficientes ou colunas não encontradas. Verifique se o CSV contém 'Valor Unitário' e 'Área Privativa'.")

except Exception as e:
    st.error(f"Erro crítico: {e}")
