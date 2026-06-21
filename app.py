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

    # 2. Definição das colunas
    col_alvo = 'Valor Unitário'
    features = ['Área Construída', 'Área do Terreno', 'Evento', 'Padrão de Acabamento', 
                'Estado de Conservação', 'Setor urbano', 'Data do Evento', 'Quartos', 'Suítes']
    
    # 3. Limpeza
    df_modelo = pd.DataFrame()
    if col_alvo in df.columns:
        df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    for col in features:
        if col in df.columns:
            df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    df_modelo = df_modelo.dropna()

    if not df_modelo.empty:
        # 4. Regressão
        X = df_modelo.drop(columns=[col_alvo])
        y = df_modelo[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        # 5. Extração da Equação
        intercepto = modelo.intercept_
        coefs = dict(zip(X.columns, modelo.coef_))
        
        # Exibição da Equação
        st.subheader("Equação do Modelo")
        eq = f"V.U. = {intercepto:.2f} "
        for col, coef in coefs.items():
            eq += f"+ ({coef:.2f} * {col}) "
        st.latex(eq)
        
        # 6. Interface e Cálculo de Valor Total
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = [st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median())) for col in X.columns]
        
        # Predições
        pred_unit = modelo.predict([inputs])[0]
        # Multiplicamos pela 'Área Construída' para o Valor Total
        area_idx = X.columns.get_loc('Área Construída') if 'Área Construída' in X.columns else 0
        pred_total = pred_unit * inputs[area_idx]
        
        # Exibição Final
        col1, col2 = st.columns(2)
        col1.metric("Valor Unitário Estimado", f"R$ {pred_unit:,.2f} / m²")
        col2.metric("Valor Total Estimado", f"R$ {pred_total:,.2f}")
    else:
        st.error("Dados insuficientes.")

except Exception as e:
    st.error(f"Erro ao processar: {e}")
