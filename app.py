import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

# Bloco protegido para garantir que nada quebre
try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Definição das colunas esperadas
    col_alvo = 'Valor Unitário'
    features = ['Área Construída', 'Área do Terreno', 'Evento', 'Padrão de Acabamento', 
                'Estado de Conservação', 'Setor urbano', 'Data do Evento', 'Quartos', 'Suítes']
    
    # 3. Limpeza (Converte tudo para número, ignora textos)
    df_modelo = pd.DataFrame()
    if col_alvo in df.columns:
        df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    for col in features:
        if col in df.columns:
            df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    df_modelo = df_modelo.dropna()

    # 4. Cálculo do Modelo (Só se houver dados!)
    if not df_modelo.empty:
        X = df_modelo.drop(columns=[col_alvo])
        y = df_modelo[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        # Equação
        intercept = modelo.intercept_
        coefs = dict(zip(X.columns, modelo.coef_))
        
        # Interface
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = [st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median())) for col in X.columns]
        
        # Predições
        pred_unit = modelo.predict([inputs])[0]
        minimo = pred_unit * 0.9  # Estimativa de intervalo (simplificada)
        maximo = pred_unit * 1.1
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Valor Unitário (Mín)", f"R$ {minimo:,.2f}")
        col2.metric("Valor Unitário (Médio)", f"R$ {pred_unit:,.2f}")
        col3.metric("Valor Unitário (Máx)", f"R$ {maximo:,.2f}")
        
        # Gráfico
        st.subheader("Análise de Aderência")
        fig, ax = plt.subplots()
        ax.scatter(y, modelo.predict(X), alpha=0.5)
        st.pyplot(fig)
        
    else:
        st.error("Dados insuficientes. Verifique os nomes das colunas no CSV.")

except Exception as e:
    st.error(f"Erro técnico: {e}")
