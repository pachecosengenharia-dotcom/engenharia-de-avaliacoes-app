import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Configuração ---
st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Laudo Técnico")

# --- Função PDF ---
def gerar_pdf(pred_unit, pred_total, minimo, maximo, eq_str):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Equação: {eq_str}")
    c.drawString(50, 740, f"V.U. Médio: R$ {pred_unit:,.2f}")
    c.drawString(50, 710, f"Valor Total: R$ {pred_total:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

# --- Fluxo Principal ---
arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        # 1. Carregamento com tratamento de separador e encoding
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        # 2. Seleção de colunas numéricas de forma robusta
        col_alvo = 'Valor Unitário'
        # Convertemos todas as colunas para numérico, forçando erro para o que não for número (NaN)
        df_num = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce'))
        
        # Removemos colunas que ficaram vazias após a conversão
        df_num = df_num.dropna(axis=1, how='all')
        # Removemos linhas que contêm NaN
        df_num = df_num.dropna(subset=[col_alvo])
        
        features = [c for c in df_num.columns if c != col_alvo]

        if not df_num.empty:
            X = df_num[features].fillna(0) # Preenche eventuais nulos restantes com 0
            y = df_num[col_alvo]
            
            modelo = LinearRegression().fit(X, y)
            
            # Equação para o Laudo
            eq_str = f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f} * {n})" for n, c in zip(features, modelo.coef_)])
            st.subheader("Equação do Modelo")
            st.latex(eq_str)

            # ... (seu código de inputs e métricas continua aqui) ...
            
        else:
            st.error("Dados insuficientes ou inválidos após limpeza.")
    except Exception as e:
        st.error(f"Erro técnico: {e}")
