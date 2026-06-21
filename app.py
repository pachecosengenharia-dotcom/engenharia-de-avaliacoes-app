import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Laudo Técnico")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

def gerar_pdf(regiao, pred_unit, minimo, maximo, eq_str, features, n_amostras):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "LAUDO TÉCNICO DE AVALIAÇÃO")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Região: {regiao} | Amostras: {n_amostras}")
    c.drawString(50, 740, f"Equação: {eq_str}")
    c.drawString(50, 700, f"V.U. Médio: R$ {pred_unit:,.2f}")
    c.drawString(50, 685, f"Intervalo: R$ {minimo:,.2f} a R$ {maximo:,.2f}")
    c.drawString(50, 650, "Variáveis: " + ", ".join(features))
    c.save()
    buffer.seek(0)
    return buffer

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = df.columns.str.strip()
        
        # O nome da sua coluna de preço (verificado na imagem debug)
        col_alvo = "Valor Unitário"
        
        # Filtro de variáveis
        excluir = ['Idade Aparente', 'Endereço', 'Complemento', 'Bairro', 'Informante', 'Telefone', 'Data do Evento']
        features = [c for c in df.columns if c != col_alvo and c not in excluir]
        if 'Setor Urbano' not in features: features.append('Setor Urbano')
        
        # Limpeza de dados
        df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.','').str.replace(',','.'), errors='coerce'))
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X, y = df_modelo[features], df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # Equação para o Laudo
            eq_str = f"VU = {modelo.intercept_:.2f}"
            st.latex(eq_str)

            # Inputs na Sidebar
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
            pred_unit = modelo.predict([inputs])[0]
            
            # Métricas
            residuos = y - modelo.predict(X)
            minimo, maximo = pred_unit * 0.90, pred_unit * 1.10
            
            c1, c2, c3 = st.columns(3)
            c1.metric("V.U. Mínimo", f"R$ {minimo:,.2f}")
            c2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            c3.metric("V.U. Máximo", f"R$ {maximo:,.2
