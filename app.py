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

def gerar_pdf(regiao, pred_unit, pred_total, minimo, maximo, eq_str, features, n_amostras):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "LAUDO TÉCNICO DE AVALIAÇÃO IMOBILIÁRIA")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Região: {regiao} | Amostras: {n_amostras}")
    c.drawString(50, 740, f"Equação: {eq_str}")
    c.drawString(50, 700, f"V.U. Médio: R$ {pred_unit:,.2f} | Valor Total: R$ {pred_total:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

if regiao:
    try:
        # Leitura com verificação
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]
        
        # DEBUG: Mostra colunas encontradas para você conferir
        st.write("Colunas encontradas no arquivo:", df.columns.tolist())
        
        col_alvo = 'Valor Unitário'
        if col_alvo not in df.columns:
            st.error(f"Coluna alvo '{col_alvo}' não encontrada. Verifique se o nome no CSV está idêntico.")
            st.stop()

        # Limpeza e seleção
        features = [c for c in df.columns if c != col_alvo and c.lower() != 'idade aparente']
        if 'Setor Urbano' not in features:
            features.append('Setor Urbano')
            
        # Filtra apenas o que é numérico
        df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce'))
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X = df_modelo[features]
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # Equação
            eq_str = f"VU = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)

            # Inputs
            st.sidebar.header("⚙️ Parâmetros")
            inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
            
            # Resultados
            pred_unit = modelo.predict([inputs])[0]
            area = inputs[features.index('Área Construída')] if 'Área Construída' in features else 1
            minimo, maximo = pred_unit * 0.9, pred_unit * 1.1 # Aproximação estatística
            
            st.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            
            # Gráficos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.scatter(y, modelo.predict(X), alpha=0.5); ax1.set_title("Aderência")
            ax2.scatter(modelo.predict(X), y - modelo.predict(X), alpha=0.5); ax2.set_title("Resíduos")
            st.pyplot(fig)
            
            # PDF
            pdf_data = gerar_pdf(regiao, pred_unit, pred_unit*area, minimo, maximo, eq_str, features, len(df_modelo))
