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
st.title("📊 Engenharia de Avaliações - Laudo Técnico Completo")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

def gerar_pdf(pred_unit, pred_total, minimo, maximo, eq_str):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Equação: {eq_str}")
    c.drawString(50, 740, f"V.U. Mínimo: R$ {minimo:,.2f} | Médio: R$ {pred_unit:,.2f} | Máximo: R$ {maximo:,.2f}")
    c.drawString(50, 710, f"Valor Total Estimado: R$ {pred_total:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        col_alvo = 'Valor Unitário'
        # Ajuste: Excluímos explicitamente 'Valor Total' da lista de variáveis de entrada (features)
        excluir_variaveis = [col_alvo, 'Valor Total', 'Valor Total Estimado']
        features = [c for c in df.columns if c not in excluir_variaveis and pd.to_numeric(df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce').notna().sum() > len(df)*0.5]

        df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce'))
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X = df_modelo[features]
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # Equação
            intercept = modelo.intercept_
            eq_str = f"V.U. = {intercept:.2f} " + " ".join([f"+ ({c:.2f} * {n})" for n, c in zip(features, modelo.coef_)])
            st.subheader("Equação do Modelo")
            st.latex(eq_str)

            # Cálculos
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
            pred_unit = modelo.predict([inputs])[0]
            residuos = y - modelo.predict(X)
            erro_padrao = np.std(residuos)
            minimo, maximo = pred_unit - (1.96 * erro_padrao), pred_unit + (1.96 * erro_padrao)
            
            # Cálculo de área apenas para exibição do Valor Total, não como variável de entrada
            col_area = next((c for c in features if 'área' in c.lower()), features[0])
            area = inputs[features.index(col_area)] if col_area in features else 1
            
            col1, col2, col3 = st.columns(3)
            col1.metric("V.U. Mínimo", f"R$ {minimo:,.2f}")
            col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            col3.metric("V.U. Máximo", f"R$ {maximo:,.2f}")
            st.metric("Valor Total (Médio)", f"R$ {pred_unit *
