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



def gerar_pdf(regiao, pred_unit, pred_total, minimo, maximo, eq_str, features, n_amostras):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Cabeçalho Formal
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 820, "LAUDO TÉCNICO DE AVALIAÇÃO IMOBILIÁRIA")
    c.line(50, 810, 550, 810)
    
    # Dados da Análise
    c.setFont("Helvetica", 12)
    c.drawString(50, 780, f"Região Analisada: {regiao}")
    c.drawString(50, 765, f"Tamanho da Amostra: {n_amostras} imóveis")
    
    # Equação
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 735, "Equação de Regressão:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 720, eq_str)
    
    # Resultados
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 680, "RESULTADOS DA AVALIAÇÃO:")
    c.setFont("Helvetica", 12)
    c.drawString(50, 660, f"Valor Unitário Médio: R$ {pred_unit:,.2f} / m²")
    c.drawString(50, 645, f"Intervalo de Confiança: R$ {minimo:,.2f} a R$ {maximo:,.2f}")
    c.drawString(50, 625, f"VALOR TOTAL ESTIMADO: R$ {pred_total:,.2f}")
    
    # Variáveis
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 590, "Variáveis Consideradas:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 575, ", ".join(features))
    
    c.save()
    buffer.seek(0)
    return buffer


if regiao:

    try:

        df = pd.read_csv(regiao, sep=";", encoding='latin-1')

        df.columns = [c.strip() for c in df.columns]



        col_alvo = 'Valor Unitário'

        features = [c for c in df.columns if c != col_alvo and pd.to_numeric(df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce').notna().sum() > len(df)*0.5]



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

            area = inputs[features.index(next((c for c in features if 'área' in c.lower()), features[0]))]

            

            col1, col2, col3 = st.columns(3)

            col1.metric("V.U. Mínimo", f"R$ {minimo:,.2f}")

            col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")

            col3.metric("V.U. Máximo", f"R$ {maximo:,.2f}")

            st.metric("Valor Total (Médio)", f"R$ {pred_unit * area:,.2f}")



            # Diagnóstico estatístico

            n, p = len(y), len(features) + 1

            X_mat = np.column_stack([np.ones(n), X])

            leverage = np.diag(X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T)

            cooks_dist = (residuos**2 / (p * np.var(residuos))) * (leverage / (1 - leverage)**2)

            

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            ax1.scatter(y, modelo.predict(X), alpha=0.5); ax1.set_title("Aderência (Obs vs Prev)")

            ax2.scatter(modelo.predict(X), residuos, alpha=0.5, color='orange'); ax2.axhline(0, color='black', linestyle='--'); ax2.set_title("Resíduos")

            ax3.stem(cooks_dist); ax3.set_title("Distância de Cook (Influência)")

            st.pyplot(fig)



            # PDF com dados completos
            pdf_data = gerar_pdf(regiao, pred_unit, pred_unit * area, minimo, maximo, eq_str, features, len(df_modelo))
            st.download_button(
                "📥 Baixar Laudo Completo em PDF", 
                data=pdf_data, 
                file_name=f"Laudo_{regiao.replace('.csv', '')}.pdf", 
                mime="application/pdf"
            )

        else:

            st.error("Dados insuficientes.")

    except Exception as e:

        st.error(f"Erro: {e}") 

