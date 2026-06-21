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
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "LAUDO TÉCNICO DE AVALIAÇÃO IMOBILIÁRIA")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Região: {regiao} | Amostras: {n_amostras}")
    c.drawString(50, 740, f"Equação: {eq_str}")
    c.drawString(50, 700, f"V.U. Mínimo: R$ {minimo:,.2f} | Médio: R$ {pred_unit:,.2f} | Máximo: R$ {maximo:,.2f}")
    c.drawString(50, 670, f"VALOR TOTAL ESTIMADO: R$ {pred_total:,.2f}")
    c.drawString(50, 640, "Variáveis: " + ", ".join(features))
    c.save()
    buffer.seek(0)
    return buffer

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        col_alvo = 'Valor Unitário'
        # Seleção dinâmica garantindo 'Setor Urbano' e removendo 'Idade Aparente'
        cols_base = [c for c in df.columns if c != col_alvo and c.lower() != 'idade aparente']
        features = list(set(cols_base + ['Setor Urbano'])) 
        features = [f for f in features if f in df.columns]

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
            pred_unit = modelo.predict([inputs])[0]
            
            # Métricas
            residuos = y - modelo.predict(X)
            erro_padrao = np.std(residuos)
            minimo, maximo = pred_unit - (1.96 * erro_padrao), pred_unit + (1.96 * erro_padrao)
            area = inputs[features.index('Área Construída')] if 'Área Construída' in features else 1
            
            col1, col2, col3 = st.columns(3)
            col1.metric("V.U. Mínimo", f"R$ {minimo:,.2f}")
            col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            col3.metric("V.U. Máximo", f"R$ {maximo:,.2f}")

            # Gráficos de Diagnóstico
            n, p = len(y), len(features) + 1
            X_mat = np.column_stack([np.ones(n), X])
            leverage = np.diag(X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T)
            cooks_dist = (residuos**2 / (p * np.var(residuos))) * (leverage / (1 - leverage)**2)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            ax1.scatter(y, modelo.predict(X), alpha=0.5); ax1.set_title("Aderência (Obs vs Prev)")
            ax2.scatter(modelo.predict(X), residuos, alpha=0.5, color='orange'); ax2.axhline(0, color='black', linestyle='--'); ax2.set_title("Resíduos")
            ax3.stem(cooks_dist); ax3.set_title("Distância de Cook")
            st.pyplot(fig)
            
            
            

            # PDF
            pdf_data = gerar_pdf(regiao, pred_unit, pred_unit * area, minimo, maximo, eq_str, features, len(df_modelo))
            st.download_button("📥 Baixar Laudo Completo", data=pdf_data, file_name="laudo_tecnico.pdf", mime="application/pdf")
        else:
            st.error("Dados insuficientes.")
    except Exception as e:
        st.error(f"Erro: {e}")
