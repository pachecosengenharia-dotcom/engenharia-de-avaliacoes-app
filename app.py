import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

# --- 1. FUNÇÃO PDF ---
def gerar_laudo_pdf(d):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Equação: {d['eq']}")
    c.drawString(50, 750, f"V.U. Médio Estimado: R$ {d['vu']:,.2f}")
    c.drawString(50, 730, f"Intervalo de Confiança (95%): R$ {d['min']:,.2f} a R$ {d['max']:,.2f}")
    c.drawString(50, 710, f"Valor Total Estimado: R$ {d['total']:,.2f}")
    c.save(); buf.seek(0)
    return buf

# --- 2. DADOS E MODELO ---
arquivo = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")

if arquivo:
    df = pd.read_csv(arquivo, sep=";", encoding='latin-1')
    features = ['Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 'Padrão de Acabamento', 
                'Suite', 'Estado de Conservação', 'Idade Aparente', 'Setor urbano', 'Data do Evento']
    df_c = pd.DataFrame()
    for col in features + ['Valor Unitário']:
        df_c[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    df_c = df_c.dropna()

    if not df_c.empty:
        X, y = df_c[features], df_c['Valor Unitário']
        modelo = LinearRegression().fit(X, y)
        
        # --- 3. EXIBIÇÃO: EQUAÇÃO E GRÁFICOS (Sempre visíveis) ---
        st.subheader("Equação do Modelo")
        st.latex(f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)]))
        
        preds = modelo.predict(X)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.scatter(y, preds); ax1.set_title("Aderência (Obs vs Prev)")
        ax2.scatter(preds, y-preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
        st.pyplot(fig)

        # --- 4. INPUTS COM VALIDAÇÃO NBR 14653 ---
        st.sidebar.header("⚙️ Parâmetros (Limites NBR)")
        inputs = {}
        extrapolou = False
        for f in features:
            min_v, max_v = df_c[f].min(), df_c[f].max()
            val = st.sidebar.number_input(f"{f} (Limites: {min_v:.1f} a {max_v:.1f})", value=float(df_c[f].median()))
            inputs[f] = val
            if val < min_v or val > max_v:
                st.sidebar.warning(f"⚠️ Extrapolação em {f} (Fora da amostra)!")
                extrapolou = True

        # --- 5. RESULTADOS ---
        if st.sidebar.button("Calcular Precificação"):
            if extrapolou:
                st.error("AVISO: O imóvel avaliando apresenta extrapolação de limites conforme NBR 14653. Valide a amostra.")
            
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            std = np.std(y - preds)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            total = vu * inputs['Área Privativa']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
            c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
            c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
            st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
            st.download_button("📥 Baixar Laudo PDF", gerar_laudo_pdf({'eq': '...', 'vu': vu, 'min': min_v, 'max': max_v, 'total': total}), "laudo.pdf")
