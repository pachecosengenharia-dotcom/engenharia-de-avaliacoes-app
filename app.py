import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações (Universal)")

# --- Função PDF ---
def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 9)
    y = 770
    for i in range(0, len(eq_str), 100):
        c.drawString(50, y, eq_str[i:i+100]); y -= 15
    for k, v in inputs.items():
        c.drawString(60, y-10, f"- {k}: {v:.2f}"); y -= 12
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y-20, f"Resultados: V.U. Médio R$ {d['vu']:,.2f} | Total R$ {d['total']:,.2f}")
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 50, width=400, height=200)
    c.save(); buffer.seek(0)
    return buffer

# --- 1. CARREGAMENTO E SELEÇÃO DINÂMICA ---
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type="csv")
if arquivo:
    df = pd.read_csv(arquivo, sep=";", encoding='latin-1')
    
    # Usuário escolhe a coluna alvo e as variáveis
    colunas_disponiveis = df.columns.tolist()
    col_alvo = st.sidebar.selectbox("Selecione a coluna do Valor Unitário:", colunas_disponiveis, index=len(colunas_disponiveis)-1)
    features = st.sidebar.multiselect("Selecione as variáveis explicativas:", [c for c in colunas_disponiveis if c != col_alvo])
    
    # Limpeza Universal
    df_c = pd.DataFrame()
    for col in features + [col_alvo]:
        df_c[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
    df_c = df_c.dropna()

    if not df_c.empty and features:
        modelo = LinearRegression().fit(df_c[features], df_c[col_alvo])
        eq_str = f"{col_alvo} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
        
        # Exibição Técnica
        st.latex(eq_str)
        preds = modelo.predict(df_c[features])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.scatter(df_c[col_alvo], preds); ax1.set_title("Aderência")
        ax2.scatter(preds, df_c[col_alvo] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
        st.pyplot(fig)

        # Inputs Dinâmicos com Limites
        st.sidebar.header("⚙️ Parâmetros")
        inputs = {}
        for f in features:
            inputs[f] = st.sidebar.number_input(f"{f} ({df_c[f].min():.1f} a {df_c[f].max():.1f})", value=float(df_c[f].median()))
        
        if st.sidebar.button("Calcular Precificação"):
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            std = np.std(df_c[col_alvo] - preds)
            total = vu * (inputs.get('Área Privativa', 1)) # Tenta pegar área privativa, se não, 1
            
            c1, c2, c3 = st.columns(3)
            c1.metric("V.U. Mínimo", f"R$ {vu - (1.96*std):,.2f}")
            c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
            c3.metric("V.U. Máximo", f"R$ {vu + (1.96*std):,.2f}")
            st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
            
            pdf = gerar_laudo_pdf({'vu': vu, 'min': vu - (1.96*std), 'max': vu + (1.96*std), 'total': total}, fig, eq_str, inputs)
            st.download_button("📥 Baixar PDF", pdf, "laudo_tecnico.pdf")
