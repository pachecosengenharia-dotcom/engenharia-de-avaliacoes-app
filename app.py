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
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

# --- 1. FUNÇÃO DE PDF ---
def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 9)
    # Equação
    c.drawString(50, 770, "Equação:")
    for i, line in enumerate([eq_str[i:i+100] for i in range(0, len(eq_str), 100)]):
        c.drawString(50, 755 - (i*12), line)
    # Parâmetros
    y = 700
    c.drawString(50, y, "Parâmetros Utilizados:")
    for k, v in inputs.items():
        c.drawString(60, y-15, f"- {k}: {v:.2f}")
        y -= 12
    # Resultados
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y-20, f"Resultados: V.U. Médio R$ {d['vu']:,.2f} | Total R$ {d['total']:,.2f}")
    c.drawString(50, y-35, f"Intervalo (95%): R$ {d['min']:,.2f} a R$ {d['max']:,.2f}")
    # Imagem
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 50, width=400, height=200)
    c.save(); buffer.seek(0)
    return buffer

# --- 2. DADOS ---
arquivo = st.sidebar.file_uploader("Carregar CSV", type="csv")
if arquivo:
    df = pd.read_csv(arquivo, sep=";", encoding='latin-1')
    feats = ['Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 'Padrão de Acabamento', 
             'Suite', 'Estado de Conservação', 'Idade Aparente', 'Setor urbano', 'Data do Evento']
    df_c = pd.DataFrame()
    for col in feats + ['Valor Unitário']:
        df_c[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
    df_c = df_c.dropna()

    if not df_c.empty:
        modelo = LinearRegression().fit(df_c[feats], df_c['Valor Unitário'])
        preds = modelo.predict(df_c[feats])
        
        # --- 3. EQUAÇÃO E GRÁFICOS ---
        eq_str = f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(feats, modelo.coef_)])
        st.latex(eq_str)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.scatter(df_c['Valor Unitário'], preds); ax1.set_title("Aderência")
        ax2.scatter(preds, df_c['Valor Unitário'] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
        st.pyplot(fig)

        # --- 4. INPUTS E CÁLCULO ---
        st.sidebar.header("⚙️ Parâmetros (Limites NBR)")
        inputs = {}
        for f in feats:
            min_v, max_v = df_c[f].min(), df_c[f].max()
            inputs[f] = st.sidebar.number_input(f"{f} ({min_v:.1f} a {max_v:.1f})", value=float(df_c[f].median()))
        
        vu = modelo.predict(np.array([list(inputs.values())]))[0]
        std = np.std(df_c['Valor Unitário'] - preds)
        min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
        total = vu * inputs['Área Privativa']
        
        c1, c2, c3 = st.columns(3)
        c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
        c2.metric("V.U. Médio", f"R$ {vu:,.2f}")
        c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
        st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
        
        pdf = gerar_laudo_pdf({'vu': vu, 'min': min_v, 'max': max_v, 'total': total}, fig, eq_str, inputs)
        st.download_button("📥 Baixar Laudo PDF Completo", pdf, "laudo_tecnico.pdf")
