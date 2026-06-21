import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import unicodedata
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

def normalizar_texto(texto):
    return unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower()

def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica", 10)
    c.drawString(50, 800, "Laudo Tecnico (NBR 14653)")
    c.drawString(50, 780, f"Equacao: {eq_str[:80]}...")
    c.drawString(50, 765, f"Min: R$ {d['min']:,.2f} | Medio: R$ {d['vu']:,.2f} | Max: R$ {d['max']:,.2f}")
    c.drawString(50, 750, f"Total: R$ {d['total']:,.2f}")
    
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 400, width=400, height=200)
    c.save(); buffer.seek(0)
    return buffer

arquivo = st.sidebar.file_uploader("Carregar CSV", type=["csv", "txt"])
if arquivo:
    raw_data = arquivo.getvalue().decode('latin-1')
    sep = ';' if raw_data.count(';') > raw_data.count(',') else ','
    df = pd.read_csv(io.StringIO(raw_data), sep=sep)
    df.columns = [normalizar_texto(c) for c in df.columns]
    
    cols = df.columns.tolist()
    target = st.sidebar.selectbox("Coluna Valor Unitario:", cols)
    features = st.sidebar.multiselect("Variaveis Explicativas:", [c for c in cols if c != target])
    
    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            eq_str = f"{target} = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)
            
            preds = modelo.predict(df_c[features])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.scatter(df_c[target], preds); ax1.set_title("Aderencia")
            ax2.scatter(preds, df_c[target] - preds); ax2.axhline(0, color='red'); ax2.set_title("Residuos")
            st.pyplot(fig)

            st.sidebar.header("⚙️ Parametros (Limites)")
            inputs = {f: st.sidebar.number_input(f"{f} ({df_c[f].min():.1f} a {df_c[f].max():.1f})", value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificacao"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                std = np.std(df_c[target] - preds)
                min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
                
                col_area = next((c for c in features if 'area' in c), None)
                total = vu * inputs[col_area] if col_area else vu
                
                c1
