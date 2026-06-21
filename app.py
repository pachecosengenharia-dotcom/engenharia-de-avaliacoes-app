import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    # Usando fonte 'Helvetica' padrão que não gera tarjas pretas
    c.setFont("Helvetica", 10)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.drawString(50, 780, f"V.U. Médio: R$ {d['vu']:,.2f} | Total: R$ {d['total']:,.2f}")
    
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
    # Limpa nomes das colunas de espaços extras
    df.columns = df.columns.str.strip()
    
    cols = df.columns.tolist()
    target = st.sidebar.selectbox("Coluna Valor Unitário:", cols)
    features = st.sidebar.multiselect("Variáveis Explicativas:", [c for c in cols if c != target])
    
    if features and target:
        df_c = df.copy()
        for col in features + [target]:
            df_c[col] = pd.to_numeric(df_c[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.'), errors='coerce')
        df_c = df_c.dropna()

        if not df_c.empty:
            modelo = LinearRegression().fit(df_c[features], df_c[target])
            st.sidebar.header("⚙️ Parâmetros")
            inputs = {f: st.sidebar.number_input(f, value=float(df_c[f].median())) for f in features}
            
            if st.sidebar.button("Calcular Precificação"):
                vu = modelo.predict(np.array([list(inputs.values())]))[0]
                preds = modelo.predict(df_c[features])
                std = np.std(df_c[target] - preds)
                
                # Busca por área de forma mais flexível
                col_area = next((c for c in features if 'area' in c.lower() or 'área' in c.lower()), None)
                total = vu * inputs[col_area] if col_area else vu
                
                st.metric("Valor Total Estimado", f"R$ {total:,.2f}")
                
                fig, ax = plt.subplots()
                ax.scatter(df_c[target], preds)
                
                pdf = gerar_laudo_pdf({'vu': vu, 'total': total}, fig, "eq", inputs)
                st.download_button("📥 Baixar Laudo", pdf, "laudo.pdf")
