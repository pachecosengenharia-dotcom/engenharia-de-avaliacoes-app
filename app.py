import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Configuração ---
st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações (NBR 14653)")

# --- Função PDF ---
def gerar_laudo_pdf(dados):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Equação: {dados['eq']}")
    c.drawString(50, 750, f"V.U. Médio: R$ {dados['vu']:,.2f}")
    c.drawString(50, 730, f"V.U. Mínimo: R$ {dados['min']:,.2f} | V.U. Máximo: R$ {dados['max']:,.2f}")
    c.drawString(50, 710, f"Valor Total: R$ {dados['total']:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

# --- Processamento ---
arquivo = st.sidebar.file_uploader("Upload CSV", type="csv")
if arquivo:
    df = pd.read_csv(arquivo, sep=";", encoding='latin-1')
    features = ['Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 
                'Padrão de Acabamento', 'Suite', 'Estado de Conservação', 
                'Idade Aparente', 'Setor urbano', 'Data do Evento']
    col_alvo = 'Valor Unitário'
    
    df_c = pd.DataFrame()
    for col in features + [col_alvo]:
        df_c[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    df_c = df_c.dropna()

    if not df_c.empty:
        modelo = LinearRegression().fit(df_c[features], df_c[col_alvo])
        
        # 1. Equação e Gráficos
        st.latex(f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)]))
        
        preds = modelo.predict(df_c[features])
        residuos = df_c[col_alvo] - preds
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.scatter(df_c[col_alvo], preds); ax1.set_title("Aderência")
        ax2.scatter(preds, residuos); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
        st.pyplot(fig)
        
        # 2. Parâmetros e Cálculos
        st.sidebar.header("⚙️ Parâmetros")
        inputs = {f: st.sidebar.number_input(f, value=float(df_c[f].median())) for f in features}
        
        if st.sidebar.button("Calcular"):
            vu_medio = modelo.predict(np.array([list(inputs.values())]))[0]
            std = np.std(residuos)
            min_v, max_v = vu_medio - (1.96 * std), vu_medio + (1.96 * std)
            area = inputs['Área Privativa']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("V.U. Mínimo", f"R$ {min_v:,.2f}")
            c2.metric("V.U. Médio", f"R$ {vu_medio:,.2f}")
            c3.metric("V.U. Máximo", f"R$ {max_v:,.2f}")
            st.metric("Valor Total Estimado", f"R$ {vu_medio * area:,.2f}")
            
            # Download
            pdf_data = gerar_laudo_pdf({'eq': '...', 'vu': vu_medio, 'min': min_v, 'max': max_v, 'total': vu_medio*area})
            st.download_button("📥 Baixar PDF", pdf_data, "laudo.pdf")
