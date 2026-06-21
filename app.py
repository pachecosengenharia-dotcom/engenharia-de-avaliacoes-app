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

# --- Função PDF Corrigida ---
def gerar_laudo_pdf(d, fig, eq_str, inputs):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    
    # Equação formatada em várias linhas para não cortar
    c.setFont("Helvetica", 9)
    c.drawString(50, 770, "Equação do Modelo:")
    y_pos = 755
    # Quebra a string da equação a cada 100 caracteres
    for i in range(0, len(eq_str), 100):
        c.drawString(50, y_pos, eq_str[i:i+100])
        y_pos -= 15
    
    # Parâmetros Utilizados
    c.drawString(50, y_pos - 10, "Parâmetros Utilizados:")
    y_pos -= 25
    for k, v in inputs.items():
        c.drawString(60, y_pos, f"- {k}: {v:.2f}")
        y_pos -= 12
    
    # Métricas
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y_pos - 10, f"Resultados: V.U. Médio R$ {d['vu']:,.2f} | Total R$ {d['total']:,.2f}")
    c.drawString(50, y_pos - 25, f"Intervalo (95%): R$ {d['min']:,.2f} a R$ {d['max']:,.2f}")
    
    # Gráfico
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    img_buf.seek(0)
    c.drawImage(ImageReader(img_buf), 50, 50, width=400, height=250)
    
    c.save(); buffer.seek(0)
    return buffer

# --- Processamento ---
arquivo = st.sidebar.file_uploader("Carregar Base (CSV)", type="csv")
if arquivo:
    df = pd.read_csv(arquivo, sep=";", encoding='latin-1')
    features = ['Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 'Padrão de Acabamento', 
                'Suite', 'Estado de Conservação', 'Idade Aparente', 'Setor urbano', 'Data do Evento']
    df_c = pd.DataFrame()
    for col in features + ['Valor Unitário']:
        df_c[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
    df_c = df_c.dropna()

    if not df_c.empty:
        modelo = LinearRegression().fit(df_c[features], df_c['Valor Unitário'])
        eq_str = f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features, modelo.coef_)])
        
        st.subheader("Equação e Diagnóstico")
        st.latex(eq_str)
        preds = modelo.predict(df_c[features])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.scatter(df_c['Valor Unitário'], preds); ax1.set_title("Aderência")
        ax2.scatter(preds, df_c['Valor Unitário'] - preds); ax2.axhline(0, color='red'); ax2.set_title("Resíduos")
        st.pyplot(fig)

        st.sidebar.header("⚙️ Parâmetros (Limites NBR)")
        inputs = {f: st.sidebar.number_input(f, value=float(df_c[f].median())) for f in features}
        
        if st.sidebar.button("Calcular"):
            vu = modelo.predict(np.array([list(inputs.values())]))[0]
            std = np.std(df_c['Valor Unitário'] - preds)
            min_v, max_v = vu - (1.96 * std), vu + (1.96 * std)
            total = vu * inputs['Área Privativa']
            
            st.metric("V.U. Médio", f"R$ {vu:,.2f}")
            pdf = gerar_laudo_pdf({'vu': vu, 'min': min_v, 'max': max_v, 'total': total}, fig, eq_str, inputs)
            st.download_button("📥 Baixar Laudo PDF Completo", pdf, "laudo_tecnico.pdf")
