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

# --- FUNÇÃO DE GERAÇÃO DE PDF ---
def gerar_laudo_pdf(dados):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Equação: {dados['eq']}")
    c.drawString(50, 750, f"V.U. Médio Estimado: R$ {dados['vu_medio']:,.2f}")
    c.drawString(50, 730, f"Campo de Arbítrio: R$ {dados['min']:,.2f} a R$ {dados['max']:,.2f}")
    c.drawString(50, 710, f"Valor Total: R$ {dados['total']:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

arquivo_csv = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")

if arquivo_csv is not None:
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    features_list = ['Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 
                     'Padrão de Acabamento', 'Suite', 'Estado de Conservação', 
                     'Idade Aparente', 'Setor urbano', 'Data do Evento']
    col_alvo = 'Valor Unitário'
    
    df_clean = pd.DataFrame()
    for col in features_list + [col_alvo]:
        if col in df.columns:
            df_clean[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
    df_clean = df_clean.dropna()

    if not df_clean.empty:
        modelo = LinearRegression().fit(df_clean[features_list], df_clean[col_alvo])
        eq_str = f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features_list, modelo.coef_)])
        st.latex(eq_str)
        
        # Inputs e Validação
        st.sidebar.header("⚙️ Parâmetros")
        inputs = {f: st.sidebar.number_input(f, value=float(df_clean[f].median())) for f in features_list}
        
        if st.sidebar.button("Calcular e Gerar Laudo"):
            pred_unit = modelo.predict(np.array([list(inputs.values())]))[0]
            residuos = df_clean[col_alvo] - modelo.predict(df_clean[features_list])
            erro_padrao = np.std(residuos)
            
            # Preparar dados para o PDF
            dados_pdf = {
                'eq': eq_str, 'vu_medio': pred_unit,
                'min': pred_unit - (1.96 * erro_padrao),
                'max': pred_unit + (1.96 * erro_padrao),
                'total': pred_unit * inputs.get('Área Privativa', 1)
            }
            
            # Exibir resultados na tela
            st.metric("V.U. Médio", f"R$ {dados_pdf['vu_medio']:,.2f}")
            st.download_button("📥 Baixar Laudo Completo (PDF)", 
                               data=gerar_laudo_pdf(dados_pdf), 
                               file_name="laudo_avaliacao.pdf", mime="application/pdf")
