import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# 1. Limpeza mais robusta
def clean_numeric(x):
    return pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

# 2. Verificar VIF (Para eliminar variáveis redundantes)
def check_vif(X):
    X_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_data

# No fluxo:
df_modelo = df[features + [col_alvo]].apply(clean_numeric)
# ... após o dropna ...
# Recomendo usar statsmodels para o modelo final, pois ele fornece o R² ajustado e p-values 
# que são exigidos em laudos de Engenharia de Avaliações.
st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Laudo Técnico")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

def gerar_pdf(pred_unit, pred_total, minimo, maximo, eq_str):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Equação: {eq_str}")
    c.drawString(50, 740, f"V.U. Mínimo: R$ {minimo:,.2f} | Médio: R$ {pred_unit:,.2f} | Máximo: R$ {maximo:,.2f}")
    c.drawString(50, 710, f"Valor Total: R$ {pred_total:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        col_alvo = 'Valor Unitário'
        # Seleção dinâmica de variáveis numéricas
        features = [c for c in df.columns if c != col_alvo and pd.to_numeric(df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce').notna().sum() > len(df)*0.5]
        
        df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce'))
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X = df_modelo[features]
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # Equação para o Laudo
            eq_str = f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f} * {n})" for n, c in zip(features, modelo.coef_)])
            st.subheader("Equação do Modelo")
            st.latex(eq_str)

            # Cálculos de Parâmetros
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
            pred_unit = modelo.predict([inputs])[0]
            
            # Cálculo de amplitude (resíduos)
            residuos = y - modelo.predict(X)
            erro_padrao = np.std(residuos)
            minimo, maximo = pred_unit - (1.96 * erro_padrao), pred_unit + (1.96 * erro_padrao)
            area = inputs[features.index(next((c for c in features if 'área' in c.lower()), features[0]))] if any('área' in f.lower() for f in features) else 1
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            col1.metric("V.U. Mínimo", f"R$ {minimo:,.2f}")
            col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            col3.metric("V.U. Máximo", f"R$ {maximo:,.2f}")
            st.metric("Valor Total (Médio)", f"R$ {pred_unit * area:,.2f}")

            # Diagnóstico visual
            n, p = len(y), len(features) + 1
            X_mat = np.column_stack([np.ones(n), X])
            leverage = np.diag(X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T)
            cooks_dist = (residuos**2 / (p * np.var(residuos))) * (leverage / (1 - leverage)**2)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            ax1.scatter(y, modelo.predict(X), alpha=0.5); ax1.set_title("Aderência (Obs vs Prev)")
            ax2.scatter(modelo.predict(X), residuos, alpha=0.5, color='orange'); ax2.axhline(0, color='black', linestyle='--'); ax2.set_title("Resíduos")
            ax3.stem(cooks_dist); ax3.set_title("Distância de Cook")
            st.pyplot(fig)

            # Botão de Download
            pdf_data = gerar_pdf(pred_unit, pred_unit * area, minimo, maximo, eq_str)
            st.download_button("📥 Baixar Laudo Completo em PDF", data=pdf_data, file_name="laudo_avaliacao.pdf", mime="application/pdf")
        else:
            st.error("Dados insuficientes.")
    except Exception as e:
        st.error(f"Erro técnico: {e}") 

