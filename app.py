import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Laudo Técnico")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        col_alvo = 'Valor Unitário'
        features = [c for c in df.columns if c != col_alvo and pd.to_numeric(df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce').notna().sum() > len(df)*0.5]

        df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce'))
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X = df_modelo[features]
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # 1. Equação
            st.subheader("Equação do Modelo")
            intercept = modelo.intercept_
            eq_str = f"V.U. = {intercept:.2f} " + " ".join([f"+ ({c:.2f} * {n})" for n, c in zip(features, modelo.coef_)])
            st.latex(eq_str)

            # 2. Parâmetros e Cálculos
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
            
            pred_unit = modelo.predict([inputs])[0]
            residuos = y - modelo.predict(X)
            erro_padrao = np.std(residuos)
            
            # Mínimo e Máximo (95% confiança)
            minimo, maximo = pred_unit - (1.96 * erro_padrao), pred_unit + (1.96 * erro_padrao)
            area = inputs[features.index(next((c for c in features if 'área' in c.lower()), features[0]))]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("V.U. Mínimo", f"R$ {minimo:,.2f}")
            col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            col3.metric("V.U. Máximo", f"R$ {maximo:,.2f}")
            st.metric("Valor Total (Médio)", f"R$ {pred_unit * area:,.2f}")

            # 3. Gráficos
            st.subheader("Diagnóstico de Aderência e Resíduos")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.scatter(y, modelo.predict(X), alpha=0.5)
            ax1.set_title("Observado vs Previsto")
            ax2.scatter(modelo.predict(X), residuos, alpha=0.5, color='orange')
            ax2.axhline(0, color='black', linestyle='--')
            ax2.set_title("Resíduos")
            st.pyplot(fig)

        else:
            st.error("Dados insuficientes.")
    except Exception as e:
        st.error(f"Erro: {e}")
