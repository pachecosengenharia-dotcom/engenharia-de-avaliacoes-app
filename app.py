import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Modelo Final")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        # 1. FORÇAR ALVO CORRETO
        col_alvo = 'Valor Unitário'
        
        # 2. Selecionar apenas variáveis de entrada válidas (excluir o alvo da lista)
        features = [c for c in df.columns if c != col_alvo and pd.to_numeric(df[c].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce').notna().sum() > len(df)*0.5]

        # 3. Limpeza
        df_modelo = df[features + [col_alvo]].apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce'))
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X = df_modelo[features]
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)

            # 4. Interface e Cálculo
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = [st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median())) for col in features]
            
            pred_unit = modelo.predict([inputs])[0]
            
            # Cálculo de Valor Total (procurando área construída dinamicamente)
            col_area = next((c for c in features if 'área' in c.lower() and 'construída' in c.lower()), features[0])
            area = inputs[features.index(col_area)] if col_area in features else 1
            
            col1, col2 = st.columns(2)
            col1.metric("Valor Unitário Estimado", f"R$ {pred_unit:,.2f} / m²")
            col2.metric("Valor Total Estimado", f"R$ {pred_unit * area:,.2f}")
            
            # Gráfico
            st.subheader("Análise de Aderência")
            fig, ax = plt.subplots()
            ax.scatter(y, modelo.predict(X), alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            st.pyplot(fig)
        else:
            st.error("Não foram encontrados dados numéricos suficientes para o cálculo.")

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
