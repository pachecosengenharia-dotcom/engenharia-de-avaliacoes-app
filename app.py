import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Seletor de Regiões")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        col_alvo = 'Valor Unitário'
        features = ['Área Construída', 'Área do Terreno', 'Evento', 'Padrão de Acabamento', 
                    'Estado de Conservação', 'Setor urbano', 'Data do Evento', 'Quartos', 'Suítes']
        
        df_modelo = pd.DataFrame()
        if col_alvo in df.columns:
            df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        
        for col in features:
            if col in df.columns:
                df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        
        df_modelo = df_modelo.dropna()

        if not df_modelo.empty:
            X = df_modelo.drop(columns=[col_alvo])
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # --- CÁLCULO DO INTERVALO (MÍNIMO E MÁXIMO) ---
            residuos = y - modelo.predict(X)
            erro_padrao = np.std(residuos)
            
            # Interface de Parâmetros
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = [st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median())) for col in X.columns]
            
            # Predições
            pred_unit = modelo.predict([inputs])[0]
            # Intervalo de 95% de confiança aproximado
            minimo = pred_unit - (1.96 * erro_padrao)
            maximo = pred_unit + (1.96 * erro_padrao)
            
            # Exibição
            area = inputs[0] if len(inputs) > 0 else 1
            
            col1, col2, col3 = st.columns(3)
            col1.metric("V.U. Mínimo", f"R$ {minimo:,.2f}")
            col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            col3.metric("V.U. Máximo", f"R$ {maximo:,.2f}")
            
            st.metric("Valor Total Estimado (Médio)", f"R$ {pred_unit * area:,.2f}")
            
            st.subheader("Análise de Aderência")
            fig, ax = plt.subplots()
            ax.scatter(y, modelo.predict(X), alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            st.pyplot(fig)
        else:
            st.error("Dados insuficientes nesta planilha.")
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
