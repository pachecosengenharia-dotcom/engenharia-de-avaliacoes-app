import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Seletor de Regiões")

# 1. Seleção da Região
arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        # 2. Leitura
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        # 3. Definição das colunas esperadas
        col_alvo = 'Valor Unitário'
        features = ['Área Construída', 'Área do Terreno', 'Evento', 'Padrão de Acabamento', 
                    'Estado de Conservação', 'Setor urbano', 'Data do Evento', 'Quartos', 'Suítes']
        
        # 4. Limpeza de dados (Força numérica)
        df_modelo = pd.DataFrame()
        if col_alvo in df.columns:
            df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        
        for col in features:
            if col in df.columns:
                df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
        
        df_modelo = df_modelo.dropna()

        # 5. Cálculo do Modelo (Apenas se houver dados)
        if not df_modelo.empty:
            X = df_modelo.drop(columns=[col_alvo])
            y = df_modelo[col_alvo]
            modelo = LinearRegression().fit(X, y)
            
            # Equação
            eq = f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({coef:.2f} * {col})" for col, coef in zip(X.columns, modelo.coef_)])
            st.subheader("Equação de Regressão")
            st.latex(eq)
            
            # Interface de Parâmetros
            st.sidebar.header("⚙️ Parâmetros do Imóvel")
            inputs = [st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median())) for col in X.columns]
            
            # Predições
            pred_unit = modelo.predict([inputs])[0]
            area = inputs[0] if len(inputs) > 0 else 1
            st.metric("Valor Unitário Estimado", f"R$ {pred_unit:,.2f} / m²")
            st.metric("Valor Total Estimado", f"R$ {pred_unit * area:,.2f}")
            
            # Gráfico de Aderência
            st.subheader("Análise de Aderência")
            fig, ax = plt.subplots()
            ax.scatter(y, modelo.predict(X), alpha=0.5)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            st.pyplot(fig)
        else:
            st.error("A planilha selecionada não possui dados numéricos válidos para processamento.")

    except Exception as e:
        st.error(f"Erro ao processar: {e}")
