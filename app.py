import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

st.title("📊 Engenharia de Avaliações")

# Carrega arquivos
arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        # Lê o arquivo
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = df.columns.str.strip()
        
        # O nome exato que o seu debug mostrou
        col_alvo = "Valor Unitário"
        
        if col_alvo not in df.columns:
            st.error(f"Coluna '{col_alvo}' não encontrada. Verifique o seu CSV.")
        else:
            # Seleciona variáveis numéricas (remove endereços, datas, etc)
            features = [c for c in df.columns if c != col_alvo and df[c].dtype != 'object']
            
            # Prepara o modelo
            df_modelo = df[features + [col_alvo]].dropna()
            
            if not df_modelo.empty:
                X = df_modelo[features]
                y = df_modelo[col_alvo]
                modelo = LinearRegression().fit(X, y)
                
                # Exibe a equação de forma simples
                st.write("Equação calculada com sucesso!")
                
                # Interface
                st.sidebar.header("Parâmetros do Imóvel")
                inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
                
                # Resultado
                pred = modelo.predict([inputs])[0]
                st.metric("Valor Unitário Estimado", f"R$ {pred:,.2f}")
                
            else:
                st.error("Dados insuficientes para calcular.")
                
    except Exception as e:
        st.error(f"Erro inesperado: {e}")
