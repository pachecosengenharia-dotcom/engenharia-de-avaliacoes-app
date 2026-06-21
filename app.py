import streamlit as st
import pandas as pd
import os

st.title("📊 Diagnóstico de Ficheiro CSV")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        # Tenta ler com a codificação mais comum (utf-8) ou latin-1
        df = pd.read_csv(regiao, sep=None, engine='python', encoding='latin-1')
        
        st.write("### Sucesso na leitura do arquivo!")
        st.write(f"Tamanho do arquivo: {df.shape[0]} linhas e {df.shape[1]} colunas.")
        
        st.write("### Nomes das colunas encontrados:")
        st.write(df.columns.tolist())
        
        st.write("### Primeiras 5 linhas:")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
