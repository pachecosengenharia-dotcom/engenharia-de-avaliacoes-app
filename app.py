import streamlit as st
import pandas as pd
import glob
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Seletor de Regiões")

# 1. Listar planilhas disponíveis (ex: 'Goiânia - Setor Bueno.csv', 'Goiânia - Marista.csv')
arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região (Planilha):", arquivos)

try:
    if regiao:
        # 2. Leitura da planilha selecionada
        df = pd.read_csv(regiao, sep=";", encoding='latin-1')
        df.columns = [c.strip() for c in df.columns]

        # 3. Definição das colunas e Limpeza
        # ... (manter aqui a lógica de limpeza de dados que validamos antes)

        st.success(f"Região carregada: {regiao}")
        
        # 4. Cálculo do Modelo e Equação
        # (O mesmo código de regressão e exibição que construímos)
        
    else:
        st.info("Por favor, selecione uma planilha de região na barra lateral.")

except Exception as e:
    st.error(f"Erro ao processar a planilha {regiao}: {e}")
