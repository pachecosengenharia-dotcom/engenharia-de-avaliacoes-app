import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

st.title("📊 Engenharia de Avaliações - Debug")

arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        # Lê o ficheiro tentando várias codificações
        df = pd.read_csv(regiao, sep=';', encoding='latin-1')
        
        # Mostra as colunas encontradas para você conferir!
        st.write("Colunas encontradas no seu CSV:")
        st.write(df.columns.tolist())
        
        # Limpeza básica: remove espaços nos nomes das colunas
        df.columns = df.columns.str.strip()
        
        # Verifica se 'Valor Unitário' existe
        if 'Valor Unitário' not in df.columns:
            st.error("ERRO: A coluna 'Valor Unitário' não foi encontrada!")
        else:
            st.success("Coluna 'Valor Unitário' encontrada!")
            
            # Tenta converter colunas para numérico
            df_num = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace('.','').str.replace(',','.'), errors='coerce'))
            df_num = df_num.dropna()
            
            if df_num.empty:
                st.error("Não foram encontrados dados numéricos suficientes para treinar o modelo.")
            else:
                st.write(f"Dados prontos! Linhas válidas: {len(df_num)}")
                
    except Exception as e:
        st.error(f"Erro ao ler o ficheiro: {e}")
