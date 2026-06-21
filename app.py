import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

st.title("📊 Engenharia de Avaliações")

# Verifica arquivos disponíveis
arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
regiao = st.sidebar.selectbox("Selecione a Região:", arquivos)

if regiao:
    try:
        # Leitura básica
        df = pd.read_csv(regiao, sep=';', encoding='latin-1')
        df.columns = df.columns.str.strip()
        
        # Filtro de coluna alvo
        col_alvo = "Valor Unitário"
        
        if col_alvo not in df.columns:
            st.error(f"Coluna '{col_alvo}' não encontrada. Colunas presentes: {df.columns.tolist()}")
        else:
            # Seleção de variáveis numéricas
            features = [c for c in df.columns if c != col_alvo and df[c].dtype != 'object']
            
            # Limpeza de dados
            df_modelo = df[features + [col_alvo]].dropna()
            
            if not df_modelo.empty:
                X = df_modelo[features]
                y = df_modelo[col_alvo]
                
                # Modelo
                modelo = LinearRegression().fit(X, y)
                
                # Interface de Input
                st.sidebar.header("Parâmetros do Imóvel")
                inputs = [st.sidebar.number_input(f"{n}", value=float(df_modelo[n].median())) for n in features]
                
                # Previsão
                pred = modelo.predict([inputs])[0]
                st.metric("Valor Unitário Estimado", f"R$ {pred:,.2f}")
                st.success("Cálculo realizado com sucesso!")
            else:
                st.error("Dados insuficientes para o cálculo.")
    except Exception as e:
        st.error(f"Erro no sistema: {e}")
