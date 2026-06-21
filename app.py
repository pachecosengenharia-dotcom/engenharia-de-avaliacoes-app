import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Engenharia de Avaliações - Goiânia")

# Bloco único de processamento
try:
    # 1. Leitura
    df = pd.read_csv("Goiânia - GO.csv", sep=";", encoding='latin-1')
    df.columns = [c.strip() for c in df.columns]

    # 2. Definição das colunas
    # Nota: Ajuste os nomes abaixo se necessário para coincidir exatamente com seu CSV
    col_alvo = 'Valor Unitário'
    features = ['Área Construída', 'Área do Terreno', 'Evento', 'Padrão de Acabamento', 
                'Estado de Conservação', 'Setor urbano', 'Data do Evento', 'Quartos', 'Suítes']
    
    # 3. Limpeza
    df_modelo = pd.DataFrame()
    if col_alvo in df.columns:
        df_modelo[col_alvo] = pd.to_numeric(df[col_alvo].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    for col in features:
        if col in df.columns:
            df_modelo[col] = pd.to_numeric(df[col].astype(str).str.replace('.', '').str.replace(',', '.'), errors='coerce')
    
    df_modelo = df_modelo.dropna()

    # 4. Só executa cálculos se houver dados
    if not df_modelo.empty:
        X = df_modelo.drop(columns=[col_alvo])
        y = df_modelo[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        # Interface
        st.sidebar.header("⚙️ Parâmetros")
        inputs = [st.sidebar.number_input(f"{col}", value=float(df_modelo[col].median())) for col in X.columns]
        
        pred_unit = modelo.predict([inputs])[0]
        st.metric("Valor Unitário Estimado", f"R$ {pred_unit:,.2f} / m²")
        
        # Gráfico de Aderência
        st.subheader("Análise de Aderência")
        fig, ax = plt.subplots()
        ax.scatter(y, modelo.predict(X), alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        st.pyplot(fig)
        
    else:
        st.error("Dados insuficientes. Verifique se as colunas estão preenchidas corretamente no CSV.")

except Exception as e:
    st.error(f"Erro ao processar: {e}")
