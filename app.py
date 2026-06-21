import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Configuração da Página ---
st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# --- Lógica Principal ---
arquivo_csv = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")

if arquivo_csv is not None:
    # 1. Leitura
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    
    # 2. Definição das colunas numéricas (sem 'Valor Total')
    cols_numericas = [
        'Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 
        'Padrão de Acabamento', 'Suite', 'Estado de Conservação', 
        'Idade Aparente', 'Setor urbano', 'Data do Evento'
    ]
    col_alvo = 'Valor Unitário'
    
    # 3. Limpeza dos dados
    df_clean = pd.DataFrame()
    
    # Função auxiliar para limpar números brasileiros (vírgula decimal)
    def clean_num(col):
        return pd.to_numeric(col.astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

    for col in cols_numericas + [col_alvo]:
        if col in df.columns:
            df_clean[col] = clean_num(df[col])
    
    df_clean = df_clean.dropna()

    # 4. Treinamento
    if not df_clean.empty:
        X = df_clean[cols_numericas]
        y = df_clean[col_alvo]
        
        modelo = LinearRegression().fit(X, y)
        st.success("Modelo treinado com sucesso!")
        
        # Exibir Coeficientes
        st.write("### Impacto das Variáveis (Coeficientes):")
        coefs = pd.Series(modelo.coef_, index=cols_numericas)
        st.bar_chart(coefs)
        
        # Predição
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = {f: st.sidebar.number_input(f, value=float(df_clean[f].median())) for f in cols_numericas}
        
        if st.sidebar.button("Calcular"):
            pred = modelo.predict(np.array([list(inputs.values())]))[0]
            st.metric("V.U. Estimado", f"R$ {pred:,.2f}")
    else:
        st.error("Erro: Dados insuficientes. Verifique se o separador e os nomes das colunas estão corretos.")
else:
    st.info("Por favor, carregue o arquivo CSV para iniciar.")
