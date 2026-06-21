import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Configuração ---
st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

arquivo_csv = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")

if arquivo_csv is not None:
    # 1. Leitura e Limpeza
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    features_list = ['Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 
                     'Padrão de Acabamento', 'Suite', 'Estado de Conservação', 
                     'Idade Aparente', 'Setor urbano', 'Data do Evento']
    col_alvo = 'Valor Unitário'
    
    df_clean = pd.DataFrame()
    for col in features_list + [col_alvo]:
        if col in df.columns:
            df_clean[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()

    if not df_clean.empty:
        X = df_clean[features_list]
        y = df_clean[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        # --- EXIBIÇÃO DA EQUAÇÃO ---
        eq_str = f"V.U. = {modelo.intercept_:.2f} " + " ".join([f"+ ({c:.2f}*{n})" for n, c in zip(features_list, modelo.coef_)])
        st.subheader("Equação do Modelo")
        st.latex(eq_str)
        
        # --- CÁLCULOS PARA GRÁFICOS ---
        predicoes = modelo.predict(X)
        residuos = y - predicoes
        
        # Distância de Cook (cálculo simplificado)
        n, p = len(y), len(features_list) + 1
        X_mat = np.column_stack([np.ones(n), X])
        leverage = np.diag(X_mat @ np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T)
        cooks_dist = (residuos**2 / (p * np.var(residuos))) * (leverage / (1 - leverage)**2)
        
        # --- EXIBIÇÃO DOS GRÁFICOS ---
        st.subheader("Diagnóstico do Modelo")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        ax1.scatter(y, predicoes, alpha=0.5); ax1.set_title("Aderência (Obs vs Prev)")
        ax2.scatter(predicoes, residuos, alpha=0.5, color='orange'); ax2.axhline(0, color='black', linestyle='--'); ax2.set_title("Resíduos")
        ax3.stem(cooks_dist); ax3.set_title("Distância de Cook")
        
        st.pyplot(fig)
        
        # --- CÁLCULOS E METRICAS DE AVALIAÇÃO ---
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = {f: st.sidebar.number_input(f, value=float(df_clean[f].median())) for f in features_list}
        
        if st.sidebar.button("Calcular Precificação"):
            pred_unit = modelo.predict(np.array([list(inputs.values())]))[0]
            erro_padrao = np.std(residuos)
            
            area = inputs.get('Área Privativa', 1)
            
            st.subheader("Resultados")
            col1, col2, col3 = st.columns(3)
            col1.metric("V.U. Mínimo", f"R$ {pred_unit - (1.96 * erro_padrao):,.2f}")
            col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            col3.metric("V.U. Máximo", f"R$ {pred_unit + (1.96 * erro_padrao):,.2f}")
            st.metric("Valor Total Estimado", f"R$ {pred_unit * area:,.2f}")
    else:
        st.error("Erro nos dados.")
