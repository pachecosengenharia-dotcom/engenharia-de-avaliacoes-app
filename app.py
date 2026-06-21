import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Configuração ---
st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

arquivo_csv = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")

if arquivo_csv is not None:
    # 1. Leitura
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    
    # 2. Definição das colunas (incluindo todas que você precisa)
    features_list = [
        'Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 
        'Padrão de Acabamento', 'Suite', 'Estado de Conservação', 
        'Idade Aparente', 'Setor urbano', 'Data do Evento'
    ]
    col_alvo = 'Valor Unitário'
    
    # 3. Limpeza de dados (conversão de BR para float)
    df_clean = pd.DataFrame()
    for col in features_list + [col_alvo]:
        if col in df.columns:
            df_clean[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean = df_clean.dropna()

    if not df_clean.empty:
        # 4. Treinamento
        X = df_clean[features_list]
        y = df_clean[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        st.success("Modelo treinado com sucesso!")
        
        # 5. Interface de Parâmetros
        st.sidebar.header("⚙️ Parâmetros do Imóvel")
        inputs = {f: st.sidebar.number_input(f, value=float(df_clean[f].median())) for f in features_list}
        
        if st.sidebar.button("Calcular Precificação"):
            # Predição do Valor Unitário
            input_array = np.array([inputs[f] for f in features_list]).reshape(1, -1)
            pred_unit = modelo.predict(input_array)[0]
            
            # Cálculo de amplitude (erro padrão)
            residuos = y - modelo.predict(X)
            erro_padrao = np.std(residuos)
            minimo = pred_unit - (1.96 * erro_padrao)
            maximo = pred_unit + (1.96 * erro_padrao)
            
            # Cálculo do Valor Total (usando Área Privativa)
            area = inputs.get('Área Privativa', 1)
            pred_total = pred_unit * area
            min_total = minimo * area
            max_total = maximo * area
            
            # 6. Exibição Profissional
            st.subheader("Resultados da Avaliação")
            col1, col2, col3 = st.columns(3)
            col1.metric("V.U. Mínimo", f"R$ {minimo:,.2f}")
            col2.metric("V.U. Médio", f"R$ {pred_unit:,.2f}")
            col3.metric("V.U. Máximo", f"R$ {maximo:,.2f}")
            
            st.markdown("---")
            st.subheader("Estimativa de Valor Total")
            st.metric("Valor Total Estimado", f"R$ {pred_total:,.2f}")
            st.write(f"Intervalo de Valor Total: **R$ {min_total:,.2f} a R$ {max_total:,.2f}**")
    else:
        st.error("Erro no processamento dos dados.")
else:
    st.info("Aguardando upload do CSV.")
