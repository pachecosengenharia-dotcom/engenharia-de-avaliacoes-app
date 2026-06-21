import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# Carrega o arquivo
arquivo_csv = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")

if arquivo_csv is not None:
    # 1. Leitura do arquivo
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    
    # 2. Definição explícita das colunas que são números
    # Estas são as colunas que estão exatamente no seu CSV
    # 2. Definição explícita das colunas que são números
  # 2. Definição explícita das colunas que são números
    cols_numericas = [
        'Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 
        'Padrão de Acabamento', 'Suite', 'Estado de Conservação', 
        'Idade Aparente', 'Setor urbano', 'Data do Evento'
        # 'Valor Total' foi removido propositalmente daqui
    ]
    col_alvo = 'Valor Unitário'
    
    # ... (seu código de limpeza e conversão segue o mesmo) ...

    # 4. Ajuste do Modelo
    # Aqui garantimos que apenas as colunas da lista acima entrem no X
    features = [c for c in cols_numericas if c in df_clean.columns]
    
    X = df_clean[features]
    y = df_clean[col_alvo]
    
    modelo = LinearRegression().fit(X, y)
    
    st.success(f"Modelo treinado! Variáveis utilizadas: {features}")
    
    # 3. Limpeza rigorosa: converte tudo para float, tratando vírgula decimal
    df_clean = pd.DataFrame()
    for col in cols_numericas:
        if col in df.columns:
            # Substitui vírgula por ponto e converte para float
            df_clean[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    
    # Remove linhas vazias
    df_clean = df_clean.dropna()

    # 4. Verificação se temos dados para treinar
    if not df_clean.empty:
        col_alvo = 'Valor Unitário'
        features = [c for c in df_clean.columns if c != col_alvo]
        
        X = df_clean[features]
        y = df_clean[col_alvo]
        
        # Treinamento
        modelo = LinearRegression().fit(X, y)
        
        st.success("Modelo treinado com sucesso!")
        st.write("Variáveis utilizadas:", features)
        
        # Exibição de coeficientes
        st.write("Impacto das variáveis (Coeficientes):")
        st.bar_chart(pd.Series(modelo.coef_, index=features))
        
        # Input para predição
        st.sidebar.header("⚙️ Parâmetros para Avaliação")
        inputs = {}
        for f in features:
            inputs[f] = st.sidebar.number_input(f, value=float(df_clean[f].median()))
            
        if st.sidebar.button("Calcular V.U."):
            input_array = np.array([inputs[f] for f in features]).reshape(1, -1)
            pred = modelo.predict(input_array)[0]
            st.metric("V.U. Estimado", f"R$ {pred:,.2f}")
    else:
        st.error("Erro: Não foi possível processar os dados. Verifique se o CSV contém números válidos.")
else:
    st.info("Por favor, carregue o arquivo CSV para ver os gráficos e resultados.")
