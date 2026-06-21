import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import io
import re

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")

st.title("📊 Sistema Profissional de Engenharia de Avaliações")
st.markdown("Insira sua planilha de mercado e ajuste os parâmetros para calcular o valor de mercado.")

# Barra Lateral - Upload dos Dados
st.sidebar.header("📁 Base de Dados")
arquivo_upload = st.sidebar.file_uploader("Arraste sua planilha (.csv)", type=["csv"])

if arquivo_upload:
    try:
        df = pd.read_csv(arquivo_upload, delimiter=';', encoding='latin-1')
        if len(df.columns) <= 1:
            df = pd.read_csv(arquivo_upload, delimiter=',', encoding='latin-1')
    except:
        df = pd.read_csv(arquivo_upload, delimiter=',', encoding='latin-1')
        
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Limpeza simples e segura das colunas (remove espaços nas pontas)
    df.columns = df.columns.astype(str).str.strip()

    # Dicionário calibrado EXATAMENTE com os cabeçalhos reais da sua planilha
    colunas_possiveis = {
        'Preco': ['Preco', 'Preço', 'Valor', 'Vlr', 'PRECO', 'PREÇO', 'Valor Total', 'Valor_Total'],
        'Area_Construida': ['Area_Construida', 'Área_Construída', 'Area Construida', 'Área Construída', 'Area Const', 'Área Const', 'AREA_CONSTRUIDA', 'Área Privativa', 'Area Privativa'],
        'Area_Terreno': ['Area_Terreno', 'Área_Terreno', 'Area Terreno', 'Terreno', 'AREA_TERRENO', 'Área do Terreno', 'Area do Terreno'],
        'Quartos': ['Quartos', 'Dormitorios', 'Dormitórios', 'QUARTOS'],
        'Suites': ['Suites', 'Suítes', 'SUITES', 'Suite', 'Suíte'],
        'Vagas': ['Vagas', 'Garagem', 'VAGAS'],
        'Conservacao': ['Conservacao', 'Conservação', 'CONSERVACAO', 'Estado de Conservação', 'Estado de Conservacao'],
        'Padrao_Acabamento': ['Padrao_Acabamento', 'Padrão_Acabamento', 'Padrao', 'Padrão', 'Acabamento', 'PADRAO', 'Padrão de Acabamento', 'Padrao de Acabamento'],
        'Setor_Urbano': ['Setor_Urbano', 'Setor Urbano', 'Setor', 'SETOR', 'SETOR_URBANO', 'Setor urbano'],
        'Data_Evento': ['Data_Evento', 'Data Evento', 'Data', 'DATA', 'Data do Evento'],
        'Evento': ['Evento', 'EVENTO']
    }

    # Renomeação das colunas identificadas
    for padrao, sinonimos in colunas_possiveis.items():
        for col in df.columns:
            if col in sinonimos:
                df = df.rename(columns={col: padrao})

    # Obrigatoriedades mínimas
    if 'Preco' not in df.columns or 'Area_Construida' not in df.columns:
        st.error(f"Não mapeamos as colunas essenciais (Preco e Area_Construida). Verifique se os nomes das colunas na sua planilha batem com as esperadas. Colunas atuais: {list(df.columns)}")
    else:
        todas_vars = ['Area_Construida', 'Area_Terreno', 'Quartos', 'Suites', 'Vagas', 'Conservacao', 'Padrao_Acabamento', 'Setor_Urbano', 'Data_Evento', 'Evento']
        variaveis_independentes = [v for v in todas_vars if v in df.columns]

        def limpar_numero(valor):
            txt = str(valor).strip().replace('R$', '').replace(' ', '')
            if not txt or txt.lower() in ['nan', 'null', '']: return np.nan
            if ',' in txt and '.' in txt: txt = txt.replace('.', '')
            txt = txt.replace(',', '.')
            txt = re.sub(r'[^\d.]', '', txt)
            try: return float(txt)
            except: return np.nan

        # Limpeza numérica estrita de todas as colunas
        df['Preco'] = df['Preco'].astype(str).apply(limpar_numero)
        for col in variaveis_independentes:
            df[col] = df[col].astype(str).apply(limpar_numero)
            if col == 'Setor_Urbano':
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 500.0)
            else:
                df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 1.0)

        df = df.dropna(subset=['Preco', 'Area_Construida'])

        # Painel de Controle Lateral Dinâmico
        st.sidebar.header("⚙️ Características do Imóvel")
        caracteristicas_avaliando = {}
        
        if 'Area_Construida' in variaveis_independentes:
            caracteristicas_avaliando['Area_Construida'] = st.sidebar.number_input("Área Construída (m²)", value=120.0, step=1.0)
        if 'Area_Terreno' in variaveis_independentes:
            caracteristicas_avaliando['Area_Terreno'] = st.sidebar.number_input("Área do Terreno (m²)", value=360.0, step=1.0)
        if 'Quartos' in variaveis_independentes:
            caracteristicas_avaliando['Quartos'] = st.sidebar.slider("Quantidade de Quartos", 1, 5, 3)
        if 'Suites' in variaveis_independentes:
            caracteristicas_avaliando['Suites'] = st.sidebar.slider("Quantidade de Suítes", 0, 5, 1)
        if 'Vagas' in variaveis_independentes:
            caracteristicas_avaliando['Vagas'] = st.sidebar.slider("Vagas de Garagem", 0, 5, 2)
        if 'Conservacao' in variaveis_independentes:
            caracteristicas_avaliando['Conservacao'] = st.sidebar.selectbox("Estado de Conservação", [1, 2, 3], index=1, format_func=lambda x: {1:"Regular", 2:"Bom", 3:"Excelente"}[x])
        if 'Padrao_Acabamento' in variaveis_independentes:
            caracteristicas_avaliando['Padrao_Acabamento'] = st.sidebar.selectbox("Padrão de Acabamento", [1, 2, 3], index=1, format_func=lambda x: {1:"Baixo", 2:"Médio", 3:"Alto"}[x])
        
        # Campo numérico para Setor Urbano (0 a 5000)
        if 'Setor_Urbano' in variaveis_independentes:
            try:
                valor_inicial = float(df['Setor_Urbano'].median())
                if np.isnan(valor_inicial): valor_inicial = 500.0
            except:
                valor_inicial = 500.0
                
            caracteristicas_avaliando['Setor_Urbano'] = st.sidebar.number_input(
                "Setor_Urbano", 
                min_value=0.0,
                max_value=5000.0,
                value=valor_inicial, 
                step=10.0
            )

        if 'Data_Evento' in variaveis_independentes:
            caracteristicas_avaliando['Data_Evento'] = st.sidebar.number_input("Data do Evento", value=2026.0, step=1.0)
        if 'Evento' in variaveis_independentes:
            caracteristicas_avaliando['Evento'] = st.sidebar.number_input("Fator de Evento", value=1.0, step=0.05)

        if len(df) >= 2:
            X = df[variaveis_independentes]
            y = df['Preco']
            
            # Converte para matriz NumPy pura eliminando dependência estrita de nomes no fit/predict
            modelo = Ridge(alpha=1.0).fit(X.values, y.values)
            
            # Monta os dados de teste seguindo exatamente a ordem correta das colunas
            dados_imovel_lista = [caracteristicas_avaliando[var] for var in variaveis_independentes]
            dados_imovel = np.array([dados_imovel_lista])
            
            preco_estimado = max(0, modelo.predict(dados_imovel)[0])
            r2_score = modelo.score(X.values, y.values)
            limite_inferior, limite_superior = preco_estimado * 0.85, preco_estimado * 1.15

            # Exibição dos Resultados
            c1, c2, c3 = st.columns(3)
