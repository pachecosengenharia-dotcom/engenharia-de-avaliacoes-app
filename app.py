import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import io
import os
import re
import unicodedata

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")

st.title("📊 Sistema Profissional de Engenharia de Avaliações")
st.markdown("Modelagem estatística adaptativa e laudos técnicos em conformidade com as diretrizes normativas.")

# --- GERENCIAMENTO DE BANCO DE DADOS ---
st.sidebar.header("📁 Base de Dados")

planilhas_disponiveis = []
for f in os.listdir('.'):
    if f.endswith('.csv'):
        planilhas_disponiveis.append(f)

if os.path.exists("dados"):
    for f in os.listdir("dados"):
        if f.endswith('.csv') and f not in planilhas_disponiveis:
            planilhas_disponiveis.append(os.path.join("dados", f))

fonte_dados = st.sidebar.selectbox(
    "Escolha a Região/Município",
    options=["Selecionar região salva..."] + planilhas_disponiveis + ["Fazer Upload Manual (.csv)"]
)

df = None

def carregar_csv_com_seguranca(caminho_ou_buffer):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        for sep in [';', ',']:
            try:
                if isinstance(caminho_ou_buffer, str):
                    temp_df = pd.read_csv(caminho_ou_buffer, delimiter=sep, encoding=enc)
                else:
                    caminho_ou_buffer.seek(0)
                    temp_df = pd.read_csv(caminho_ou_buffer, delimiter=sep, encoding=enc)
                
                if len(temp_df.columns) > 1:
                    return temp_df
            except:
                continue
    return None

if fonte_dados and fonte_dados != "Selecionar região salva..." and fonte_dados != "Fazer Upload Manual (.csv)":
    df = carregar_csv_com_seguranca(fonte_dados)
    if df is not None:
        st.sidebar.success(f"📍 Região carregada: {os.path.basename(fonte_dados).replace('.csv', '')}")
elif fonte_dados == "Fazer Upload Manual (.csv)":
    arquivo_upload = st.sidebar.file_uploader("Arraste sua planilha (.csv)", type=["csv"])
    if arquivo_upload:
        df = carregar_csv_com_seguranca(arquivo_upload)

# --- PROCESSAMENTO TÉCNICO ADAPTATIVO ---
if df is not None:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    def remover_acentos(texto):
        return ''.join(c for c in unicodedata.normalize('NFD', str(texto)) if unicodedata.category(c) != 'Mn')
    
    df.columns = [remover_acentos(col).strip() for col in df.columns]

    colunas_possiveis = {
        'Preco': ['Preco', 'Valor', 'Vlr', 'PRECO', 'Valor Total', 'Valor_Total', 'Valor Unitario'],
        'Area_Construida': ['Area_Construida', 'Area Construida', 'Area Const', 'Area Privativa', 'Area Util', 'M2 Privativa'],
        'Area_Terreno': ['Area_Terreno', 'Area Terreno', 'Terreno', 'AREA_TERRENO', 'Area do Terreno'],
        'Quartos': ['Quartos', 'Dormitorios', 'QUARTOS', 'Dormitorios Total'],
        'Suites': ['Suites', 'SUITES', 'Suite', 'SUITE'],
        'Vagas': ['Vagas', 'Garagem', 'VAGAS', 'Vagas Garagem'],
        'Conservacao': ['Conservacao', 'CONSERVACAO', 'Estado de Conservacao', 'Conservacao Imovel'],
        'Padrao_Acabamento': ['Padrao_Acabamento', 'Padrao', 'Acabamento', 'PADRAO', 'Padrao de Acabamento'],
        'Setor_Urbano': ['Setor_Urbano', 'Setor Urbano', 'Setor', 'SETOR', 'SETOR_URBANO', 'Setor urbano'],
        'Data_Evento': ['Data_Evento', 'Data Evento', 'Data', 'DATA', 'Data do Evento'],
        'Evento': ['Evento', 'EVENTO', 'Fator Evento'],
        'Idade_Aparent': ['Idade Aparente', 'Idade_Aparent', 'Idade']
    }

    for padrao, sinonimos in colunas_possiveis.items():
        for col in df.columns:
            if col in sinonimos:
                df = df.rename(columns={col: padrao})

    if 'Preco' not in df.columns or 'Area_Construida' not in df.columns:
        st.error(f"Não conseguimos mapear as colunas essenciais nesta planilha. Cabeçalhos atuais: {list(df.columns)}")
    else:
        todas_vars_possiveis = ['Area_Construida', 'Area_Terreno', 'Quartos', 'Suites', 'Vagas', 'Conservacao', 'Padrao_Acabamento', 'Setor_Urbano', 'Data_Evento', 'Evento', 'Idade_Aparent']
        variaveis_independentes = [v for v in todas_vars_possiveis if v in df.columns]

        def limpar_para_numero_puro(valor):
            txt = str(valor).strip().replace('R$', '').replace(' ', '')
            if not txt or txt.lower() in ['nan', 'null', '']: return np.nan
            if ',' in txt and '.' in txt:
                if txt.find('.') < txt.find(','): txt = txt.replace('.', '')
                else: txt = txt.replace(',', '')
            if ',' in txt and '.' not in txt:
                txt = txt.replace(',', '.')
            txt = re.sub(r'[^\d.]', '', txt)
            try: return float(txt)
            except: return np.nan

        df['Preco'] = df['Preco'].apply(limpar_para_numero_puro)
        for col in variaveis_independentes:
            df[col] = df[col].apply(limpar_para_numero_puro)
            df[col] = df[col].fillna(df[col].median() if not df[col].isnull().all() else 1.0)

        df = df.dropna(subset=['Preco', 'Area_Construida'])

        # Painel Lateral
        st.sidebar.header("⚙️ Características do Imóvel")
        caracteristicas_avaliando = {}
        
        for var in variaveis_independentes:
            if var == 'Area_Construida':
                caracteristicas_avaliando[var] = st.sidebar.number_input("Área Construída (m²)", value=120.0, step=1.0)
            elif var == 'Area_Terreno':
                caracteristicas_avaliando[var] = st.sidebar.number_input("Área do Terreno (m²)", value=360.0, step=1.0)
            elif var in ['Quartos', 'Suites', 'Vagas']:
                label_nome = "Quartos" if var == 'Quartos' else ("Suítes" if var == 'Suites' else "Vagas de Garagem")
                caracteristicas_avaliando[var] = st.sidebar.slider(f"Quantidade de {label_nome}", 0, 5, 1 if var != 'Quartos' else 3)
            elif var in ['Conservacao', 'Padrao_Acabamento']:
                label = "Estado de Conservação" if var == 'Conservacao' else "Padrão de Acabamento"
                caracteristicas_avaliando[var] = st.sidebar.selectbox(label, [1, 2, 3], index=1, format_func=lambda x: {1:"Baixo/Regular", 2:"Médio/Bom", 3:"Alto/Excelente"}[x])
            elif var == 'Setor_Urbano':
                valor_inicial = float(df['Setor_Urbano'].median()) if len(df) > 0 else 500.0
                caracteristicas_avaliando[var] = st.sidebar.number_input("Setor Urbano (Fator)", min_value=0.0, max_value=5000.0, value=valor_inicial, step=10.0)
            elif var == 'Data_Evento':
                valor_data = float(df['Data_Evento'].median()) if len(df) > 0 else 1.0
                caracteristicas_avaliando[var] = st.sidebar.number_input("Data do Evento (Mês/Fator)", value=valor_data, step=1.0)
            elif var == 'Evento':
                caracteristicas_avaliando[var] = st.sidebar.number_input("Fator de Evento", value=1.0, step=0.05)
            elif var == 'Idade_Aparent':
                caracteristicas_avaliando[var] = st.sidebar.number_input("Idade Aparente (Anos)", value=10.0, step=1.0)

        if len(df) >= len(variaveis_independentes) + 2:
            X = df[variaveis_independentes]
            y = df['Preco']
            
            # Escalonamento estatístico estável
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            
            modelo = LinearRegression().fit(X_scaled, y.values)
            
            # Previsões usando dados na escala correta
            y_pred_todo
