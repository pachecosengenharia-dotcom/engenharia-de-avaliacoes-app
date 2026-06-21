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
import os

st.set_page_config(page_title="Engenharia de Avaliações", layout="wide")

st.title("📊 Sistema Profissional de Engenharia de Avaliações")
st.markdown("Selecione uma região salva no seu repositório ou insira uma nova planilha de mercado.")

# --- GERENCIAMENTO DE BANCO DE DADOS INTELIGENTE ---
st.sidebar.header("📁 Base de Dados")

# Procura arquivos .csv na raiz do projeto e na pasta 'dados'
planilhas_disponiveis = []
for f in os.listdir('.'):
    if f.endswith('.csv'):
        planilhas_disponiveis.append(f)

if os.path.exists("dados"):
    for f in os.listdir("dados"):
        if f.endswith('.csv') and f not in planilhas_disponiveis:
            planilhas_disponiveis.append(os.path.join("dados", f))

# Menu seletor de regiões
fonte_dados = st.sidebar.selectbox(
    "Escolha a Região/Município",
    options=["Selecionar região salva..."] + planilhas_disponiveis + ["Fazer Upload Manual (.csv)"]
)

df = None

# Função de leitura com detecção automática de codificação e separador
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

# Carregamento baseado na escolha
if fonte_dados and fonte_dados != "Selecionar região salva..." and fonte_dados != "Fazer Upload Manual (.csv)":
    df = carregar_csv_com_seguranca(fonte_dados)
    if df is not None:
        nome_exibicao = os.path.basename(fonte_dados).replace('.csv', '')
        st.sidebar.success(f"📍 Região carregada: {nome_exibicao}")

elif fonte_dados == "Fazer Upload Manual (.csv)":
    arquivo_upload = st.sidebar.file_uploader("Arraste sua planilha (.csv)", type=["csv"])
    if arquivo_upload:
        df = carregar_csv_com_seguranca(arquivo_upload)

# --- PROCESSAMENTO TÉCNICO DOS DADOS ---
if df is not None:
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.astype(str).str.strip()

    # Mapeamento robusto para evitar KeyError de qualquer tipo
    colunas_possiveis = {
        'Preco': ['Preco', 'Preço', 'Valor', 'Vlr', 'PRECO', 'PREÇO', 'Valor Total', 'Valor_Total'],
        'Area_Construida': ['Area_Construida', 'Área_Construída', 'Area Construida', 'Área Construída', 'Area Const', 'Área Const', 'AREA_CONSTRUIDA', 'Área Privativa', 'Area Privativa'],
        'Area_Terreno': ['Area_Terreno', 'Área_Terreno', 'Area Terreno', 'Terreno', 'AREA_TERRENO', 'Área do Terreno', 'Area do Terreno'],
        'Quartos': ['Quartos', 'Dormitorios', 'Dormitórios', 'QUARTOS'],
        'Suites': ['Suites', 'Suítes', 'SUITES', 'Suite', 'Suíte', 'SUITE'],
        'Vagas': ['Vagas', 'Garagem', 'VAGAS'],
        'Conservacao': ['Conservacao', 'Conservação', 'CONSERVACAO', 'Estado de Conservação', 'Estado de Conservacao'],
        'Padrao_Acabamento': ['Padrao_Acabamento', 'Padrão_Acabamento', 'Padrao', 'Padrão', 'Acabamento', 'PADRAO', 'Padrão de Acabamento', 'Padrao de Acabamento'],
        'Setor_Urbano': ['Setor_Urbano', 'Setor Urbano', 'Setor', 'SETOR', 'SETOR_URBANO', 'Setor urbano'],
        'Data_Evento': ['Data_Evento', 'Data Evento', 'Data', 'DATA', 'Data do Evento'],
        'Evento': ['Evento', 'EVENTO']
    }

    # Renomeia as colunas encontradas com segurança
    for padrao, sinonimos in colunas_possiveis.items():
        for col in df.columns:
            if col in sinonimos:
                df = df.rename(columns={col: padrao})

    # Verificação obrigatória apenas para os dois pilares
