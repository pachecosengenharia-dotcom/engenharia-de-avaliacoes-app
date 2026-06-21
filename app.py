import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import pdfplumber
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- Configuração da Página ---
st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# --- Funções de Apoio ---
def extrair_atributos_do_texto(texto):
    atributos = {}
    match_area = re.search(r'(?:Área|Área Total|Área Útil)\D*(\d+[\.,]?\d*)', texto, re.IGNORECASE)
    if match_area:
        atributos['area'] = float(match_area.group(1).replace(',', '.'))
    return atributos

def validar_dados_imovel(dados):
    if 'area' in dados and not (20 <= dados['area'] <= 2000):
        return False, f"Área de {dados['area']}m² fora dos limites aceitáveis (20-2000m²)."
    return True, None

def gerar_laudo_pdf(endereco, valor_medio, eq_str):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "Laudo Técnico de Avaliação Imobiliária")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Endereço: {endereco}")
    c.drawString(50, 750, f"Equação: {eq_str}")
    c.drawString(50, 730, f"Valor Médio Estimado: R$ {valor_medio:,.2f}")
    c.save()
    buffer.seek(0)
    return buffer

# --- Lógica Principal ---
arquivo_csv = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")

# --- Lógica de Modelagem Corrigida ---
if arquivo_csv is not None:
    # 1. Leitura
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    
    # 2. Definição do alvo e limpeza
    # O CSV está usando vírgula como decimal. Precisamos converter para ponto.
    col_alvo = 'Valor Unitário'
    
    # Selecionamos apenas as colunas que são numéricas para o modelo
    # Removemos colunas de texto (Endereço, Bairro, etc)
    df_clean = df.copy()
    
    # Função para limpar colunas numéricas que estão como texto (troca vírgula por ponto)
    def limpar_coluna(col):
        if col.dtype == 'object':
            return pd.to_numeric(col.astype(str).str.replace(',', '.'), errors='coerce')
        return col

    # Aplica a limpeza em todas as colunas
    for col in df_clean.columns:
        df_clean[col] = limpar_coluna(df_clean[col])
        
    # Remove colunas que não são úteis para o modelo (texto puro)
    # Lista de colunas que devem ser removidas do treinamento
    cols_drop = ['Endereço', 'Complemento', 'Bairro', 'Informante', 'Telefone', 'Evento']
    # Ajuste o nome das colunas conforme necessário (considerando o encoding)
    df_clean = df_clean.drop(columns=[c for c in cols_drop if c in df_clean.columns], errors='ignore')
    
    # Remove linhas com valores vazios (NaN)
    df_clean = df_clean.dropna()
    
    # 3. Modelagem
    features = [c for c in df_clean.columns if c != col_alvo]
    X = df_clean[features]
    y = df_clean[col_alvo]
    
    if X.empty or y.empty:
        st.error("Erro: Dados insuficientes ou inválidos após a limpeza.")
    else:
        modelo = LinearRegression().fit(X, y)
        st.success("Modelo treinado com sucesso!")
        
        # ... resto do seu código (inputs, cálculo, download) ...
    
    if tipo_input == "Via PDF":
        doc = st.sidebar.file_uploader("Subir Laudo", type="pdf")
        if doc:
            texto = pdfplumber.open(doc).pages[0].extract_text()
            extraidos = extrair_atributos_do_texto(texto)
            valido, msg = validar_dados_imovel(extraidos)
            if valido:
                for k, v in extraidos.items(): dados_imovel[k] = v
                st.sidebar.success("Dados validados!")
            else:
                st.sidebar.error(msg)
    else:
        for f in features:
            dados_imovel[f] = st.sidebar.number_input(f"{f}", value=float(df_clean[f].median()))
            
    # Cálculo e Exportação
    if st.button("Calcular Avaliação"):
        input_array = np.array([dados_imovel[f] for f in features]).reshape(1, -1)
        pred = modelo.predict(input_array)[0]
        st.metric("Valor Unitário Estimado", f"R$ {pred:,.2f}")
        
        pdf_gerado = gerar_laudo_pdf(endereco, pred, eq_str)
        st.download_button("📥 Baixar Laudo Completo", data=pdf_gerado, file_name="laudo.pdf")
else:
    st.info("Por favor, carregue a base de dados (CSV) na barra lateral.")
