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
# --- Carregamento e Limpeza Robusta ---
if arquivo_csv is not None:
    # 1. Leitura forçada
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    
    # 2. Identificação dinâmica do alvo (buscando pelo que termina com 'Unitário')
    col_alvo = [c for c in df.columns if 'Unit' in c][0]
    
    # 3. Limpeza rigorosa
    df_clean = df.copy()
    
    # Converte tudo para numérico, tratando a vírgula como ponto
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Remove pontos de milhar, troca vírgula por ponto
            df_clean[col] = df_clean[col].astype(str).str.replace('.', '').str.replace(',', '.')
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove colunas que ficaram com tudo NaN (como texto puro que não virou número)
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # Remove linhas com valores vazios na coluna alvo
    df_clean = df_clean.dropna(subset=[col_alvo])
    
    # 4. Modelagem
    # 3. Modelagem Robusta (Substitua por este bloco)
    col_alvo = 'Valor Unitário'
    
    # 1. Converter tudo para numérico, forçando erro para o que não for número (vira NaN)
    df_numerico = df.apply(pd.to_numeric, errors='coerce')
    
    # 2. Adicionar o Valor Unitário de volta (pois ele tem vírgulas e foi convertido acima)
    df[col_alvo] = df[col_alvo].astype(str).str.replace(',', '.').astype(float)
    df_numerico[col_alvo] = df[col_alvo]
    
    # 3. Remover colunas que ficaram vazias (que eram texto puro)
    df_final = df_numerico.dropna(axis=1, how='all')
    
    # 4. Remover linhas vazias
    df_final = df_final.dropna()
    
    # 5. Definir X e Y
    features = [c for c in df_final.columns if c != col_alvo]
    X = df_final[features]
    y = df_final[col_alvo]
    
    # 6. Fit
    modelo = LinearRegression().fit(X, y)
    st.success(f"Modelo treinado com sucesso com as variáveis: {features}")
    
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
