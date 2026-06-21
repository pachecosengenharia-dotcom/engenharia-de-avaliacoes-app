import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import pdfplumber
from sklearn.linear_model import LinearRegression
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
# --- LOGICA DE TRATAMENTO ANTES DO FIT ---
# 1. Garante que X e y sejam numéricos e remove linhas com valores faltantes
df_clean = df.copy()

# Converte todas as colunas de X para numérico, forçando erro para NaN
for col in features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Remove linhas onde qualquer dado essencial é nulo
df_clean = df_clean.dropna(subset=features + [col_alvo])

# Atualiza X e y com os dados limpos
X = df_clean[features]
y = df_clean[col_alvo]

# 2. Validação extra: checa se ainda restaram dados após a limpeza
if not X.empty and not y.empty:
    modelo = LinearRegression().fit(X, y)
else:
    st.error("Erro: A base de dados contém valores não numéricos ou está vazia após a limpeza.")
    st.stop() # Para o app aqui e evita o erro
# --- 1. FUNÇÕES AUXILIARES DE ENGENHARIA ---
def extrair_atributos_do_texto(texto):
    """Extrai área e dados básicos via Regex."""
    atributos = {}
    match_area = re.search(r'(?:Área|Área Total|Área Útil)\D*(\d+[\.,]?\d*)', texto, re.IGNORECASE)
    if match_area:
        atributos['area'] = float(match_area.group(1).replace(',', '.'))
    return atributos

def validar_dados_imovel(dados):
    """Validação técnica para evitar erros de precificação."""
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

# --- 2. INTERFACE E LÓGICA ---
st.set_page_config(layout="wide")
st.title("📊 AVM - Engenharia de Avaliações")

# Upload do CSV de base
arquivo_csv = st.sidebar.file_uploader("Carregar Base de Dados (CSV)", type="csv")
if arquivo_csv:
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    
    # [Lógica do Modelo - Simplificada para o exemplo]
    col_alvo = 'Valor Unitário'
    features = [c for c in df.columns if c != col_alvo]
    X, y = df[features], df[col_alvo]
    modelo = LinearRegression().fit(X, y)
    eq_str = " + ".join([f"{c:.2f}*{n}" for n, c in zip(features, modelo.coef_)])

    # Entrada de Dados
    st.sidebar.subheader("📍 Entrada do Imóvel")
    tipo_input = st.sidebar.radio("Fonte de Dados:", ["Manual", "Via PDF"])
    
    dados_imovel = {n: 0.0 for n in features}
    endereco = st.sidebar.text_input("Endereço do Imóvel:")

    if tipo_input == "Via PDF":
        doc = st.sidebar.file_uploader("Subir Laudo/Escritura", type="pdf")
        if doc:
            texto = pdfplumber.open(doc).pages[0].extract_text()
            extraidos = extrair_atributos_do_texto(texto)
            valido, msg = validar_dados_imovel(extraidos)
            if valido:
                st.sidebar.success("Dados validados com sucesso!")
                for k, v in extraidos.items(): dados_imovel[k] = v
            else:
                st.sidebar.error(msg)
    else:
        for f in features:
            dados_imovel[f] = st.sidebar.number_input(f"{f}", value=float(df[f].median()))

    # Execução
    if st.button("Calcular Avaliação"):
        input_array = np.array([dados_imovel[f] for f in features]).reshape(1, -1)
        pred = modelo.predict(input_array)[0]
        
        st.metric("Valor Unitário Estimado", f"R$ {pred:,.2f}")
        
        pdf_gerado = gerar_laudo_pdf(endereco, pred, eq_str)
        st.download_button("📥 Baixar Laudo Completo", data=pdf_gerado, file_name="laudo.pdf")

else:
    st.info("Por favor, carregue o arquivo CSV de amostras para iniciar.")
