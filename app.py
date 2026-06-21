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
# --- MODELAGEM ROBUSTA (CÓDIGO AJUSTADO PARA O SEU CSV) ---
if arquivo_csv is not None:
    # 1. Leitura do arquivo
    df = pd.read_csv(arquivo_csv, sep=";", encoding='latin-1')
    
    # 2. Definição das colunas numéricas (conforme o seu arquivo)
    cols_numericas = [
        'Área Privativa', 'Área do Terreno', 'Quartos', 'Evento', 
        'Padrão de Acabamento', 'Suite', 'Estado de Conservação', 
        'Idade Aparente', 'Setor urbano', 'Data do Evento', 'Valor Total'
    ]
    col_alvo = 'Valor Unitário'
    
    # 3. Função para limpar e converter números
    def converter_limpo(valor):
        # Transforma "2.150,53" em 2150.53 (trata vírgula e ponto)
        s = str(valor).replace('.', '').replace(',', '.')
        try:
            return float(s)
        except:
            return np.nan

    # 4. Criamos um novo DataFrame apenas com os dados limpos
    df_clean = pd.DataFrame()
    for col in cols_numericas + [col_alvo]:
        if col in df.columns:
            df_clean[col] = df[col].apply(converter_limpo)
    
    # Removemos linhas que ficaram com algum erro (NaN)
    df_clean = df_clean.dropna()

    # 5. Treinamento do Modelo
    if not df_clean.empty:
        X = df_clean[cols_numericas]
        y = df_clean[col_alvo]
        modelo = LinearRegression().fit(X, y)
        
        st.success("Modelo treinado com sucesso!")
        
        # Exibir importância das variáveis
        st.write("Coeficientes (Impacto de cada variável):")
        st.bar_chart(pd.Series(modelo.coef_, index=cols_numericas))
    else:
        st.error("Erro: Nenhum dado numérico válido foi encontrado. Verifique se o nome das colunas está correto.")
            
    # Cálculo e Exportação
    if st.button("Calcular Avaliação"):
        input_array = np.array([dados_imovel[f] for f in features]).reshape(1, -1)
        pred = modelo.predict(input_array)[0]
        st.metric("Valor Unitário Estimado", f"R$ {pred:,.2f}")
        
        pdf_gerado = gerar_laudo_pdf(endereco, pred, eq_str)
        st.download_button("📥 Baixar Laudo Completo", data=pdf_gerado, file_name="laudo.pdf")
else:
    st.info("Por favor, carregue a base de dados (CSV) na barra lateral.")
