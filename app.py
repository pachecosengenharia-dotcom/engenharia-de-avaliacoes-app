def gerar_pdf(regiao, pred_unit, pred_total, minimo, maximo, eq_str, features, n_amostras):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Cabeçalho
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 820, "LAUDO TÉCNICO DE AVALIAÇÃO IMOBILIÁRIA")
    c.line(50, 810, 550, 810)
    
    # Dados da Análise
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 780, f"Região Analisada: {regiao}")
    c.drawString(50, 765, f"Tamanho da Amostra: {n_amostras} imóveis")
    
    # Equação do Modelo
    c.drawString(50, 735, "Equação de Regressão Linear:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 720, eq_str)
    
    # Resultados
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 690, "Resultado da Estimativa:")
    c.setFont("Helvetica", 12)
    c.drawString(50, 675, f"Valor Unitário (Médio): R$ {pred_unit:,.2f} / m²")
    c.drawString(50, 660, f"Intervalo de Confiança: R$ {minimo:,.2f} a R$ {maximo:,.2f}")
    c.drawString(50, 645, f"VALOR TOTAL ESTIMADO: R$ {pred_total:,.2f}")
    
    # Variáveis consideradas
    c.drawString(50, 615, "Variáveis utilizadas no modelo:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 600, ", ".join(features))
    
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Este laudo foi gerado automaticamente por sistema de Engenharia Olfativa e Avaliações.")
    
    c.save()
    buffer.seek(0)
    return buffer
