def gerar_pdf(regiao, pred_unit, pred_total, minimo, maximo, eq_str, features):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # Título
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 820, "LAUDO TÉCNICO DE AVALIAÇÃO IMOBILIÁRIA")
    
    c.setFont("Helvetica", 12)
    c.drawString(50, 790, f"Região Analisada: {regiao}")
    
    # Equação
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 760, "Equação do Modelo:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 745, eq_str)
    
    # Valores
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 715, "Resultados da Avaliação:")
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"V.U. Mínimo: R$ {minimo:,.2f} | Médio: R$ {pred_unit:,.2f} | Máximo: R$ {maximo:,.2f}")
    c.drawString(50, 685, f"Valor Total Estimado: R$ {pred_total:,.2f}")
    
    # Variáveis consideradas
    c.drawString(50, 655, "Variáveis consideradas no modelo:")
    c.drawString(50, 640, ", ".join(features))
    
    c.save()
    buffer.seek(0)
    return buffer
