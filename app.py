def gerar_pdf(pred_unit, pred_total, minimo, maximo, eq_str, features):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    # Título
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "LAUDO TÉCNICO DE AVALIAÇÃO IMOBILIÁRIA")
    c.line(50, 790, 550, 790)
    
    # Detalhes
    c.setFont("Helvetica", 12)
    y = 750
    c.drawString(50, y, f"Equação do Modelo: {eq_str}")
    y -= 30
    c.drawString(50, y, "RESULTADOS DA AVALIAÇÃO:")
    y -= 25
    c.drawString(70, y, f"- V.U. Mínimo (95%): R$ {minimo:,.2f}")
    y -= 20
    c.drawString(70, y, f"- V.U. Médio Estimado: R$ {pred_unit:,.2f}")
    y -= 20
    c.drawString(70, y, f"- V.U. Máximo (95%): R$ {maximo:,.2f}")
    y -= 30
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"VALOR TOTAL ESTIMADO: R$ {pred_total:,.2f}")
    
    # Rodapé Técnico
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 100, "Variáveis consideradas: " + ", ".join(features))
    c.drawString(50, 80, "Laudo gerado automaticamente para fins de avaliação de mercado.")
    
    c.save()
    buffer.seek(0)
    return buffer
