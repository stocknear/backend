import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib.pyplot as plt
import io
import os

def get_dummy_data(days: int = 30) -> pd.DataFrame:
    """
    Generate a DataFrame with 'days' worth of dummy OHLCV data.
    """
    # Create a date index for the past 'days' calendar days
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq='B')
    
    # Generate a rising price series with small random noise
    np.random.seed(42)
    base = np.linspace(100, 120, days)  # price from 100 → 120
    noise = np.random.normal(scale=1.0, size=days)
    close = base + noise
    
    # Open/High/Low derived around the close
    open_  = close + np.random.normal(scale=0.5, size=days)
    high   = np.maximum(open_, close) + np.abs(np.random.normal(scale=0.5, size=days))
    low    = np.minimum(open_, close) - np.abs(np.random.normal(scale=0.5, size=days))
    
    # Volume: random ints in a reasonable range
    volume = np.random.randint(1_000_000, 5_000_000, size=days)
    
    df = pd.DataFrame({
        'Open':  open_,
        'High':  high,
        'Low':   low,
        'Close': close,
        'Volume': volume
    }, index=idx)
    return df

def create_price_chart(data: pd.DataFrame, label: str) -> io.BytesIO:
    """Generate a price chart and return it as PNG in memory."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index, data['Close'], label=f"{label} Close")
    ax.set_title(f"{label} Stock Price (Dummy Data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def create_stock_report_pdf(
    data: pd.DataFrame,
    label: str,
    output_filename: str = "dummy_stock_report.pdf"
):
    """Generate a PDF report with a table and embedded chart using the provided DataFrame."""
    doc = SimpleDocTemplate(output_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph(f"Stock Report: {label} (Dummy Data)", styles['Title']))
    story.append(Spacer(1, 12))

    if not data.empty:
        latest = data.iloc[-1]
        date_str = latest.name.strftime("%Y-%m-%d")

        # Build table data
        table_data = [
            ["Field", "Value"],
            ["Date", date_str],
            ["Open", f"{latest['Open']:.2f}"],
            ["High", f"{latest['High']:.2f}"],
            ["Low", f"{latest['Low']:.2f}"],
            ["Close", f"{latest['Close']:.2f}"],
            ["Volume", f"{int(latest['Volume']):,}"],
        ]
        tbl = Table(table_data, colWidths=[100, 100], hAlign='LEFT')
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 24))

        # Embed chart
        chart_buf = create_price_chart(data, label)
        img = Image(chart_buf, width=400, height=200)
        story.append(img)
    else:
        story.append(Paragraph("No data available.", styles['Normal']))

    # Build the PDF
    doc.build(story)
    print(f"Dummy report saved to {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    label = "TEST"
    dummy_df = get_dummy_data(days=30)
    create_stock_report_pdf(dummy_df, label)
