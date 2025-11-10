import os
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
# MODIFIED: Added Image to the import below
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# These imports are for the SVG chart functionality, which is currently commented out in your example
from matplotlib import pyplot as plt
from io import BytesIO
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

text_primary = "#000"


class StockAnalysisReport:
    def __init__(self, filename="stock_analysis_report.pdf"):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=letter,
                                     rightMargin=40, leftMargin=40,
                                     topMargin=10, bottomMargin=18)
        self.styles = getSampleStyleSheet()
        self.story = []
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Create custom styles for the report"""
        # Header style (Original, seems unused for the main title "Stocknear" now)
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=22,
            spaceAfter=30,
            textColor=colors.HexColor(text_primary)
        )

        # Modified Subheader style for the "Overview" text itself
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=colors.HexColor('#2F5597'),
            leftIndent=10,
            rightIndent=10,
            spaceBefore=0,
            spaceAfter=0,
            alignment=TA_LEFT,
            leading=18
        )

        # Company name style
        self.company_style = ParagraphStyle(
            'CompanyName',
            parent=self.styles['Heading1'],
            fontSize=12,
            spaceAfter=0,
            textColor="#0A0708",
        )

        # Description style
        self.desc_style = ParagraphStyle(
            'Description',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            textColor="#0A0708",
            alignment=TA_LEFT
        )

    def create_header(self, title="Stocknear", logo_path=None, generated_date=None, page_info="Page 1 of 11"):
        if generated_date is None:
            generated_date = datetime.now().strftime("Generated on %b %d, %Y, %I:%M:%S %p %Z")

        logo_title_elements = []

        if logo_path:
            try:
                logo_image = Image(logo_path, width=0.4*inch, height=0.4*inch)
                logo_title_elements.append(logo_image)
            except Exception as e:
                print(f"Error loading logo: {e}")

        title_text_style = ParagraphStyle(
            'HeaderTitleText',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=16,
            textColor=colors.HexColor(text_primary),
            leading=16
        )
        title_paragraph = Paragraph(title, title_text_style)
        logo_title_elements.append(title_paragraph)

        nested_logo_title_table = Table(
            [logo_title_elements],
            colWidths=[0.4*inch + 0.05*inch, None] if logo_path else [None],
            spaceAfter=0,
            spaceBefore=0
        )

        nested_logo_title_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))

        header_data = [
            [nested_logo_title_table, generated_date, page_info]
        ]

        # --- MODIFIED COLUMN WIDTHS FOR CLOSER ALIGNMENT ---
        # Reduce the width of the second column (generated_date)
        # and adjust the third column (page_info) if necessary.
        # Experiment with these values to find what looks best with your content.
        header_table = Table(header_data, colWidths=[3.7*inch, 2.0*inch, 2.0*inch]) # Example adjustment
        # You could also use: colWidths=[3.7*inch, None, None] if you want them to share remaining space
        # or calculate based on text length.

        header_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (1,0), (-1,-1), 8),
            ('TEXTCOLOR', (1,0), (-1,-1), colors.grey),
            ('ALIGN', (0,0), (0,0), 'LEFT'),
            ('ALIGN', (1,0), (1,0), 'CENTER'), # Center the date
            ('ALIGN', (2,0), (2,0), 'RIGHT'), # Right align the page info
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ]))

        self.story.append(header_table)
        self.story.append(Spacer(1, 12))


    def create_overview_section(self):
        overview_paragraph = Paragraph("Overview", self.subheader_style)
        background_color = colors.HexColor('#DDEBF7')
        overview_section_table = Table([[overview_paragraph]], colWidths=[self.doc.width])
        overview_section_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), background_color),
            ('VALIGN', (0,0), (0,0), 'MIDDLE'),
            ('LEFTPADDING', (0,0), (0,0), 0),
            ('RIGHTPADDING', (0,0), (0,0), 0),
            ('TOPPADDING', (0,0), (0,0), 8),
            ('BOTTOMPADDING', (0,0), (0,0), 8)
        ]))
        self.story.append(overview_section_table)
        self.story.append(Spacer(1, 20))

    def create_company_info(self, symbol, company_name):
        company_text = f"{symbol} – {company_name}"
        self.story.append(Paragraph(company_text, self.company_style))
        self.story.append(Spacer(1, 6))
        self.story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        self.story.append(Spacer(1, 20))

    def create_financial_metrics_table(self, m):
        cell_template = "<font color='#888888' size=9>{label}</font><br/><font color='black' size=10>{value}</font>"
        rows = [
            [("Market Cap", m['market_cap']), ("Industry", m['industry']),
             ("EPS (TTM)", m['eps_ttm']), ("P/E (TTM)", m['pe_ttm'])],
            [("Div & Yield", m['div_yield']), ("FCF Payout Ratio", m['fcf_payout_ratio']),
             ("P/S (TTM)", m['ps_ttm']), ("P/B", m['pb'])],
            [("Shares Outstanding", m['shares_outstanding']), ("Ex-Dividend", m['ex_dividend']),
             ("Next Earnings", m['next_earnings']), ("Forward P/E", m['forward_pe'])],
            [("Payout Ratio", m['payout_ratio']), ("P/FCF (TTM)", m['p_fcf_ttm']),
             ("FCF Yield", m['fcf_yield']), ("Earnings Yield", m['earnings_yield'])]
        ]
        table_data = []
        for row in rows:
            table_data.append([
                Paragraph(cell_template.format(label=lbl, value=val), self.styles['Normal'])
                for lbl, val in row
            ])
        tbl = Table(table_data, colWidths=[1.8*inch]*4)
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F5F5F5')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING',(0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
        ]))
        self.story.append(tbl)
        self.story.append(Spacer(1, 30))

    def create_description_section(self, description_text):
        desc_header = Paragraph("Description", self.company_style)
        self.story.append(desc_header)
        self.story.append(Spacer(0, 3))
        desc_with_bg_style = ParagraphStyle(
            'DescriptionWithBG',
            parent=self.desc_style,
            backColor=colors.HexColor('#F5F5F5'),
            spaceBefore=6,
            spaceAfter=6,
            leftIndent=6,
            rightIndent=6,
            borderPadding=10
        )
        description = Paragraph(description_text, desc_with_bg_style)
        self.story.append(description)

    def add_svg_line_chart(self):
        """Add a high-quality vector line chart using SVG"""
        x = list(range(10))
        y = [2, 3, 5, 6, 4, 7, 8, 7, 9, 10]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, y, marker='o')
        ax.set_title("High-Quality SVG Line Chart")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        svg_buffer = BytesIO()
        plt.tight_layout()
        fig.savefig(svg_buffer, format='svg')
        plt.close(fig)
        svg_buffer.seek(0)
        drawing = svg2rlg(svg_buffer)
        self.story.append(Spacer(1, 20))
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))


    def create_insights_section(self):
        """Draw the Insights panel (donut + legend + returns vs SPY) with dummy data."""
        # --- Dummy data for the donut chart ---
        labels = ['Very Good', 'Good', 'Average', 'Bad', 'Very Bad']
        sizes  = [39,        17,     33,        8,     3]
        colors = ['#2E8B57', '#90EE90', '#FFA500', '#FF6347', '#8B0000']

        # --- Dummy data for Returns vs SPY ---
        dates = np.arange('2024-06', '2025-06', dtype='datetime64[M]')
        x = np.arange(len(dates))
        spy = np.cumsum(np.random.randn(len(x)) * 1.2 + 1)  # SPY
        aapl = np.cumsum(np.random.randn(len(x)) * 1.5 + 0.8)  # AAPL

        # --- Create the figure ---
        fig = plt.figure(figsize=(8, 4.5))
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], width_ratios=[1, 1, 1])

        # Donut chart
        ax0 = fig.add_subplot(grid[0, 0])
        wedges, _ = ax0.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.4))
        ax0.set_aspect('equal')
        ax0.text(0, 0, "3.81\nGood", ha='center', va='center', fontsize=14, weight='bold')
        ax0.set_title("Insights", pad=10, fontsize=12)

        # Legend as a table
        ax1 = fig.add_subplot(grid[0, 1:])
        ax1.axis('off')
        # build legend table data
        legend_data = [[f"{labels[i]} ({sizes[i]}%)", '○'] for i in range(len(labels))]
        table = ax1.table(
            cellText=legend_data,
            colWidths=[0.6, 0.1],
            cellLoc='left',
            colLoc='left',
            cellColours=[['white','white']]*len(labels)
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

        # Returns vs SPY area chart
        ax2 = fig.add_subplot(grid[1, :])
        ax2.fill_between(x, spy, color='blue', alpha=0.3, label='SPY')
        ax2.fill_between(x, aapl, color='green', alpha=0.3, label='AAPL')
        ax2.plot(x, spy, color='blue')
        ax2.plot(x, aapl, color='green')
        ax2.set_xticks(x[::2])
        ax2.set_xticklabels([str(d)[:7] for d in dates[::2]], rotation=45, fontsize=8)
        ax2.set_ylabel('Return %')
        ax2.set_title('Returns vs SPY', pad=10, fontsize=12)
        ax2.legend(fontsize=8, loc='upper left')
        ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()

        # --- Render to PNG buffer and insert into story ---
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image(buf, width=self.doc.width, height=4*inch * (8/8))  # adjust height to taste
        self.story.append(Spacer(1, 20))
        self.story.append(img)
        self.story.append(Spacer(1, 20))

    def generate_report(self, company_data, logo_path="your_logo.png"):
        """Generate the complete report"""
        self.create_header(logo_path=logo_path)
        self.create_overview_section()
        self.create_company_info(company_data['symbol'], company_data['company_name'])
        self.create_financial_metrics_table(company_data['metrics'])
        self.create_description_section(company_data['description'])
        # — NEW: insert the Insights panel here —
        self.create_insights_section()
        #self.add_svg_line_chart()
        self.doc.build(self.story)
        print(f"Report generated successfully: {self.filename}")

# Example usage
def create_apple_report():
    """Create a sample Apple Inc. report"""
    apple_data = {
        'symbol': 'AAPL',
        'company_name': 'Apple Inc',
        'metrics': {
            'market_cap': '$3.26T', 'industry': 'Technology', 'eps_ttm': '$6.49', 'pe_ttm': '32.46',
            'div_yield': '$1.00 (0.46%)', 'fcf_payout_ratio': '14.65%', 'ps_ttm': '8.54', 'pb': '43.92',
            'shares_outstanding': '15.33B', 'ex_dividend': '2024-05-10', 'next_earnings': '08-01', 'forward_pe': '31.27',
            'payout_ratio': '14.87%', 'p_fcf_ttm': '31.97', 'fcf_yield': '3.13%', 'earnings_yield': '3.08%'
        },
        'description': '''Apple Inc., with a market capitalization of around $3.26 trillion USD, stands as a titan in the technology sector. As one of the Big Four tech companies, Apple's product line extends from the ubiquitous iPhone to services like iCloud and Apple Music. The company's extensive hardware range includes iPads, Macs, iPods, Apple Watches, and more, complemented by software such as iOS and professional applications like Final Cut Pro. Their innovation in consumer electronics and digital services solidifies their position as a key player in the global market.'''
    }

    report = StockAnalysisReport("apple_analysis_report.pdf")
    # IMPORTANT: Ensure "your_actual_logo.png" exists or provide the correct path.
    # For testing, you might create a dummy PNG file named "your_actual_logo.png"
    # in the same directory as your script.
    report.generate_report(apple_data, logo_path="img/logo.png") # Pass your logo path here


if __name__ == '__main__':
    create_apple_report()