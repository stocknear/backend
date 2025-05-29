import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT

from matplotlib import pyplot as plt
from io import BytesIO
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF




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
        # Header style
        self.header_style = ParagraphStyle(
            'CustomHeader',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#4472C4')
        )
        
        # Modified Subheader style for the "Overview" text itself
        # Properties like background color and full width will be handled by a Table
        self.subheader_style = ParagraphStyle(
            'CustomSubHeader',
            parent=self.styles['Normal'],  # Base on Normal for more direct control
            fontName='Helvetica-Bold',
            fontSize=16,                   # Adjusted for a typical section header look
            textColor=colors.HexColor('#2F5597'), # Dark blue text from screenshot
            leftIndent=10,                 # Padding from the left edge of the blue bar
            rightIndent=10,                # Padding from the right edge of the blue bar
            spaceBefore=0,                 # No extra space before paragraph within its container
            spaceAfter=0,                  # No extra space after paragraph within its container
            alignment=TA_LEFT,
            leading=18                     # Line spacing (fontSiz * 1.2 is a common value)
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
    
    def create_header(self, title="Stocknear", 
                      generated_date=None, page_info="Page 1 of 11"):
        """Create the report header"""
        if generated_date is None:
            generated_date = datetime.now().strftime("Generated on %b %d, %Y, %I:%M:%S %p %Z")
        
        # Create header table
        header_data = [
            [f"🔓 {title}", generated_date, page_info]
        ]
        
        header_table = Table(header_data, colWidths=[3.7*inch, 2.5*inch, 1.5*inch])
        header_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (0,0), 14),
            ('FONTSIZE', (1,0), (-1,-1), 8),
            ('TEXTCOLOR', (0,0), (0,0), colors.HexColor('#4472C4')),
            ('TEXTCOLOR', (1,0), (-1,-1), colors.grey),
            ('ALIGN', (0,0), (0,0), 'LEFT'),
            ('ALIGN', (1,0), (1,0), 'CENTER'),
            ('ALIGN', (2,0), (2,0), 'RIGHT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ]))
        
        self.story.append(header_table)
        self.story.append(Spacer(1, 12))
    
    def create_overview_section(self):
        overview_paragraph = Paragraph("Overview", self.subheader_style)

        # Define the background color for the bar (light blue from screenshot: #DDEBF7)
        background_color = colors.HexColor('#DDEBF7')

        # Create a table to act as the full-width colored bar
        # The table will have one cell containing the overview_paragraph
        # self.doc.width gives the available width between page margins
        overview_section_table = Table([[overview_paragraph]], colWidths=[self.doc.width])

        overview_section_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), background_color),
            ('VALIGN', (0,0), (0,0), 'MIDDLE'),
            # Cell padding to create the bar's height.
            # The paragraph's own leftIndent/rightIndent handles text padding horizontally.
            ('LEFTPADDING', (0,0), (0,0), 0), 
            ('RIGHTPADDING', (0,0), (0,0), 0),
            ('TOPPADDING', (0,0), (0,0), 8),   # Padding above the text
            ('BOTTOMPADDING', (0,0), (0,0), 8) # Padding below the text
        ]))
        
        self.story.append(overview_section_table)
        # Add space after the entire overview bar, similar to original spaceAfter in subheader_style
        self.story.append(Spacer(1, 20)) 
    
    def create_company_info(self, symbol, company_name):
        """Create company name and thin divider"""
        company_text = f"{symbol} – {company_name}"
        self.story.append(Paragraph(company_text, self.company_style))
        # thin light-grey line
        self.story.append(Spacer(1, 6))
        self.story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        self.story.append(Spacer(1, 20))

    def create_financial_metrics_table(self, m):
        """4×4 table with label in grey / value in black bold"""
        # prepare a little inline style for label and value
        cell_template = "<font color='#888888' size=9>{label}</font><br/><font color='black' size=10>{value}</font>"

        # list your metrics in order
        rows = [
            [("Market Cap", m['market_cap']),     ("Industry", m['industry']),
             ("EPS (TTM)", m['eps_ttm']),         ("P/E (TTM)", m['pe_ttm'])],

            [("Div & Yield", m['div_yield']),     ("FCF Payout Ratio", m['fcf_payout_ratio']),
             ("P/S (TTM)", m['ps_ttm']),          ("P/B", m['pb'])],

            [("Shares Outstanding", m['shares_outstanding']), ("Ex-Dividend", m['ex_dividend']),
             ("Next Earnings", m['next_earnings']),           ("Forward P/E", m['forward_pe'])],

            [("Payout Ratio", m['payout_ratio']), ("P/FCF (TTM)", m['p_fcf_ttm']),
             ("FCF Yield", m['fcf_yield']),       ("Earnings Yield", m['earnings_yield'])]
        ]

        # build table_data of Paragraphs
        table_data = []
        for row in rows:
            table_data.append([
                Paragraph(cell_template.format(label=lbl, value=val), self.styles['Normal'])
                for lbl, val in row
            ])

        tbl = Table(table_data, colWidths=[1.8*inch]*4)
        tbl.setStyle(TableStyle([
            ('BACKGROUND',     (0,0), (-1,-1), colors.HexColor('#F5F5F5')),
            ('VALIGN',         (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING',(0,0), (-1,-1), 6),
            ('TOPPADDING',  (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
            # no grid or lines
        ]))

        self.story.append(tbl)
        self.story.append(Spacer(1, 30))

    
    def create_description_section(self, description_text):
        """Create the Description section"""
        desc_header = Paragraph("Description", self.company_style) # Using company_style for "Description" title
        self.story.append(desc_header)
        self.story.append(Spacer(1, 3))
        
        description = Paragraph(description_text, self.desc_style)
        self.story.append(description)
    
    def add_svg_line_chart(self):
        """Add a high-quality vector line chart using SVG"""
        # Generate dummy data
        x = list(range(10))
        y = [2, 3, 5, 6, 4, 7, 8, 7, 9, 10]

        # Create plot
        fig, ax = plt.subplots(figsize=(6, 3))  # Inches
        ax.plot(x, y, marker='o')
        ax.set_title("High-Quality SVG Line Chart")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)

        # Save SVG to memory buffer
        svg_buffer = BytesIO()
        plt.tight_layout()
        fig.savefig(svg_buffer, format='svg')
        plt.close(fig)
        svg_buffer.seek(0)

        # Convert SVG to ReportLab Drawing
        drawing = svg2rlg(svg_buffer)
        self.story.append(Spacer(1, 20))
        self.story.append(drawing)
        self.story.append(Spacer(1, 20))


    def generate_report(self, company_data):
        """Generate the complete report"""
        self.create_header()
        self.create_overview_section()
        self.create_company_info(company_data['symbol'], company_data['company_name'])
        self.create_financial_metrics_table(company_data['metrics'])
        self.create_description_section(company_data['description'])
        self.add_svg_line_chart()
        self.doc.build(self.story)
        print(f"Report generated successfully: {self.filename}")


# Example usage
def create_apple_report():
    """Create a sample Apple Inc. report"""
    
    # Sample data matching the image
    apple_data = {
        'symbol': 'AAPL',
        'company_name': 'Apple Inc',
        'metrics': {
            'market_cap': '$3.26T',
            'industry': 'Technology',
            'eps_ttm': '$6.49',
            'pe_ttm': '32.46',
            'div_yield': '$1.00 (0.46%)',
            'fcf_payout_ratio': '14.65%',
            'ps_ttm': '8.54',
            'pb': '43.92',
            'shares_outstanding': '15.33B',
            'ex_dividend': '2024-05-10',
            'next_earnings': '08-01',
            'forward_pe': '31.27',
            'payout_ratio': '14.87%',
            'p_fcf_ttm': '31.97',
            'fcf_yield': '3.13%',
            'earnings_yield': '3.08%'
        },
        'description': '''Apple Inc., with a market capitalization of around $3.26 trillion USD, stands as a titan in the technology sector. As one of the Big Four tech companies, Apple's product line extends from the ubiquitous iPhone to services like iCloud and Apple Music. The company's extensive hardware range includes iPads, Macs, iPods, Apple Watches, and more, complemented by software such as iOS and professional applications like Final Cut Pro. Their innovation in consumer electronics and digital services solidifies their position as a key player in the global market.'''
    }
    
    # Create report
    report = StockAnalysisReport("apple_analysis_report.pdf")
    report.generate_report(apple_data)

create_apple_report()