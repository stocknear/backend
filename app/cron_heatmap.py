import pandas as pd
import plotly.express as px
import requests
import orjson

def get_spy_heatmap():
    # Load stock screener data
    with open(f"json/stock-screener/data.json", 'rb') as file:
        stock_screener_data = orjson.loads(file.read())
    stock_screener_data_dict = {item['symbol']: item for item in stock_screener_data}

    with open(f"json/etf/holding/SPY.json","rb") as file:
        data = orjson.loads(file.read())['holdings']

    for item in data:
        try:
            item['marketCap'] = stock_screener_data_dict[item['symbol']]['marketCap']
            item['sector'] = stock_screener_data_dict[item['symbol']]['sector']
            item['industry'] = stock_screener_data_dict[item['symbol']]['industry']
            item['change1W'] = stock_screener_data_dict[item['symbol']]['change1W']
            item['change1M'] = stock_screener_data_dict[item['symbol']]['change1M']
            item['change3M'] = stock_screener_data_dict[item['symbol']]['change3M']
            item['change6M'] = stock_screener_data_dict[item['symbol']]['change6M']
            item['change1Y'] = stock_screener_data_dict[item['symbol']]['change1Y']
            item['change3Y'] = stock_screener_data_dict[item['symbol']]['change3Y']
        except:
            pass

    # Create DataFrame
    df = pd.DataFrame(data)
    # Convert relevant columns to numeric types
    df["marketCap"] = pd.to_numeric(df["marketCap"])
   
    # Drop rows where the marketCap == 0
    df = df[df["marketCap"] > 0]
    
    # Fix None values in sector and industry to prevent treemap errors
    df['sector'] = df['sector'].fillna('Other')
    df['industry'] = df['industry'].fillna('Other')
    
    # Also drop rows where sector or industry are still None after fillna
    df = df.dropna(subset=['sector', 'industry'])
    
    return df

def create_treemap(time_period):
    save_html = True

    df = get_spy_heatmap()

    if (time_period == '1D'):
        change_percent = 'changesPercentage'
        range_color = (-3,3)
    elif (time_period == '1W'):
        change_percent = 'change1W'
        range_color = (-5,5)
    elif (time_period == '1M'):
        change_percent = 'change1M'
        range_color = (-20,20)
    elif (time_period == '3M'):
        change_percent = 'change3M'
        range_color = (-30,30)
    elif (time_period == '6M'):
        change_percent = 'change6M'
        range_color = (-50,50)
    elif (time_period == '1Y'):
        change_percent = 'change1Y'
        range_color = (-100,100)
    elif (time_period == '3Y'):
        change_percent = 'change3Y'
        range_color = (-100,100)

    color_scale = [
        (0, "#ff2c1c"),  # Bright red at -5%
        (0.5, "#484454"),  # Grey around 0%
        (1, "#30dc5c"),  # Bright green at 5%
    ]
    
    # Generate the treemap with fixed dimensions
    fig = px.treemap(
        df,
        path=[px.Constant("S&P 500 - Stocknear.com"), "sector", "industry", "symbol"],
        values="marketCap",
        color=change_percent,
        hover_data=[change_percent, "symbol", "marketCap"],
        color_continuous_scale=color_scale,
        range_color=range_color,
        color_continuous_midpoint=0,
        width=1200,  # Fixed width
        height=800   # Fixed height
    )
    
    # Update layout with fixed dimensions and other settings
    fig.update_layout(
        margin=dict(t=20, l=0, r=0, b=10),
        font=dict(size=13),  # Default font size for sectors/industries
        coloraxis_colorbar=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,  # Disable autosize
        width=1200,     # Fixed width
        height=1200     # Fixed height
    )
    
    templates = {
        "root": "<span style='font-size:24px; color: white !important;'><b>%{label}</b></span>",
        "sector": "<span style='font-size:22px; color: white !important;'><b>%{label}</b></span>",
        "industry": "<span style='font-size:20px; color: white !important;'><b>%{label}</b></span>",
        "symbol": "<span style='font-size:20px; color: white !important;'><b>%{customdata[1]}</b></span><br>" +
                 "<span style='font-size:18px; color: white !important;'>%{customdata[0]:.2f}%</span>"
    }

    # Update text templates based on the level
    fig.data[0].texttemplate = templates["symbol"]  # Default template for symbols

    
    # Set the text position, border, and ensure the custom font sizes are applied
    fig.update_traces(
        textposition="middle center",
        marker=dict(line=dict(color="black", width=1)),
        hoverinfo='skip',
        hovertemplate=None,
    )
    
    # Disable the color bar
    fig.update(layout_coloraxis_showscale=False)
    
    
    if save_html:
        fixed_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    background-color: white;
                }}
                
                @media (prefers-color-scheme: dark) {{
                    body {{
                        background-color: #09090B;
                    }}
                }}

                .plot-container {{
                    width: 1200px;
                    height: 1200px;
                }}
            </style>
        </head>
        <body class="container">
            <div class="plot-container">
                {fig.to_html(
                    include_plotlyjs='cdn',
                    full_html=False,
                    config=dict(
                        displayModeBar=False,
                        responsive=False,
                        staticPlot=True
                    )
                )}
            </div>
        </body>
        </html>
        """
        with open(f"json/heatmap/{time_period}.html", "w") as f:
            f.write(fixed_html)


if __name__ == "__main__":
    for time in ['1D',"1W","1M","3M","6M","1Y","3Y"]:
        create_treemap(time_period = time)