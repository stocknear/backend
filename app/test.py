import plotly.graph_objects as go

# Data for the semiconductor stocks
data = {
    'labels': ['Semiconductors', 'NVDA', 'AVGO', 'AMD', 'TXN', 'QCOM', 'ADI', 'INTC', 'MU', 'NXPI', 'MPWR'],
    'parents': ['', 'Semiconductors', 'Semiconductors', 'Semiconductors', 'Semiconductors', 
               'Semiconductors', 'Semiconductors', 'Semiconductors', 'Semiconductors', 
               'Semiconductors', 'Semiconductors'],
    'values': [100, 40, 15, 10, 8, 7, 6, 5, 4, 3, 2],  # Approximate sizes
    'performance': [0, -0.02, 0.29, -4.31, -0.29, -0.9, 2.12, -0.65, -2.45, -1.55, -1.9]
}

# Function to determine color based on performance with updated colors
def get_color(perf):
    if perf > 0:
        return f'rgb(75, 192, 75)'  # Brighter green
    elif perf < 0:
        return f'rgb(255, 82, 82)'  # Brighter red
    else:
        return f'rgb(128, 128, 128)'  # Gray for neutral

# Create color list
colors = [get_color(perf) for perf in data['performance']]

# Create text labels with performance
text = [f"{label}<br>{perf}%" if i > 0 else "" 
        for i, (label, perf) in enumerate(zip(data['labels'], data['performance']))]

# Create the treemap
fig = go.Figure(go.Treemap(
    labels=data['labels'],
    parents=data['parents'],
    values=data['values'],
    text=text,
    textinfo="label",
    hovertext=text,
    marker=dict(
        colors=colors,
        line=dict(width=2, color='white')
    ),
    textfont=dict(
        size=16,
        color='white'
    ),
))

# Update layout
fig.update_layout(
    title="Semiconductors",
    width=800,
    height=500,
    margin=dict(t=50, l=0, r=0, b=0),
    paper_bgcolor='rgb(128, 128, 128)',
    plot_bgcolor='rgb(128, 128, 128)',
    showlegend=False,
)

# Configuration to remove interactivity
config = {
    'displayModeBar': False,
    'staticPlot': True,  # Makes the plot completely static
    'scrollZoom': False,
    'doubleClick': False,
    'showTips': False,
    'responsive': False
}

# Save as HTML file
fig.write_html("json/heatmap/data.html", config=config)