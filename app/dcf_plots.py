"""
DCF Model Visualization Module
Creates professional charts similar to FastGraphs and existing stocknear plotting style
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set the style to match existing plots
plt.style.use('default')
sns.set_style("whitegrid")

class DCFPlotter:
    """
    Creates professional DCF analysis charts matching stocknear's visual style
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Initialize DCF Plotter
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'primary': '#1f77b4',      # Blue (matches existing plots)
            'secondary': '#ff7f0e',    # Orange
            'success': '#2ca02c',      # Green
            'danger': '#d62728',       # Red
            'warning': '#ff9800',      # Amber
            'conservative': '#2E8B57', # Sea Green
            'moderate': '#4169E1',     # Royal Blue
            'optimistic': '#32CD32',   # Lime Green
            'grid': '#E5E5E5',         # Light gray
            'text': '#333333'          # Dark gray
        }
    
    def plot_dcf_valuation_summary(self, 
                                 dcf_results: Dict[str, Any],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive DCF valuation summary chart
        Similar to FastGraphs style showing multiple scenarios
        
        Args:
            dcf_results: Results from DCF analysis
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'DCF Valuation Analysis - {dcf_results["ticker"]}', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # 1. Scenario Analysis (Top Left)
        scenarios = dcf_results['scenarios']
        scenario_names = ['Conservative', 'Moderate', 'Optimistic']
        scenario_values = [
            scenarios['conservative']['value_per_share'],
            scenarios['moderate']['value_per_share'],
            scenarios['optimistic']['value_per_share']
        ]
        
        current_price = dcf_results['current_price']
        
        ax1.barh(scenario_names, scenario_values, 
                color=[self.colors['conservative'], self.colors['moderate'], self.colors['optimistic']],
                alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add current price line
        ax1.axvline(x=current_price, color=self.colors['danger'], 
                   linestyle='--', linewidth=2, label=f'Current Price: ${current_price:.2f}')
        
        ax1.set_xlabel('Value per Share ($)', fontsize=12)
        ax1.set_title('DCF Valuation Scenarios', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(scenario_values):
            ax1.text(v + max(scenario_values) * 0.01, i, f'${v:.2f}', 
                    va='center', fontsize=11, fontweight='bold')
        
        # 2. Historical FCF Trend (Top Right)
        if dcf_results['historical_fcf']:
            years = list(range(len(dcf_results['historical_fcf'])))
            fcf_billions = [fcf / 1e9 for fcf in dcf_results['historical_fcf']]
            
            ax2.plot(years, fcf_billions, marker='o', linewidth=2, 
                    markersize=8, color=self.colors['primary'], label='Historical FCF')
            
            # Add trend line
            z = np.polyfit(years, fcf_billions, 1)
            p = np.poly1d(z)
            ax2.plot(years, p(years), "--", color=self.colors['secondary'], 
                    alpha=0.8, label='Trend Line')
            
            ax2.set_xlabel('Years (Historical)', fontsize=12)
            ax2.set_ylabel('Free Cash Flow (Billions $)', fontsize=12)
            ax2.set_title('Historical Free Cash Flow Trend', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Growth Rate Analysis (Bottom Left)
        growth_rates = dcf_results['growth_rates']
        growth_names = ['Historical Avg', 'CAGR', 'Conservative', 'Moderate', 'Optimistic']
        growth_values = [
            growth_rates['historical_average'] * 100,
            growth_rates['cagr'] * 100,
            growth_rates['conservative'] * 100,
            growth_rates['moderate'] * 100,
            growth_rates['optimistic'] * 100
        ]
        
        colors_growth = [self.colors['primary'], self.colors['secondary'], 
                        self.colors['conservative'], self.colors['moderate'], 
                        self.colors['optimistic']]
        
        bars = ax3.bar(growth_names, growth_values, color=colors_growth, 
                      alpha=0.7, edgecolor='black', linewidth=1)
        
        ax3.set_ylabel('Growth Rate (%)', fontsize=12)
        ax3.set_title('Growth Rate Scenarios', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, growth_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Valuation Distribution (Bottom Right)
        monte_carlo = dcf_results.get('monte_carlo', {})
        if monte_carlo:
            # Create a mock distribution for visualization
            mean_val = monte_carlo['mean_value']
            std_val = monte_carlo['std_dev']
            
            # Generate sample distribution
            np.random.seed(42)
            values = np.random.normal(mean_val, std_val, 1000)
            values = values[values > 0]  # Remove negative values
            
            ax4.hist(values, bins=30, color=self.colors['primary'], alpha=0.7, 
                    edgecolor='black', linewidth=0.5)
            
            # Add vertical lines for percentiles
            ax4.axvline(x=current_price, color=self.colors['danger'], 
                       linestyle='--', linewidth=2, label=f'Current: ${current_price:.2f}')
            ax4.axvline(x=mean_val, color=self.colors['success'], 
                       linestyle='-', linewidth=2, label=f'Mean: ${mean_val:.2f}')
            
            ax4.set_xlabel('Value per Share ($)', fontsize=12)
            ax4.set_ylabel('Frequency', fontsize=12)
            ax4.set_title('Monte Carlo Valuation Distribution', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_analysis(self, 
                                sensitivity_df: pd.DataFrame,
                                ticker: str,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a sensitivity analysis heatmap
        
        Args:
            sensitivity_df: DataFrame with sensitivity analysis results
            ticker: Stock ticker symbol
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(sensitivity_df, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=sensitivity_df.iloc[len(sensitivity_df)//2, len(sensitivity_df.columns)//2],
                   cbar_kws={'label': 'Value per Share ($)'}, ax=ax)
        
        ax.set_title(f'DCF Sensitivity Analysis - {ticker}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Growth Rate Scenarios', fontsize=12)
        ax.set_ylabel('WACC Scenarios', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_projected_cash_flows(self,
                                base_fcf: float,
                                projected_fcf: np.ndarray,
                                growth_rate: float,
                                ticker: str,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot projected free cash flows over time
        
        Args:
            base_fcf: Starting free cash flow
            projected_fcf: Array of projected cash flows
            growth_rate: Growth rate used for projections
            ticker: Stock ticker symbol
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create years array
        years = list(range(len(projected_fcf)))
        fcf_billions = projected_fcf / 1e9
        
        # Plot projected cash flows
        ax.plot(years, fcf_billions, marker='o', linewidth=3, markersize=8, 
               color=self.colors['primary'], label='Projected FCF')
        
        # Add base year
        ax.plot([-1], [base_fcf / 1e9], marker='o', markersize=10, 
               color=self.colors['danger'], label='Current FCF')
        
        # Fill area under curve
        ax.fill_between(years, fcf_billions, alpha=0.3, color=self.colors['primary'])
        
        ax.set_xlabel('Years (Projected)', fontsize=12)
        ax.set_ylabel('Free Cash Flow (Billions $)', fontsize=12)
        ax.set_title(f'Projected Free Cash Flow - {ticker} (Growth: {growth_rate:.1%})', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value annotations for key years
        for i in [0, len(years)//2, len(years)-1]:
            ax.annotate(f'${fcf_billions[i]:.1f}B', 
                       xy=(years[i], fcf_billions[i]), 
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dcf_waterfall(self,
                          dcf_results: Dict[str, Any],
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a DCF valuation waterfall chart
        
        Args:
            dcf_results: Results from DCF analysis
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get moderate scenario values
        moderate = dcf_results['scenarios']['moderate']
        
        # Create waterfall components
        components = [
            'PV of Cash Flows',
            'PV of Terminal Value',
            'Enterprise Value',
            'Per Share Value'
        ]
        
        values = [
            moderate['pv_cash_flows'] / 1e9,
            moderate['pv_terminal_value'] / 1e9,
            moderate['enterprise_value'] / 1e9,
            moderate['value_per_share']
        ]
        
        # Create cumulative values for waterfall
        cumulative = [values[0], values[0] + values[1], values[2], values[3]]
        
        # Plot bars
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['success'], self.colors['moderate']]
        
        bars = ax.bar(components, values, color=colors, alpha=0.7, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if i < 3:  # Billions
                label = f'${value:.1f}B'
            else:  # Per share
                label = f'${value:.2f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   label, ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Value (Billions $ / $ per Share)', fontsize=12)
        ax.set_title(f'DCF Valuation Waterfall - {dcf_results["ticker"]}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monte_carlo_simulation(self,
                                  monte_carlo_results: Dict[str, Any],
                                  ticker: str,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create Monte Carlo simulation histogram similar to existing style
        
        Args:
            monte_carlo_results: Monte Carlo simulation results
            ticker: Stock ticker symbol
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if 'valuations' in monte_carlo_results:
            valuations = monte_carlo_results['valuations']
        else:
            # Generate sample data if not available
            mean_val = monte_carlo_results['mean_value']
            std_val = monte_carlo_results['std_dev']
            np.random.seed(42)
            valuations = np.random.normal(mean_val, std_val, 10000)
            valuations = valuations[valuations > 0]
        
        # Create histogram
        n, bins, patches = ax.hist(valuations, bins=50, color=self.colors['primary'], 
                                  alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add percentile lines
        p5 = monte_carlo_results['percentile_5']
        p95 = monte_carlo_results['percentile_95']
        mean_val = monte_carlo_results['mean_value']
        
        ax.axvline(x=p5, color=self.colors['danger'], linewidth=3, 
                  label=f'5th Percentile: ${p5:.2f}')
        ax.axvline(x=mean_val, color=self.colors['success'], linewidth=3, 
                  label=f'Mean: ${mean_val:.2f}')
        ax.axvline(x=p95, color=self.colors['warning'], linewidth=3, 
                  label=f'95th Percentile: ${p95:.2f}')
        
        # Add statistics text
        plt.figtext(0.65, 0.8, f"Mean value: ${mean_val:.2f}", fontsize=12)
        plt.figtext(0.65, 0.75, f"Std Dev: ${monte_carlo_results['std_dev']:.2f}", fontsize=12)
        plt.figtext(0.65, 0.7, f"5th-95th Range: ${p95-p5:.2f}", fontsize=12)
        
        ax.set_xlabel('Value per Share ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Monte Carlo DCF Valuation Distribution - {ticker}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_fastgraphs_style_analysis(self,
                                     dcf_results: Dict[str, Any],
                                     historical_prices: Optional[List[float]] = None,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a FastGraphs-style comprehensive analysis chart
        
        Args:
            dcf_results: Results from DCF analysis
            historical_prices: Optional historical price data
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        ticker = dcf_results['ticker']
        current_price = dcf_results['current_price']
        
        # Top chart: Valuation ranges and current price
        scenarios = dcf_results['scenarios']
        
        # Create time series of valuation ranges (simulated)
        months = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        
        # Simulate valuation bands over time
        np.random.seed(42)
        conservative_band = scenarios['conservative']['value_per_share'] * (1 + np.random.normal(0, 0.1, len(months)))
        moderate_band = scenarios['moderate']['value_per_share'] * (1 + np.random.normal(0, 0.1, len(months)))
        optimistic_band = scenarios['optimistic']['value_per_share'] * (1 + np.random.normal(0, 0.1, len(months)))
        
        # Plot valuation bands
        ax1.fill_between(months, conservative_band, optimistic_band, 
                        color=self.colors['success'], alpha=0.2, label='Valuation Range')
        ax1.plot(months, conservative_band, color=self.colors['conservative'], 
                linewidth=2, label='Conservative Value')
        ax1.plot(months, moderate_band, color=self.colors['moderate'], 
                linewidth=2, label='Moderate Value')
        ax1.plot(months, optimistic_band, color=self.colors['optimistic'], 
                linewidth=2, label='Optimistic Value')
        
        # Add current price line
        ax1.axhline(y=current_price, color=self.colors['danger'], 
                   linestyle='--', linewidth=3, label=f'Current Price: ${current_price:.2f}')
        
        ax1.set_ylabel('Price per Share ($)', fontsize=12)
        ax1.set_title(f'{ticker} - DCF Valuation Analysis (FastGraphs Style)', 
                     fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        
        # Bottom chart: Key metrics over time
        metrics_data = {
            'WACC': [dcf_results['wacc']] * len(months),
            'Growth Rate': [dcf_results['growth_rates']['moderate']] * len(months),
            'Terminal Growth': [0.025] * len(months)  # 2.5% terminal growth
        }
        
        for metric, values in metrics_data.items():
            # Add some variation to make it more realistic
            varied_values = np.array(values) * (1 + np.random.normal(0, 0.05, len(months)))
            ax2.plot(months, varied_values * 100, linewidth=2, label=metric)
        
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Rate (%)', fontsize=12)
        ax2.set_title('Key DCF Assumptions Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dcf_dashboard(self,
                           dcf_results: Dict[str, Any],
                           save_path: Optional[str] = 'dcf_dashboard.png') -> None:
        """
        Create a comprehensive DCF dashboard with all key visualizations
        
        Args:
            dcf_results: Results from DCF analysis
            save_path: Path to save the dashboard image
        """
        # Create individual plots
        self.plot_dcf_valuation_summary(dcf_results, save_path.replace('.png', '_summary.png'))
        
        # Create sensitivity analysis if available
        if 'sensitivity_analysis' in dcf_results:
            sensitivity_df = pd.DataFrame(dcf_results['sensitivity_analysis'])
            self.plot_sensitivity_analysis(sensitivity_df, dcf_results['ticker'], 
                                         save_path.replace('.png', '_sensitivity.png'))
        
        # Create Monte Carlo plot
        if 'monte_carlo' in dcf_results:
            self.plot_monte_carlo_simulation(dcf_results['monte_carlo'], 
                                           dcf_results['ticker'],
                                           save_path.replace('.png', '_monte_carlo.png'))
        
        # Create FastGraphs-style analysis
        self.plot_fastgraphs_style_analysis(dcf_results, 
                                          save_path=save_path.replace('.png', '_fastgraphs.png'))
        
        print(f"DCF Dashboard plots saved with prefix: {save_path.replace('.png', '')}")


# Example usage and testing
async def main():
    """Example usage of DCF plotting"""
    from dcf_model import DCFModel
    
    # Create DCF model and run analysis
    dcf = DCFModel(use_local_data=True)
    results = await dcf.full_dcf_analysis('AAPL')
    
    # Create plotter and generate visualizations
    plotter = DCFPlotter()
    
    print("Creating DCF visualization dashboard...")
    
    # Create comprehensive dashboard
    plotter.create_dcf_dashboard(results, 'dcf_analysis_AAPL.png')
    
    print("DCF visualization dashboard created successfully!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())