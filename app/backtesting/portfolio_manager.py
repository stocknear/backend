from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from backtesting.strategy_engine import TradingSignal, SignalType


class Position:
    """Represents a trading position"""
    
    def __init__(self, shares: int = 0, avg_price: float = 0.0):
        self.shares = shares
        self.avg_price = avg_price
        self.total_cost = shares * avg_price
    
    def update(self, shares_delta: int, price: float):
        """Update position with new trade"""
        if shares_delta > 0:  # Buy
            new_total_cost = self.total_cost + (shares_delta * price)
            new_shares = self.shares + shares_delta
            self.avg_price = new_total_cost / new_shares if new_shares > 0 else 0.0
            self.shares = new_shares
            self.total_cost = new_total_cost
        elif shares_delta < 0:  # Sell
            self.shares += shares_delta  # shares_delta is negative
            if self.shares <= 0:
                self.shares = 0
                self.avg_price = 0.0
                self.total_cost = 0.0
            else:
                self.total_cost = self.shares * self.avg_price
    
    def market_value(self, current_price: float) -> float:
        """Calculate current market value of position"""
        return self.shares * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss"""
        return self.market_value(current_price) - self.total_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'shares': self.shares,
            'avg_price': self.avg_price,
            'total_cost': self.total_cost
        }


class Trade:
    """Represents a completed trade"""
    
    def __init__(self, date: str, action: str, shares: int, price: float, 
                 commission: float, metadata: Dict[str, Any] = None):
        self.date = date
        self.action = action
        self.shares = shares
        self.price = price
        self.commission = commission
        self.gross_amount = shares * price
        self.net_amount = self.gross_amount - commission if action == 'SELL' else self.gross_amount + commission
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'date': self.date,
            'action': self.action,
            'shares': self.shares,
            'price': self.price,
            'gross_amount': self.gross_amount,
            'commission': self.commission,
            'net_amount': self.net_amount,
            **self.metadata
        }


class PortfolioManager:
    """Manages portfolio state, positions, and trade execution - supports multiple tickers"""
    
    def __init__(self, initial_capital: float = 100000, commission_rate: float = 0.001, 
                 position_size_pct: float = 0.95, tickers: List[str] = None):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.position_size_pct = position_size_pct  # Max percentage of capital to use per trade
        self.tickers = tickers or []
        self.num_tickers = len(self.tickers) if self.tickers else 1
        
        # Portfolio state
        self.cash = initial_capital
        # Support both single position (legacy) and multi-ticker positions
        if self.tickers:
            self.positions = {ticker: Position() for ticker in self.tickers}
            self.position = None  # Legacy single position disabled for multi-ticker
        else:
            self.position = Position()  # Legacy single position
            self.positions = {}
        
        self.trades = []
        self.portfolio_history = []
        
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        if self.tickers:
            self.positions = {ticker: Position() for ticker in self.tickers}
        else:
            self.position = Position()
        self.trades = []
        self.portfolio_history = []
    
    def get_portfolio_value(self, current_prices: Dict[str, float] = None, current_price: float = None) -> float:
        """Calculate total portfolio value - supports both single ticker and multi-ticker"""
        if self.tickers and current_prices:
            # Multi-ticker mode
            total_position_value = sum(
                self.positions[ticker].market_value(current_prices.get(ticker, 0))
                for ticker in self.tickers
            )
            return self.cash + total_position_value
        elif self.position and current_price is not None:
            # Single ticker mode (legacy)
            return self.cash + self.position.market_value(current_price)
        else:
            return self.cash  # Only cash, no positions
    
    def can_buy(self, price: float, shares: int) -> bool:
        """Check if we have enough cash to buy shares"""
        total_cost = shares * price * (1 + self.commission_rate)
        return total_cost <= self.cash
    
    def can_sell(self, shares: int, ticker: str = None) -> bool:
        """Check if we have enough shares to sell - supports both single and multi-ticker"""
        if self.tickers and ticker:
            # Multi-ticker mode
            return self.positions[ticker].shares >= shares
        elif self.position:
            # Single ticker mode (legacy)
            return self.position.shares >= shares
        else:
            return False
    
    def calculate_position_size(self, price: float, ticker: str = None) -> int:
        """Calculate optimal position size based on available cash and number of tickers"""
        if self.tickers:
            # Multi-ticker mode: divide available cash equally among tickers
            available_cash = (self.cash * self.position_size_pct) / self.num_tickers
        else:
            # Single ticker mode
            available_cash = self.cash * self.position_size_pct
        
        max_shares = int(available_cash / (price * (1 + self.commission_rate)))
        return max(0, max_shares)
    
    def execute_signal(self, signal: TradingSignal, current_price: float = None, ticker: str = None) -> Optional[Trade]:
        """
        Execute a trading signal
        
        Args:
            signal: TradingSignal object
            current_price: Current market price (uses signal price if None)
            ticker: Ticker symbol for multi-ticker support
            
        Returns:
            Trade object if executed, None if not executed
        """
        price = current_price or signal.price
        
        if signal.signal_type == SignalType.BUY:
            return self._execute_buy(signal, price, ticker)
        elif signal.signal_type == SignalType.SELL:
            return self._execute_sell(signal, price, ticker)
        
        return None
    
    def _execute_buy(self, signal: TradingSignal, price: float, ticker: str = None) -> Optional[Trade]:
        """Execute a buy order - supports both single and multi-ticker"""
        
        if self.tickers and ticker:
            # Multi-ticker mode
            position = self.positions[ticker]
            if position.shares > 0:  # Already have position in this ticker
                return None
                
            # Calculate shares to buy (divided equally among tickers)
            shares = self.calculate_position_size(price, ticker)
        else:
            # Single ticker mode (legacy)
            if self.position.shares > 0:  # Already have position
                return None
            shares = self.calculate_position_size(price)
            position = self.position
        
        if shares <= 0:
            return None
        
        # Check if we can afford it
        if not self.can_buy(price, shares):
            return None
        
        # Execute trade
        commission = shares * price * self.commission_rate
        total_cost = shares * price + commission
        
        self.cash -= total_cost
        position.update(shares, price)
        
        # Create trade record
        trade = Trade(
            date=signal.date,
            action='BUY',
            shares=shares,
            price=price,
            commission=commission,
            metadata=signal.metadata or {}
        )
        
        # Add ticker to trade metadata
        if ticker:
            trade.metadata['ticker'] = ticker
        
        # Update metadata with portfolio value
        if self.tickers:
            # For multi-ticker, we'll need current prices - store what we can
            trade.metadata['portfolio_value'] = self.cash + sum(pos.total_cost for pos in self.positions.values())
        else:
            trade.metadata['portfolio_value'] = self.get_portfolio_value(price)
        
        self.trades.append(trade)
        return trade
    
    def _execute_sell(self, signal: TradingSignal, price: float, ticker: str = None) -> Optional[Trade]:
        """Execute a sell order - supports both single and multi-ticker"""
        
        if self.tickers and ticker:
            # Multi-ticker mode
            position = self.positions[ticker]
            if position.shares <= 0:  # No position to sell
                return None
            shares = position.shares  # Sell entire position
        else:
            # Single ticker mode (legacy)
            if self.position.shares <= 0:  # No position to sell
                return None
            shares = self.position.shares  # Sell entire position
            position = self.position
        
        # Execute trade
        commission = shares * price * self.commission_rate
        gross_proceeds = shares * price
        net_proceeds = gross_proceeds - commission
        
        self.cash += net_proceeds
        position.update(-shares, price)  # Reduce position
        
        # Create trade record
        trade = Trade(
            date=signal.date,
            action='SELL',
            shares=shares,
            price=price,
            commission=commission,
            metadata=signal.metadata or {}
        )
        
        # Add ticker to trade metadata
        if ticker:
            trade.metadata['ticker'] = ticker
        
        # Update metadata with portfolio value
        if self.tickers:
            # For multi-ticker, store approximated portfolio value
            trade.metadata['portfolio_value'] = self.cash + sum(pos.total_cost for pos in self.positions.values())
        else:
            trade.metadata['portfolio_value'] = self.get_portfolio_value(price)
        
        self.trades.append(trade)
        return trade
    
    def update_portfolio_history(self, date: str, current_prices: Dict[str, float] = None, price: float = None):
        """Update portfolio value history - supports both single ticker and multi-ticker"""
        
        if self.tickers and current_prices:
            # Multi-ticker mode
            portfolio_value = self.get_portfolio_value(current_prices)
            total_position_value = sum(
                self.positions[ticker].market_value(current_prices.get(ticker, 0))
                for ticker in self.tickers
            )
            total_shares = sum(self.positions[ticker].shares for ticker in self.tickers)
            total_unrealized_pnl = sum(
                self.positions[ticker].unrealized_pnl(current_prices.get(ticker, 0))
                for ticker in self.tickers
            )
            
            history_entry = {
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'position_value': total_position_value,
                'total_shares': total_shares,
                'unrealized_pnl': total_unrealized_pnl,
                'positions': {
                    ticker: {
                        'shares': self.positions[ticker].shares,
                        'market_value': self.positions[ticker].market_value(current_prices.get(ticker, 0)),
                        'unrealized_pnl': self.positions[ticker].unrealized_pnl(current_prices.get(ticker, 0))
                    }
                    for ticker in self.tickers
                }
            }
        else:
            # Single ticker mode (legacy)
            portfolio_value = self.get_portfolio_value(current_price=price)
            history_entry = {
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'position_value': self.position.market_value(price) if self.position else 0,
                'shares': self.position.shares if self.position else 0,
                'unrealized_pnl': self.position.unrealized_pnl(price) if self.position else 0
            }
        
        self.portfolio_history.append(history_entry)
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary of all trades"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_buy_trades': 0,
                'total_sell_trades': 0,
                'total_commission_paid': 0,
                'total_shares_traded': 0
            }
        
        buy_trades = [t for t in self.trades if t.action == 'BUY']
        sell_trades = [t for t in self.trades if t.action == 'SELL']
        
        return {
            'total_trades': len(self.trades),
            'total_buy_trades': len(buy_trades),
            'total_sell_trades': len(sell_trades),
            'total_commission_paid': sum(t.commission for t in self.trades),
            'total_shares_traded': sum(t.shares for t in self.trades),
            'first_trade_date': self.trades[0].date if self.trades else None,
            'last_trade_date': self.trades[-1].date if self.trades else None
        }
    
    def calculate_performance_metrics(self, benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """Calculate portfolio performance metrics including Sortino ratio, beta, and alpha"""
        if not self.portfolio_history:
            return {}
        
        # Convert portfolio history to DataFrame
        df = pd.DataFrame(self.portfolio_history)
        portfolio_values = df['portfolio_value'].values
        
        # Basic returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return (assuming daily data)
        days = len(portfolio_values)
        annual_return = (portfolio_values[-1] / self.initial_capital) ** (252 / days) - 1 if days > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0  # 2% risk-free rate
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (annual_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        # Beta and Alpha (relative to benchmark)
        beta = 0.0
        alpha = 0.0
        if benchmark_returns is not None and len(benchmark_returns) > 1 and len(returns) > 1:
            # Ensure benchmark and portfolio returns are aligned
            min_length = min(len(returns), len(benchmark_returns))
            portfolio_returns_aligned = returns[:min_length]
            benchmark_returns_aligned = benchmark_returns.values[:min_length] if hasattr(benchmark_returns, 'values') else benchmark_returns[:min_length]
            
            # Calculate beta using covariance and variance
            covariance = np.cov(portfolio_returns_aligned, benchmark_returns_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_returns_aligned)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Calculate alpha (Jensen's Alpha)
            benchmark_annual_return = (1 + np.mean(benchmark_returns_aligned)) ** 252 - 1
            expected_return = 0.02 + beta * (benchmark_annual_return - 0.02)  # CAPM expected return
            alpha = annual_return - expected_return
        
        # Trade analysis
        trade_summary = self.get_trade_summary()
        winning_trades = 0
        total_profit = 0
        total_loss = 0
        
        # Pair buy/sell trades to calculate P&L
        buy_trades = [t for t in self.trades if t.action == 'BUY']
        sell_trades = [t for t in self.trades if t.action == 'SELL']
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            pnl = (sell_trades[i].price - buy_trades[i].price) * buy_trades[i].shares
            pnl -= (buy_trades[i].commission + sell_trades[i].commission)  # Subtract commissions
            
            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
            else:
                total_loss += abs(pnl)
        
        win_rate = winning_trades / trade_summary['total_buy_trades'] if trade_summary['total_buy_trades'] > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'beta': beta,
            'alpha': alpha,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades_completed': min(len(buy_trades), len(sell_trades)),
            **trade_summary
        }
    
    def _calculate_max_drawdown(self, portfolio_values: np.array) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return abs(np.min(drawdown))
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        return {
            'cash': self.cash,
            'position': self.position.to_dict(),
            'initial_capital': self.initial_capital,
            'total_trades': len(self.trades),
            'commission_rate': self.commission_rate,
            'position_size_pct': self.position_size_pct
        }