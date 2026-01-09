# tests/test_strategy_api.py

import pytest
import pandas as pd
import pytz
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

try:
    from .portfolios.strategy_api import AssetData, MarketData, PortfolioManager, StrategyContext
except ImportError:
    from portfolios.strategy_api import AssetData, MarketData, PortfolioManager, StrategyContext

class TestAssetData:
    """Test suite for AssetData class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for a single ticker"""
        dates = pd.date_range('2024-01-01', periods=10, freq='D', tz='America/New_York')
        return pd.DataFrame({
            'open_price': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'high_price': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            'low_price': [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            'close_price': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
    
    def test_asset_exists_with_valid_data(self, sample_data):
        """Test that AssetData correctly identifies existing asset"""
        current_time = pd.Timestamp('2024-01-05', tz='America/New_York')
        asset = AssetData('AAPL', sample_data, current_time)
        
        assert asset.Exists is True
        assert asset.Close == 104.5
        assert asset.Open == 104.0
        assert asset.High == 105.0
        assert asset.Low == 103.0
        assert asset.Volume == 1400000
    
    def test_asset_not_exists_with_empty_data(self):
        """Test that AssetData correctly handles empty DataFrame"""
        empty_df = pd.DataFrame()
        asset = AssetData('AAPL', empty_df, None)
        
        assert asset.Exists is False
        assert asset.Close is None
        assert asset.Open is None
        assert asset.High is None
        assert asset.Low is None
        assert asset.Volume is None
    
    def test_asset_with_none_current_time(self, sample_data):
        """Test that AssetData uses latest data when current_time is None"""
        asset = AssetData('AAPL', sample_data, None)
        
        assert asset.Exists is True
        assert asset.Close == 109.5  # Last row
        assert asset.Volume == 1900000
    
    def test_asset_with_future_time(self, sample_data):
        """Test that AssetData handles future timestamps correctly"""
        future_time = pd.Timestamp('2024-01-15', tz='America/New_York')
        asset = AssetData('AAPL', sample_data, future_time)
        
        assert asset.Exists is True
        assert asset.Close == 109.5  # Should use latest available
    
    def test_asset_with_past_time_before_data(self, sample_data):
        """Test that AssetData handles timestamps before data starts"""
        past_time = pd.Timestamp('2023-12-01', tz='America/New_York')
        asset = AssetData('AAPL', sample_data, past_time)
        
        assert asset.Exists is False
    
    def test_history_method(self, sample_data):
        """Test the History method returns correct lookback period"""
        current_time = pd.Timestamp('2024-01-10', tz='America/New_York')
        asset = AssetData('AAPL', sample_data, current_time)
        
        history = asset.History('3d')
        
        assert len(history) == 4
        assert history['close_price'].iloc[-1] == 109.5
    
    def test_history_with_none_time(self, sample_data):
        """Test History method when current_time is None"""
        asset = AssetData('AAPL', sample_data, None)
        history = asset.History('5d')
        
        assert len(history) == 6
        assert history['close_price'].iloc[-1] == 109.5
    
    def test_history_empty_dataframe(self):
        """Test History method with empty DataFrame"""
        empty_df = pd.DataFrame()
        asset = AssetData('AAPL', empty_df, None)
        history = asset.History('5d')
        
        assert history.empty
    
    def test_asset_with_missing_close_price(self, sample_data):
        """Test that AssetData handles missing close price correctly"""
        sample_data.loc[sample_data.index[-1], 'close_price'] = None
        current_time = sample_data.index[-1]
        asset = AssetData('AAPL', sample_data, current_time)
        
        # AssetData converts None to NaN, which still counts as Exists=True
        # but Close will be NaN
        assert asset.Exists is True
        assert pd.isna(asset.Close)
    
    def test_asset_repr(self, sample_data):
        """Test string representation of AssetData"""
        current_time = pd.Timestamp('2024-01-05', tz='America/New_York')
        asset = AssetData('AAPL', sample_data, current_time)
        
        repr_str = repr(asset)
        assert 'AAPL' in repr_str
        assert 'Close=104.5' in repr_str


class TestMarketData:
    """Test suite for MarketData class"""
    
    @pytest.fixture
    def multi_ticker_data(self):
        """Create sample market data for multiple tickers"""
        dates = pd.date_range('2024-01-01', periods=5, freq='D', tz='America/New_York')
        data = []
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            for i, date in enumerate(dates):
                data.append({
                    'timestamp': date,
                    'ticker': ticker,
                    'open_price': 100.0 + i,
                    'high_price': 101.0 + i,
                    'low_price': 99.0 + i,
                    'close_price': 100.5 + i,
                    'volume': 1000000 + i * 100000
                })
        return pd.DataFrame(data)
    
    def test_market_data_initialization(self, multi_ticker_data):
        """Test MarketData initializes correctly"""
        current_time = pd.Timestamp('2024-01-03', tz='America/New_York')
        market = MarketData(multi_ticker_data, current_time)
        
        assert 'AAPL' in market
        assert 'MSFT' in market
        assert 'GOOGL' in market
        assert 'TSLA' not in market
    
    def test_market_data_getitem(self, multi_ticker_data):
        """Test accessing individual assets via __getitem__"""
        current_time = pd.Timestamp('2024-01-03', tz='America/New_York')
        market = MarketData(multi_ticker_data, current_time)
        
        aapl = market['AAPL']
        assert isinstance(aapl, AssetData)
        assert aapl.Exists is True
        assert aapl.Close == 102.5
    
    def test_market_data_caching(self, multi_ticker_data):
        """Test that MarketData caches AssetData objects"""
        current_time = pd.Timestamp('2024-01-03', tz='America/New_York')
        market = MarketData(multi_ticker_data, current_time)
        
        aapl1 = market['AAPL']
        aapl2 = market['AAPL']
        
        assert aapl1 is aapl2  # Same object reference
    
    def test_market_data_with_empty_df(self):
        """Test MarketData with empty DataFrame"""
        empty_df = pd.DataFrame()
        current_time = pd.Timestamp('2024-01-01', tz='America/New_York')
        market = MarketData(empty_df, current_time)
        
        assert 'AAPL' not in market
        aapl = market['AAPL']
        assert aapl.Exists is False
    
    def test_market_data_contains(self, multi_ticker_data):
        """Test __contains__ method"""
        current_time = pd.Timestamp('2024-01-03', tz='America/New_York')
        market = MarketData(multi_ticker_data, current_time)
        
        assert 'AAPL' in market
        assert 'UNKNOWN' not in market


class TestPortfolioManager:
    """Test suite for PortfolioManager class"""
    
    @pytest.fixture
    def sample_positions(self):
        """Create sample positions DataFrame"""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'quantity': [100, 50, 25]
        })
    
    def test_portfolio_initialization(self, sample_positions):
        """Test PortfolioManager initializes correctly"""
        portfolio = PortfolioManager(
            cash=50000.0,
            total_value=150000.0,
            positions_df=sample_positions
        )
        
        assert portfolio.cash == 50000.0
        assert portfolio.total_value == 150000.0
        assert portfolio.positions['AAPL'] == 100
        assert portfolio.positions['MSFT'] == 50
        assert portfolio.positions['GOOGL'] == 25
    
    def test_portfolio_with_empty_positions(self):
        """Test PortfolioManager with no positions"""
        portfolio = PortfolioManager(
            cash=100000.0,
            total_value=100000.0,
            positions_df=pd.DataFrame()
        )
        
        assert portfolio.cash == 100000.0
        assert portfolio.total_value == 100000.0
        assert len(portfolio.positions) == 0
    
    def test_get_asset_value(self, sample_positions):
        """Test calculating asset value"""
        portfolio = PortfolioManager(
            cash=50000.0,
            total_value=150000.0,
            positions_df=sample_positions
        )
        
        # 100 shares * $150 = $15,000
        asset_value = portfolio.get_asset_value('AAPL', 150.0)
        assert asset_value == 15000.0
        
        # No position
        asset_value = portfolio.get_asset_value('TSLA', 200.0)
        assert asset_value == 0.0
    
    def test_get_asset_weight(self, sample_positions):
        """Test calculating asset weight"""
        portfolio = PortfolioManager(
            cash=50000.0,
            total_value=150000.0,
            positions_df=sample_positions
        )
        
        # 100 shares * $150 = $15,000 / $150,000 = 0.1
        weight = portfolio.get_asset_weight('AAPL', 150.0)
        assert weight == pytest.approx(0.1, rel=1e-6)
        
        # No position
        weight = portfolio.get_asset_weight('TSLA', 200.0)
        assert weight == 0.0
    
    def test_get_asset_weight_zero_total_value(self, sample_positions):
        """Test asset weight calculation when total value is zero"""
        portfolio = PortfolioManager(
            cash=0.0,
            total_value=0.0,
            positions_df=sample_positions
        )
        
        weight = portfolio.get_asset_weight('AAPL', 150.0)
        assert weight == 0.0
    
    def test_portfolio_repr(self, sample_positions):
        """Test string representation of PortfolioManager"""
        portfolio = PortfolioManager(
            cash=50000.0,
            total_value=150000.0,
            positions_df=sample_positions
        )
        
        repr_str = repr(portfolio)
        assert 'TotalValue=150,000.00' in repr_str
        assert 'Cash=50,000.00' in repr_str
        assert 'Positions=3' in repr_str


class TestStrategyContext:
    """Test suite for StrategyContext class"""
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data"""
        dates = pd.date_range('2024-01-01', periods=3, freq='D', tz='America/New_York')
        data = []
        for ticker in ['AAPL', 'MSFT']:
            for i, date in enumerate(dates):
                data.append({
                    'timestamp': date,
                    'ticker': ticker,
                    'open_price': 100.0 + i,
                    'high_price': 101.0 + i,
                    'low_price': 99.0 + i,
                    'close_price': 100.5 + i,
                    'volume': 1000000
                })
        return pd.DataFrame(data)
    
    @pytest.fixture
    def cash_data(self):
        """Create sample cash DataFrame"""
        return pd.DataFrame({'notional': [50000.0]})
    
    @pytest.fixture
    def positions_data(self):
        """Create sample positions DataFrame"""
        return pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'quantity': [100, 50]
        })
    
    @pytest.fixture
    def port_notional_data(self):
        """Create sample portfolio notional DataFrame"""
        return pd.DataFrame({'notional': [150000.0]})
    
    @pytest.fixture
    def mock_executor(self):
        """Create mock executor"""
        return Mock()
    
    @pytest.fixture
    def portfolio_config(self):
        """Create sample portfolio config"""
        return {'id': '1', 'name': 'TestPortfolio'}
    
    def test_context_initialization(self, market_data, cash_data, positions_data, 
                                   port_notional_data, mock_executor, portfolio_config):
        """Test StrategyContext initializes correctly"""
        current_time = pd.Timestamp('2024-01-02', tz='America/New_York')
        
        context = StrategyContext(
            market_data_df=market_data,
            cash_df=cash_data,
            positions_df=positions_data,
            port_notional_df=port_notional_data,
            current_time=current_time,
            executor=mock_executor,
            portfolio_config=portfolio_config
        )
        
        assert context.time == current_time
        assert isinstance(context.Market, MarketData)
        assert isinstance(context.Portfolio, PortfolioManager)
        assert context.Portfolio.cash == 50000.0
        assert context.Portfolio.total_value == 150000.0
    
    def test_context_with_none_time(self, market_data, cash_data, positions_data,
                                    port_notional_data, mock_executor, portfolio_config):
        """Test StrategyContext with None current_time uses latest from data"""
        context = StrategyContext(
            market_data_df=market_data,
            cash_df=cash_data,
            positions_df=positions_data,
            port_notional_df=port_notional_data,
            current_time=None,
            executor=mock_executor,
            portfolio_config=portfolio_config
        )
        
        assert context.time is not None
        assert isinstance(context.time, pd.Timestamp)
    
    def test_buy_method(self, market_data, cash_data, positions_data,
                       port_notional_data, mock_executor, portfolio_config):
        """Test buy method calls executor correctly"""
        current_time = pd.Timestamp('2024-01-02', tz='America/New_York')
        
        context = StrategyContext(
            market_data_df=market_data,
            cash_df=cash_data,
            positions_df=positions_data,
            port_notional_df=port_notional_data,
            current_time=current_time,
            executor=mock_executor,
            portfolio_config=portfolio_config
        )
        
        context.buy('AAPL', confidence=0.8)
        
        mock_executor.execute_trade.assert_called_once()
        call_args = mock_executor.execute_trade.call_args[1]
        assert call_args['ticker'] == 'AAPL'
        assert call_args['signal_type'] == 'BUY'
        assert call_args['confidence'] == 0.8
        assert call_args['portfolio_id'] == '1'
    
    def test_sell_method(self, market_data, cash_data, positions_data,
                        port_notional_data, mock_executor, portfolio_config):
        """Test sell method calls executor correctly"""
        current_time = pd.Timestamp('2024-01-02', tz='America/New_York')
        
        context = StrategyContext(
            market_data_df=market_data,
            cash_df=cash_data,
            positions_df=positions_data,
            port_notional_df=port_notional_data,
            current_time=current_time,
            executor=mock_executor,
            portfolio_config=portfolio_config
        )
        
        context.sell('MSFT', confidence=1.0)
        
        mock_executor.execute_trade.assert_called_once()
        call_args = mock_executor.execute_trade.call_args[1]
        assert call_args['ticker'] == 'MSFT'
        assert call_args['signal_type'] == 'SELL'
        assert call_args['confidence'] == 1.0
    
    def test_trade_with_invalid_ticker(self, market_data, cash_data, positions_data,
                                       port_notional_data, mock_executor, portfolio_config, capsys):
        """Test that trading invalid ticker shows warning and doesn't execute"""
        current_time = pd.Timestamp('2024-01-02', tz='America/New_York')
        
        context = StrategyContext(
            market_data_df=market_data,
            cash_df=cash_data,
            positions_df=positions_data,
            port_notional_df=port_notional_data,
            current_time=current_time,
            executor=mock_executor,
            portfolio_config=portfolio_config
        )
        
        context.buy('INVALID_TICKER', confidence=1.0)
        
        captured = capsys.readouterr()
        assert 'Warning' in captured.out
        assert 'INVALID_TICKER' in captured.out
        mock_executor.execute_trade.assert_not_called()
    
    def test_context_with_empty_data(self, mock_executor, portfolio_config):
        """Test StrategyContext with empty DataFrames"""
        context = StrategyContext(
            market_data_df=pd.DataFrame(),
            cash_df=pd.DataFrame(),
            positions_df=pd.DataFrame(),
            port_notional_df=pd.DataFrame(),
            current_time=None,
            executor=mock_executor,
            portfolio_config=portfolio_config
        )
        
        assert context.Portfolio.cash == 0.0
        assert context.Portfolio.total_value == 0.0
        assert len(context.Portfolio.positions) == 0


class TestStrategyContextIntegration:
    """Integration tests for full workflow"""
    
    def test_full_trading_workflow(self):
        """Test complete workflow from context creation to trade execution"""
        # Setup market data
        dates = pd.date_range('2024-01-01', periods=5, freq='D', tz='America/New_York')
        market_data = pd.DataFrame({
            'timestamp': dates.repeat(2),
            'ticker': ['AAPL', 'MSFT'] * 5,
            'open_price': [100, 200] * 5,
            'high_price': [102, 202] * 5,
            'low_price': [99, 199] * 5,
            'close_price': [101, 201] * 5,
            'volume': [1000000, 2000000] * 5
        })
        
        cash_df = pd.DataFrame({'notional': [100000.0]})
        positions_df = pd.DataFrame({'ticker': ['AAPL'], 'quantity': [50]})
        port_notional_df = pd.DataFrame({'notional': [105050.0]})
        
        mock_executor = Mock()
        portfolio_config = {'id': '1', 'name': 'Test'}
        current_time = dates[2]
        
        context = StrategyContext(
            market_data_df=market_data,
            cash_df=cash_df,
            positions_df=positions_df,
            port_notional_df=port_notional_df,
            current_time=current_time,
            executor=mock_executor,
            portfolio_config=portfolio_config
        )
        
        # Access market data
        aapl = context.Market['AAPL']
        assert aapl.Exists is True
        assert aapl.Close == 101
        
        # Check portfolio
        assert context.Portfolio.positions['AAPL'] == 50
        assert context.Portfolio.cash == 100000.0
        
        # Execute trade
        context.buy('MSFT', confidence=0.5)
        
        # Verify trade execution
        assert mock_executor.execute_trade.called
        call_args = mock_executor.execute_trade.call_args[1]
        assert call_args['ticker'] == 'MSFT'
        assert call_args['signal_type'] == 'BUY'
        assert call_args['arrival_price'] == 201
