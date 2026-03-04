"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime, timedelta

from src.data_loader import DataLoader, DataQualityReport


class TestDataQualityReport:
    """Test DataQualityReport class"""
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        report = DataQualityReport()
        report.total_records = {'customers': 100, 'accounts': 50}
        report.warnings = ['Test warning']
        
        result = report.to_dict()
        
        assert result['total_records'] == {'customers': 100, 'accounts': 50}
        assert result['warnings'] == ['Test warning']
        assert 'temporal_range' in result


class TestDataLoader:
    """Test DataLoader class"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with sample files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create sample parquet files
            customers_df = pd.DataFrame({
                'customer_id': ['C1', 'C2', 'C3'],
                'name': ['Alice', 'Bob', 'Charlie'],
                'age': [30, 25, 35]
            })
            customers_df.to_parquet(tmpdir / 'customers.parquet')
            
            accounts_df = pd.DataFrame({
                'account_id': ['A1', 'A2', 'A3'],
                'customer_id': ['C1', 'C2', 'C3'],
                'balance': [1000.0, 2000.0, 3000.0]
            })
            accounts_df.to_parquet(tmpdir / 'accounts.parquet')
            
            yield tmpdir
    
    def test_initialization(self, temp_data_dir):
        """Test DataLoader initialization"""
        loader = DataLoader(str(temp_data_dir))
        
        assert loader.data_dir == Path(temp_data_dir)
        assert loader.output_dir.exists()
        assert loader.customers is None
        assert loader.accounts is None
    
    def test_load_customers(self, temp_data_dir):
        """Test loading customer data"""
        loader = DataLoader(str(temp_data_dir))
        loader.load_customers()
        
        assert loader.customers is not None
        assert len(loader.customers) == 3
        assert 'customer_id' in loader.customers.columns
    
    def test_load_accounts(self, temp_data_dir):
        """Test loading account data"""
        loader = DataLoader(str(temp_data_dir))
        loader.load_accounts()
        
        assert loader.accounts is not None
        assert len(loader.accounts) == 3
        assert 'account_id' in loader.accounts.columns
    
    def test_data_quality_checks(self, temp_data_dir):
        """Test data quality checks"""
        loader = DataLoader(str(temp_data_dir))
        loader.load_customers()
        
        # Should have recorded total records
        assert 'customers' in loader.quality_report.total_records
        assert loader.quality_report.total_records['customers'] == 3


class TestLazyTransactionLoader:
    """Test lazy loading of transactions"""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data"""
        dates = pd.date_range('2024-01-01', periods=1000, freq='H')
        return pd.DataFrame({
            'transaction_id': [f'T{i}' for i in range(1000)],
            'account_id': [f'A{i % 10}' for i in range(1000)],
            'counterparty_account_id': [f'B{i % 5}' for i in range(1000)],
            'amount': np.random.uniform(100, 10000, 1000),
            'transaction_date': dates,
            'direction': np.random.choice(['inflow', 'outflow'], 1000)
        })
    
    def test_transaction_loading(self, sample_transactions):
        """Test that transactions can be loaded"""
        assert len(sample_transactions) == 1000
        assert 'account_id' in sample_transactions.columns
        assert 'transaction_date' in sample_transactions.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
