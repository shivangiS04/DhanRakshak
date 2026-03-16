"""
Data Loader for Mule Account Detection AML Solution

This module provides comprehensive data loading, validation, and integration
for the mule account detection system. It handles:
- Loading multiple data sources (customers, accounts, transactions, labels)
- Lazy loading for 400M transaction records
- Data quality checks and missing value handling
- Integration of data sources by account_id
- Temporal consistency validation
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator, Iterator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LazyTransactionLoader:
    """
    Lazy loader for large transaction datasets (400M+ records).
    
    Implements memory-efficient streaming of transaction data using generators
    and chunked processing to avoid loading entire dataset into memory.
    """
    
    def __init__(self, transactions_dir: Path, chunk_size: int = 100000):
        """
        Initialize LazyTransactionLoader.
        
        Args:
            transactions_dir: Path to directory containing transaction parquet files
            chunk_size: Number of records to load per chunk (default: 100k)
        """
        self.transactions_dir = Path(transactions_dir)
        self.chunk_size = chunk_size
        
        # Look for parquet files in batch subfolders or directly in the directory
        self.parquet_files = sorted(self.transactions_dir.glob('*.parquet'))
        
        # If no files found directly, look in batch subfolders
        if not self.parquet_files:
            batch_files = sorted(self.transactions_dir.glob('batch-*/part_*.parquet'))
            if batch_files:
                self.parquet_files = batch_files
                logger.info(f"Found parquet files in batch subfolders")
        
        self.total_records = 0
        self.total_files = len(self.parquet_files)
        
        if not self.parquet_files:
            raise FileNotFoundError(f"No parquet files found in {transactions_dir} or batch subfolders")
        
        logger.info(f"LazyTransactionLoader initialized with {self.total_files} files, chunk_size={chunk_size}")
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in GB.
        
        Returns:
            float: Memory usage in GB
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    
    def iterate_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """
        Iterate through transaction data in chunks.
        
        Yields:
            pd.DataFrame: Chunk of transaction data
        """
        chunk_count = 0
        current_chunk = []
        current_size = 0
        
        for file_path in self.parquet_files:
            logger.info(f"Processing file: {file_path.name}")
            
            # Read parquet file in chunks
            parquet_file = pd.read_parquet(file_path)
            
            for idx in range(0, len(parquet_file), self.chunk_size):
                chunk = parquet_file.iloc[idx:idx + self.chunk_size].copy()
                current_chunk.append(chunk)
                current_size += len(chunk)
                self.total_records += len(chunk)
                
                # Yield when chunk size reached
                if current_size >= self.chunk_size:
                    combined_chunk = pd.concat(current_chunk, ignore_index=True)
                    chunk_count += 1
                    mem_usage = self.get_memory_usage()
                    logger.info(f"Yielding chunk {chunk_count}: {len(combined_chunk)} records, Memory: {mem_usage:.2f}GB")
                    yield combined_chunk
                    current_chunk = []
                    current_size = 0
        
        # Yield remaining data
        if current_chunk:
            combined_chunk = pd.concat(current_chunk, ignore_index=True)
            chunk_count += 1
            mem_usage = self.get_memory_usage()
            logger.info(f"Yielding final chunk {chunk_count}: {len(combined_chunk)} records, Memory: {mem_usage:.2f}GB")
            yield combined_chunk
        
        logger.info(f"Completed iteration: {chunk_count} chunks, {self.total_records} total records")
    
    def iterate_by_account(self, account_ids: Optional[List[str]] = None) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        """
        Iterate through transactions grouped by account_id.
        
        Args:
            account_ids: Optional list of account IDs to filter. If None, all accounts are included.
            
        Yields:
            Tuple[str, pd.DataFrame]: (account_id, transactions_for_account)
        """
        account_transactions = {}
        
        for chunk in self.iterate_chunks():
            # Filter by account_ids if provided
            if account_ids is not None:
                chunk = chunk[chunk['account_id'].isin(account_ids)]
            
            # Group by account_id
            for account_id, group in chunk.groupby('account_id'):
                if account_id not in account_transactions:
                    account_transactions[account_id] = []
                account_transactions[account_id].append(group)
        
        # Yield accumulated transactions for each account
        for account_id, txn_list in account_transactions.items():
            combined_txns = pd.concat(txn_list, ignore_index=True)
            yield account_id, combined_txns
    
    def filter_transactions(self, predicate) -> Generator[pd.DataFrame, None, None]:
        """
        Filter transactions based on a predicate function.
        
        Args:
            predicate: Function that takes a DataFrame and returns filtered DataFrame
            
        Yields:
            pd.DataFrame: Filtered chunks
        """
        for chunk in self.iterate_chunks():
            filtered_chunk = predicate(chunk)
            if len(filtered_chunk) > 0:
                yield filtered_chunk
    
    def aggregate_transactions(self, agg_func) -> Any:
        """
        Aggregate transactions across all chunks.
        
        Args:
            agg_func: Function that takes a DataFrame and returns aggregated result
            
        Returns:
            Aggregated result
        """
        results = []
        for chunk in self.iterate_chunks():
            result = agg_func(chunk)
            results.append(result)
        
        # Combine results (assumes agg_func returns compatible types)
        if results:
            if isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            elif isinstance(results[0], (int, float)):
                return sum(results)
            else:
                return results
        return None
    
    def get_account_transaction_count(self) -> Dict[str, int]:
        """
        Get transaction count per account without loading all data.
        
        Returns:
            Dict[str, int]: Mapping of account_id to transaction count
        """
        account_counts = {}
        
        for chunk in self.iterate_chunks():
            counts = chunk['account_id'].value_counts()
            for account_id, count in counts.items():
                account_counts[account_id] = account_counts.get(account_id, 0) + count
        
        return account_counts
    
    def get_temporal_range(self) -> Tuple[datetime, datetime]:
        """
        Get temporal range of transactions.
        
        Returns:
            Tuple[datetime, datetime]: (min_date, max_date)
        """
        min_date = None
        max_date = None
        date_col = None
        
        # Find date column
        for chunk in self.iterate_chunks():
            date_cols = chunk.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                date_col = date_cols[0]
                chunk_min = chunk[date_col].min()
                chunk_max = chunk[date_col].max()
                
                if min_date is None or chunk_min < min_date:
                    min_date = chunk_min
                if max_date is None or chunk_max > max_date:
                    max_date = chunk_max
        
        return (min_date, max_date) if min_date and max_date else (None, None)


@dataclass
class DataQualityReport:
    """Data quality report with statistics and issues"""
    total_records: Dict[str, int] = field(default_factory=dict)
    missing_values: Dict[str, Dict[str, int]] = field(default_factory=dict)
    data_types: Dict[str, Dict[str, str]] = field(default_factory=dict)
    temporal_range: Dict[str, Tuple[datetime, datetime]] = field(default_factory=dict)
    duplicate_records: Dict[str, int] = field(default_factory=dict)
    invalid_records: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'total_records': self.total_records,
            'missing_values': self.missing_values,
            'data_types': self.data_types,
            'temporal_range': {
                k: (v[0].isoformat(), v[1].isoformat()) 
                for k, v in self.temporal_range.items()
            },
            'duplicate_records': self.duplicate_records,
            'invalid_records': self.invalid_records,
            'warnings': self.warnings,
            'errors': self.errors
        }


class DataLoader:
    """
    Main data loader class for mule account detection.
    
    Handles loading, validation, and integration of multiple data sources
    with support for lazy loading of large transaction datasets.
    """
    
    def __init__(self, data_dir: str, output_dir: str = 'data/processed'):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to directory containing all data files
            output_dir: Path to output directory for processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.customers = None
        self.accounts = None
        self.demographics = None
        self.transactions = None
        self.transactions_additional = None
        self.train_labels = None
        self.test_accounts = None
        self.branch = None
        self.product_details = None
        self.customer_account_linkage = None
        
        # Lazy transaction loader
        self.lazy_transaction_loader = None
        
        # Quality report
        self.quality_report = DataQualityReport()
        
        logger.info(f"DataLoader initialized with data_dir: {data_dir}")
    
    def load_all_data(self) -> bool:
        """
        Load all data sources.
        
        Returns:
            bool: True if all data loaded successfully, False otherwise
        """
        try:
            logger.info("Starting data loading...")
            
            # Load main data sources
            self.load_customers()
            self.load_accounts()
            self.load_demographics()
            self.load_transactions()
            self.load_transactions_additional()
            self.load_train_labels()
            self.load_test_accounts()
            
            # Load reference data
            self.load_branch()
            self.load_product_details()
            self.load_customer_account_linkage()
            
            logger.info("All data sources loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            self.quality_report.errors.append(f"Data loading failed: {str(e)}")
            return False
    
    def load_customers(self) -> None:
        """Load customer data"""
        try:
            file_path = self.data_dir / 'customers.parquet'
            if not file_path.exists():
                raise FileNotFoundError(f"customers.parquet not found at {file_path}")
            
            self.customers = pd.read_parquet(file_path)
            logger.info(f"Loaded customers: {len(self.customers)} rows, {len(self.customers.columns)} columns")
            
            # Quality checks
            self.quality_report.total_records['customers'] = len(self.customers)
            self._check_data_quality('customers', self.customers)
            
        except Exception as e:
            logger.error(f"Error loading customers: {str(e)}")
            self.quality_report.errors.append(f"Failed to load customers: {str(e)}")
    
    def load_accounts(self) -> None:
        """Load account data"""
        try:
            file_path = self.data_dir / 'accounts.parquet'
            if not file_path.exists():
                raise FileNotFoundError(f"accounts.parquet not found at {file_path}")
            
            self.accounts = pd.read_parquet(file_path)
            logger.info(f"Loaded accounts: {len(self.accounts)} rows, {len(self.accounts.columns)} columns")
            
            # Quality checks
            self.quality_report.total_records['accounts'] = len(self.accounts)
            self._check_data_quality('accounts', self.accounts)
            
        except Exception as e:
            logger.error(f"Error loading accounts: {str(e)}")
            self.quality_report.errors.append(f"Failed to load accounts: {str(e)}")
    
    def load_demographics(self) -> None:
        """Load demographics data"""
        try:
            file_path = self.data_dir / 'demographics.parquet'
            if not file_path.exists():
                raise FileNotFoundError(f"demographics.parquet not found at {file_path}")
            
            self.demographics = pd.read_parquet(file_path)
            logger.info(f"Loaded demographics: {len(self.demographics)} rows, {len(self.demographics.columns)} columns")
            
            # Quality checks
            self.quality_report.total_records['demographics'] = len(self.demographics)
            self._check_data_quality('demographics', self.demographics)
            
        except Exception as e:
            logger.error(f"Error loading demographics: {str(e)}")
            self.quality_report.errors.append(f"Failed to load demographics: {str(e)}")
    
    def load_transactions(self) -> None:
        """Load transaction data with lazy loading support"""
        try:
            transactions_dir = self.data_dir / 'transactions'
            if not transactions_dir.exists():
                raise FileNotFoundError(f"transactions directory not found at {transactions_dir}")
            
            # Initialize lazy transaction loader
            self.lazy_transaction_loader = LazyTransactionLoader(transactions_dir, chunk_size=100000)
            logger.info(f"Lazy transaction loader initialized for {self.lazy_transaction_loader.total_files} files")
            
            # Get metadata without loading all data
            account_counts = self.lazy_transaction_loader.get_account_transaction_count()
            total_transactions = sum(account_counts.values())
            
            logger.info(f"Total transactions: {total_transactions} rows across {len(account_counts)} accounts")
            
            # Quality checks - get temporal range
            self.quality_report.total_records['transactions'] = total_transactions
            min_date, max_date = self.lazy_transaction_loader.get_temporal_range()
            if min_date and max_date:
                self.quality_report.temporal_range['transactions_date'] = (min_date, max_date)
                logger.info(f"Transaction temporal range: {min_date} to {max_date}")
            
            # Log memory usage
            mem_usage = self.lazy_transaction_loader.get_memory_usage()
            logger.info(f"Current memory usage: {mem_usage:.2f}GB")
            
        except Exception as e:
            logger.error(f"Error loading transactions: {str(e)}")
            self.quality_report.errors.append(f"Failed to load transactions: {str(e)}")
    
    def load_transactions_additional(self) -> None:
        """Load additional transaction data"""
        try:
            transactions_dir = self.data_dir / 'transactions_additional'
            if not transactions_dir.exists():
                logger.warning(f"transactions_additional directory not found at {transactions_dir}")
                return
            
            # Find all parquet files
            parquet_files = sorted(transactions_dir.glob('*.parquet'))
            if not parquet_files:
                logger.warning(f"No parquet files found in {transactions_dir}")
                return
            
            logger.info(f"Found {len(parquet_files)} additional transaction batch files")
            
            # Load all batches
            dfs = []
            for file_path in parquet_files:
                df = pd.read_parquet(file_path)
                dfs.append(df)
                logger.info(f"Loaded {file_path.name}: {len(df)} rows")
            
            # Concatenate all batches
            self.transactions_additional = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total additional transactions loaded: {len(self.transactions_additional)} rows")
            
            # Quality checks
            self.quality_report.total_records['transactions_additional'] = len(self.transactions_additional)
            self._check_data_quality('transactions_additional', self.transactions_additional)
            
        except Exception as e:
            logger.error(f"Error loading additional transactions: {str(e)}")
            self.quality_report.errors.append(f"Failed to load additional transactions: {str(e)}")
    
    def load_train_labels(self) -> None:
        """Load training labels"""
        try:
            file_path = self.data_dir / 'train_labels.parquet'
            if not file_path.exists():
                raise FileNotFoundError(f"train_labels.parquet not found at {file_path}")
            
            self.train_labels = pd.read_parquet(file_path)
            logger.info(f"Loaded train_labels: {len(self.train_labels)} rows, {len(self.train_labels.columns)} columns")
            
            # Quality checks
            self.quality_report.total_records['train_labels'] = len(self.train_labels)
            self._check_data_quality('train_labels', self.train_labels)
            
        except Exception as e:
            logger.error(f"Error loading train_labels: {str(e)}")
            self.quality_report.errors.append(f"Failed to load train_labels: {str(e)}")
    
    def load_test_accounts(self) -> None:
        """Load test accounts"""
        try:
            file_path = self.data_dir / 'test_accounts.parquet'
            if not file_path.exists():
                raise FileNotFoundError(f"test_accounts.parquet not found at {file_path}")
            
            self.test_accounts = pd.read_parquet(file_path)
            logger.info(f"Loaded test_accounts: {len(self.test_accounts)} rows, {len(self.test_accounts.columns)} columns")
            
            # Quality checks
            self.quality_report.total_records['test_accounts'] = len(self.test_accounts)
            self._check_data_quality('test_accounts', self.test_accounts)
            
        except Exception as e:
            logger.error(f"Error loading test_accounts: {str(e)}")
            self.quality_report.errors.append(f"Failed to load test_accounts: {str(e)}")
    
    def load_branch(self) -> None:
        """Load branch reference data"""
        try:
            file_path = self.data_dir / 'branch.parquet'
            if not file_path.exists():
                logger.warning(f"branch.parquet not found at {file_path}")
                return
            
            self.branch = pd.read_parquet(file_path)
            logger.info(f"Loaded branch: {len(self.branch)} rows, {len(self.branch.columns)} columns")
            
            # Quality checks
            self.quality_report.total_records['branch'] = len(self.branch)
            self._check_data_quality('branch', self.branch)
            
        except Exception as e:
            logger.error(f"Error loading branch: {str(e)}")
            self.quality_report.errors.append(f"Failed to load branch: {str(e)}")
    
    def load_product_details(self) -> None:
        """Load product details reference data"""
        try:
            file_path = self.data_dir / 'product_details.parquet'
            if not file_path.exists():
                logger.warning(f"product_details.parquet not found at {file_path}")
                return
            
            self.product_details = pd.read_parquet(file_path)
            logger.info(f"Loaded product_details: {len(self.product_details)} rows, {len(self.product_details.columns)} columns")
            
            # Quality checks
            self.quality_report.total_records['product_details'] = len(self.product_details)
            self._check_data_quality('product_details', self.product_details)
            
        except Exception as e:
            logger.error(f"Error loading product_details: {str(e)}")
            self.quality_report.errors.append(f"Failed to load product_details: {str(e)}")
    
    def load_customer_account_linkage(self) -> None:
        """Load customer-account linkage reference data"""
        try:
            file_path = self.data_dir / 'customer_account_linkage.parquet'
            if not file_path.exists():
                logger.warning(f"customer_account_linkage.parquet not found at {file_path}")
                return
            
            self.customer_account_linkage = pd.read_parquet(file_path)
            logger.info(f"Loaded customer_account_linkage: {len(self.customer_account_linkage)} rows")
            
            # Quality checks
            self.quality_report.total_records['customer_account_linkage'] = len(self.customer_account_linkage)
            self._check_data_quality('customer_account_linkage', self.customer_account_linkage)
            
        except Exception as e:
            logger.error(f"Error loading customer_account_linkage: {str(e)}")
            self.quality_report.errors.append(f"Failed to load customer_account_linkage: {str(e)}")
    
    def _check_data_quality(self, source_name: str, df: pd.DataFrame) -> None:
        """
        Perform data quality checks on a dataframe.
        
        Args:
            source_name: Name of the data source
            df: DataFrame to check
        """
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            self.quality_report.missing_values[source_name] = missing[missing > 0].to_dict()
            logger.warning(f"{source_name}: {missing.sum()} missing values found")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.quality_report.duplicate_records[source_name] = duplicates
            logger.warning(f"{source_name}: {duplicates} duplicate records found")
        
        # Record data types
        self.quality_report.data_types[source_name] = df.dtypes.astype(str).to_dict()
        
        # Check temporal range if date columns exist
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            for col in date_cols:
                min_date = df[col].min()
                max_date = df[col].max()
                self.quality_report.temporal_range[f"{source_name}_{col}"] = (min_date, max_date)
                logger.info(f"{source_name}.{col}: {min_date} to {max_date}")
    
    def get_transactions_for_account(self, account_id: str) -> Optional[pd.DataFrame]:
        """
        Get transactions for a specific account using lazy loader.
        
        Args:
            account_id: Account ID to retrieve transactions for
            
        Returns:
            DataFrame with transactions for the account or None
        """
        if self.lazy_transaction_loader is None:
            logger.warning("Lazy transaction loader not initialized")
            return None
        
        try:
            for acc_id, txns in self.lazy_transaction_loader.iterate_by_account([account_id]):
                if acc_id == account_id:
                    return txns
            return None
        except Exception as e:
            logger.error(f"Error retrieving transactions for account {account_id}: {str(e)}")
            return None
    
    def iterate_transactions_by_account(self, account_ids: Optional[List[str]] = None) -> Generator[Tuple[str, pd.DataFrame], None, None]:
        """
        Iterate through transactions grouped by account_id.
        
        Args:
            account_ids: Optional list of account IDs to filter
            
        Yields:
            Tuple[str, pd.DataFrame]: (account_id, transactions_for_account)
        """
        if self.lazy_transaction_loader is None:
            logger.warning("Lazy transaction loader not initialized")
            return
        
        try:
            for account_id, txns in self.lazy_transaction_loader.iterate_by_account(account_ids):
                yield account_id, txns
        except Exception as e:
            logger.error(f"Error iterating transactions: {str(e)}")
    
    def iterate_transaction_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """
        Iterate through transaction data in chunks.
        
        Yields:
            pd.DataFrame: Chunk of transaction data
        """
        if self.lazy_transaction_loader is None:
            logger.warning("Lazy transaction loader not initialized")
            return
        
        try:
            for chunk in self.lazy_transaction_loader.iterate_chunks():
                yield chunk
        except Exception as e:
            logger.error(f"Error iterating transaction chunks: {str(e)}")
    
    def integrate_data(self) -> Optional[pd.DataFrame]:
        """
        Integrate all data sources by account_id.
        
        Note: When using lazy loading, transactions are not fully loaded into memory.
        Use iterate_transactions_by_account() for memory-efficient processing.
        
        Returns:
            Integrated DataFrame or None if integration fails
        """
        try:
            logger.info("Starting data integration...")
            
            if self.accounts is None:
                raise ValueError("Accounts data not loaded")
            
            # Check if using lazy loading
            if self.lazy_transaction_loader is not None and self.transactions is None:
                logger.info("Using lazy transaction loader - transactions not fully loaded into memory")
                # For lazy loading, we integrate account metadata only
                # Transaction integration happens during feature extraction
                integrated = self.accounts.copy()
            elif self.transactions is None:
                raise ValueError("Required data sources not loaded")
            else:
                integrated = self.accounts.copy()
            
            logger.info(f"Starting with accounts: {len(integrated)} rows")
            
            # Join with customers if available
            if self.customers is not None and 'customer_id' in integrated.columns:
                integrated = integrated.merge(
                    self.customers,
                    on='customer_id',
                    how='left',
                    suffixes=('', '_customer')
                )
                logger.info(f"After joining customers: {len(integrated)} rows")
            
            # Join with demographics if available
            if self.demographics is not None and 'customer_id' in integrated.columns:
                integrated = integrated.merge(
                    self.demographics,
                    on='customer_id',
                    how='left',
                    suffixes=('', '_demo')
                )
                logger.info(f"After joining demographics: {len(integrated)} rows")
            
            # Join with train labels if available
            if self.train_labels is not None:
                integrated = integrated.merge(
                    self.train_labels,
                    on='account_id',
                    how='left'
                )
                logger.info(f"After joining train_labels: {len(integrated)} rows")
            
            logger.info(f"Data integration complete: {len(integrated)} rows, {len(integrated.columns)} columns")
            return integrated
            
        except Exception as e:
            logger.error(f"Error integrating data: {str(e)}")
            self.quality_report.errors.append(f"Data integration failed: {str(e)}")
            return None
    
    def validate_temporal_consistency(self) -> bool:
        """
        Validate temporal consistency across all data sources.
        
        Returns:
            bool: True if temporal consistency is valid
        """
        try:
            logger.info("Validating temporal consistency...")
            
            # Expected temporal range: 2020-2025
            expected_start = datetime(2020, 1, 1)
            expected_end = datetime(2025, 12, 31)
            
            # Check transactions temporal range
            if self.lazy_transaction_loader is not None:
                # Use lazy loader to get temporal range
                min_date, max_date = self.lazy_transaction_loader.get_temporal_range()
                if min_date and max_date:
                    if min_date < expected_start or max_date > expected_end:
                        msg = f"transactions: Out of expected range ({min_date} to {max_date})"
                        logger.warning(msg)
                        self.quality_report.warnings.append(msg)
                    else:
                        logger.info(f"transactions: Valid range ({min_date} to {max_date})")
            elif self.transactions is not None:
                date_cols = self.transactions.select_dtypes(include=['datetime64']).columns
                for col in date_cols:
                    min_date = self.transactions[col].min()
                    max_date = self.transactions[col].max()
                    
                    if pd.isna(min_date) or pd.isna(max_date):
                        logger.warning(f"transactions.{col}: Contains NaT values")
                        continue
                    
                    if min_date < expected_start or max_date > expected_end:
                        msg = f"transactions.{col}: Out of expected range ({min_date} to {max_date})"
                        logger.warning(msg)
                        self.quality_report.warnings.append(msg)
                    else:
                        logger.info(f"transactions.{col}: Valid range ({min_date} to {max_date})")
            
            logger.info("Temporal consistency validation complete")
            return True
            
        except Exception as e:
            logger.error(f"Error validating temporal consistency: {str(e)}")
            self.quality_report.errors.append(f"Temporal validation failed: {str(e)}")
            return False
    
    def generate_quality_report(self, output_file: str = 'data_quality_report.txt') -> str:
        """
        Generate data quality report.
        
        Args:
            output_file: Output file path
            
        Returns:
            str: Path to generated report
        """
        try:
            report_path = self.output_dir / output_file
            
            with open(report_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("DATA QUALITY REPORT - MULE ACCOUNT DETECTION\n")
                f.write("=" * 80 + "\n\n")
                
                # Summary
                f.write("SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Report Generated: {datetime.now().isoformat()}\n")
                f.write(f"Data Directory: {self.data_dir}\n")
                f.write(f"Total Data Sources: {len(self.quality_report.total_records)}\n")
                f.write(f"Total Records Loaded: {sum(self.quality_report.total_records.values())}\n\n")
                
                # Record counts
                f.write("RECORD COUNTS BY SOURCE\n")
                f.write("-" * 80 + "\n")
                for source, count in sorted(self.quality_report.total_records.items()):
                    f.write(f"{source:30s}: {count:>15,} rows\n")
                f.write("\n")
                
                # Missing values
                if self.quality_report.missing_values:
                    f.write("MISSING VALUES\n")
                    f.write("-" * 80 + "\n")
                    for source, missing_dict in self.quality_report.missing_values.items():
                        f.write(f"\n{source}:\n")
                        for col, count in missing_dict.items():
                            f.write(f"  {col:30s}: {count:>10,} missing\n")
                    f.write("\n")
                
                # Duplicates
                if self.quality_report.duplicate_records:
                    f.write("DUPLICATE RECORDS\n")
                    f.write("-" * 80 + "\n")
                    for source, count in self.quality_report.duplicate_records.items():
                        f.write(f"{source:30s}: {count:>15,} duplicates\n")
                    f.write("\n")
                
                # Temporal ranges
                if self.quality_report.temporal_range:
                    f.write("TEMPORAL RANGES\n")
                    f.write("-" * 80 + "\n")
                    for source, (min_date, max_date) in self.quality_report.temporal_range.items():
                        f.write(f"{source:40s}: {min_date} to {max_date}\n")
                    f.write("\n")
                
                # Warnings
                if self.quality_report.warnings:
                    f.write("WARNINGS\n")
                    f.write("-" * 80 + "\n")
                    for warning in self.quality_report.warnings:
                        f.write(f"  - {warning}\n")
                    f.write("\n")
                
                # Errors
                if self.quality_report.errors:
                    f.write("ERRORS\n")
                    f.write("-" * 80 + "\n")
                    for error in self.quality_report.errors:
                        f.write(f"  - {error}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
            
            logger.info(f"Quality report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating quality report: {str(e)}")
            return ""
    
    def save_integrated_data(self, output_file: str = 'integrated_data.parquet') -> Optional[str]:
        """
        Save integrated data to parquet file.
        
        Args:
            output_file: Output file name
            
        Returns:
            str: Path to saved file or None if save fails
        """
        try:
            integrated = self.integrate_data()
            if integrated is None:
                return None
            
            output_path = self.output_dir / output_file
            integrated.to_parquet(output_path, index=False)
            logger.info(f"Integrated data saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving integrated data: {str(e)}")
            self.quality_report.errors.append(f"Failed to save integrated data: {str(e)}")
            return None


def main():
    """Main execution function"""
    # Configuration
    data_dir = '/Users/shivangisingh/Desktop/archive/'
    output_dir = 'data/processed'
    
    # Initialize loader
    loader = DataLoader(data_dir, output_dir)
    
    # Load all data
    if not loader.load_all_data():
        logger.error("Failed to load all data sources")
        return False
    
    # Validate temporal consistency
    if not loader.validate_temporal_consistency():
        logger.warning("Temporal consistency validation failed")
    
    # Integrate data
    integrated = loader.integrate_data()
    if integrated is None:
        logger.error("Failed to integrate data")
        return False
    
    # Save integrated data
    saved_path = loader.save_integrated_data()
    if not saved_path:
        logger.error("Failed to save integrated data")
        return False
    
    # Generate quality report
    report_path = loader.generate_quality_report()
    logger.info(f"Quality report: {report_path}")
    
    logger.info("Phase 1 data ingestion complete")
    return True


if __name__ == '__main__':
    main()
