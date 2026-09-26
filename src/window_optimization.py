"""
Temporal Window Optimization

Improves Temporal IoU by:
1. Padding windows (expand by N days)
2. Merging nearby windows (gap < threshold)
3. Enforcing minimum window length
4. Smart padding based on prediction confidence
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TemporalWindow:
    """Represents a temporal window with start and end dates."""
    
    def __init__(self, start: str, end: str):
        """
        Args:
            start: ISO format datetime string (e.g., '2025-01-01T00:00:00')
            end: ISO format datetime string
        """
        self.start = pd.to_datetime(start) if isinstance(start, str) else start
        self.end = pd.to_datetime(end) if isinstance(end, str) else end
    
    def to_strings(self) -> Tuple[str, str]:
        """Convert to ISO format strings."""
        return (
            self.start.strftime('%Y-%m-%dT%H:%M:%S'),
            self.end.strftime('%Y-%m-%dT%H:%M:%S')
        )
    
    def duration_days(self) -> int:
        """Get window duration in days."""
        return (self.end - self.start).days
    
    def __repr__(self):
        return f"Window({self.start.date()} to {self.end.date()})"


class WindowOptimizer:
    """Optimize temporal windows for better IoU."""
    
    def __init__(self, 
                 pad_days: int = 5,
                 merge_gap_days: int = 3,
                 min_window_days: int = 5,
                 high_confidence_threshold: float = 0.6):
        """
        Args:
            pad_days: Days to pad on each side of window
            merge_gap_days: Merge windows with gap < this many days
            min_window_days: Minimum window length
            high_confidence_threshold: Only pad windows with prob > this
        """
        self.pad_days = pad_days
        self.merge_gap_days = merge_gap_days
        self.min_window_days = min_window_days
        self.high_confidence_threshold = high_confidence_threshold
    
    def optimize_windows(self, 
                        start_str: str, 
                        end_str: str, 
                        confidence: float) -> Tuple[str, str]:
        """
        Optimize a single window.
        
        Args:
            start_str: Window start (ISO format)
            end_str: Window end (ISO format)
            confidence: Prediction confidence (0-1)
            
        Returns:
            (optimized_start, optimized_end) as ISO strings
        """
        # Empty window
        if not start_str or not end_str:
            return '', ''
        
        try:
            window = TemporalWindow(start_str, end_str)
            
            # Step 1: Enforce minimum window length
            if window.duration_days() < self.min_window_days:
                center = window.start + (window.end - window.start) / 2
                half_duration = timedelta(days=self.min_window_days / 2)
                window.start = center - half_duration
                window.end = center + half_duration
            
            # Step 2: Pad windows for high-confidence predictions
            if confidence >= self.high_confidence_threshold:
                pad = timedelta(days=self.pad_days)
                window.start -= pad
                window.end += pad
            
            return window.to_strings()
        
        except Exception as e:
            logger.warning(f"Error optimizing window: {e}")
            return start_str, end_str
    
    def merge_windows(self, windows: List[TemporalWindow]) -> List[TemporalWindow]:
        """
        Merge windows with small gaps between them.
        
        Args:
            windows: List of TemporalWindow objects
            
        Returns:
            List of merged windows
        """
        if len(windows) <= 1:
            return windows
        
        # Sort by start date
        sorted_windows = sorted(windows, key=lambda w: w.start)
        merged = [sorted_windows[0]]
        
        for current in sorted_windows[1:]:
            last = merged[-1]
            gap = (current.start - last.end).days
            
            # Merge if gap is small
            if gap < self.merge_gap_days:
                # Extend last window to include current
                last.end = max(last.end, current.end)
            else:
                # Keep as separate window
                merged.append(current)
        
        return merged
    
    def optimize_batch(self, 
                      df: pd.DataFrame,
                      start_col: str = 'suspicious_start',
                      end_col: str = 'suspicious_end',
                      confidence_col: str = 'is_mule') -> pd.DataFrame:
        """
        Optimize all windows in a dataframe.
        
        Args:
            df: DataFrame with window columns
            start_col: Name of start column
            end_col: Name of end column
            confidence_col: Name of confidence column
            
        Returns:
            DataFrame with optimized windows
        """
        df_opt = df.copy()
        
        optimized_starts = []
        optimized_ends = []
        
        for idx, row in df.iterrows():
            start = row[start_col]
            end = row[end_col]
            confidence = row[confidence_col]
            
            opt_start, opt_end = self.optimize_windows(start, end, confidence)
            optimized_starts.append(opt_start)
            optimized_ends.append(opt_end)
        
        df_opt[start_col] = optimized_starts
        df_opt[end_col] = optimized_ends
        
        logger.info(f"Optimized {len(df)} windows")
        return df_opt


def apply_window_optimization(submission_path: str,
                             output_path: str,
                             pad_days: int = 5,
                             merge_gap_days: int = 3,
                             min_window_days: int = 5,
                             high_confidence_threshold: float = 0.6) -> None:
    """
    Apply window optimization to a submission file.
    
    Args:
        submission_path: Path to input submission CSV
        output_path: Path to output optimized submission CSV
        pad_days: Days to pad on each side
        merge_gap_days: Merge windows with gap < this
        min_window_days: Minimum window length
        high_confidence_threshold: Only pad if confidence > this
    """
    logger.info(f"Loading submission from {submission_path}")
    df = pd.read_csv(submission_path, keep_default_na=False)
    
    logger.info(f"Optimizing {len(df)} windows...")
    optimizer = WindowOptimizer(
        pad_days=pad_days,
        merge_gap_days=merge_gap_days,
        min_window_days=min_window_days,
        high_confidence_threshold=high_confidence_threshold
    )
    
    df_optimized = optimizer.optimize_batch(df)
    
    logger.info(f"Saving optimized submission to {output_path}")
    df_optimized.to_csv(output_path, index=False)
    
    # Statistics
    original_windows = (df['suspicious_start'] != '').sum()
    optimized_windows = (df_optimized['suspicious_start'] != '').sum()
    
    logger.info(f"Original windows: {original_windows}")
    logger.info(f"Optimized windows: {optimized_windows}")
    logger.info(f"Optimization complete!")
