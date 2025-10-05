"""
Space Weather Data Pipeline
Handles DSCOVR RTSW and ACE data collection and processing
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SpaceWeatherPipeline:
    """
    Pipeline for space weather data (DSCOVR and ACE)
    """
    
    def __init__(self, api_key="JU1cZhz9EwpjHaCURemRiHRZNhwrHSAB4a3zkL4o"):
        self.api_key = api_key
        self.data_sources = {
            'DSCOVR RTSW': 'Solar wind data from Deep Space Climate Observatory',
            'ACE': 'Solar energetic particle data from Advanced Composition Explorer'
        }
    
    def fetch_dscovr_data(self, start_date, end_date):
        """Fetch DSCOVR Real-Time Solar Wind data"""
        base_url = "https://api.nasa.gov/DONKI/FLR"
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "api_key": self.api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching DSCOVR data: {e}")
            return []
    
    def fetch_ace_data(self, start_date, end_date):
        """Fetch ACE (Advanced Composition Explorer) data"""
        base_url = "https://api.nasa.gov/DONKI/CME"
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "api_key": self.api_key
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching ACE data: {e}")
            return []
    
    def process_dscovr_data(self, raw_data):
        """Process DSCOVR data into structured DataFrame"""
        if not raw_data:
            return pd.DataFrame()
        
        processed_records = []
        
        for event in raw_data:
            record = {
                "event_id": event.get("flrID"),
                "start_time": event.get("beginTime"),
                "peak_time": event.get("peakTime"),
                "end_time": event.get("endTime"),
                "class_type": event.get("classType"),
                "source": event.get("sourceLocation"),
                "active_region_num": event.get("activeRegionNum"),
                "link": event.get("link")
            }
            processed_records.append(record)
        
        df = pd.DataFrame(processed_records)
        
        # Convert time columns to datetime
        for col in ["start_time", "peak_time", "end_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        
        # Extract additional features
        if "class_type" in df.columns:
            df["class_letter"] = df["class_type"].str[0]
            df["class_number"] = pd.to_numeric(df["class_type"].str[1:], errors='coerce')
        
        # Calculate duration
        if all(col in df.columns for col in ["start_time", "end_time"]):
            df["duration_hours"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 3600
        
        return df
    
    def process_ace_data(self, raw_data):
        """Process ACE data into structured DataFrame"""
        if not raw_data:
            return pd.DataFrame()
        
        processed_records = []
        
        for event in raw_data:
            record = {
                "event_id": event.get("activityID"),
                "start_time": event.get("time21_5"),
                "type": event.get("type"),
                "speed": event.get("speed"),
                "angle": event.get("angle"),
                "is_most_accurate": event.get("isMostAccurate"),
                "link": event.get("link")
            }
            processed_records.append(record)
        
        df = pd.DataFrame(processed_records)
        
        # Convert time columns to datetime
        if "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        
        # Convert numeric columns
        numeric_cols = ["speed", "angle"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def run_pipeline(self, start_date=None, end_date=None, save_data=True):
        """
        Run the complete space weather pipeline
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            save_data (bool): Whether to save processed data
            
        Returns:
            dict: Processed data and metadata
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        date_range = f"{start_date}_to_{end_date}"
        
        print("=== Space Weather Pipeline Execution ===")
        print(f"Date Range: {start_date} to {end_date}")
        
        # Fetch data
        print("\nStep 1: Fetching space weather data...")
        dscovr_data = self.fetch_dscovr_data(start_date, end_date)
        ace_data = self.fetch_ace_data(start_date, end_date)
        
        # Process data
        print("\nStep 2: Processing space weather data...")
        dscovr_df = self.process_dscovr_data(dscovr_data)
        ace_df = self.process_ace_data(ace_data)
        
        # Save data if requested
        if save_data:
            print("\nStep 3: Saving processed data...")
            self._save_data(dscovr_df, "dscovr", date_range)
            self._save_data(ace_df, "ace", date_range)
        
        # Generate report
        report = {
            "data_sources": self.data_sources,
            "date_range": date_range,
            "records": {
                "dscovr": len(dscovr_df),
                "ace": len(ace_df)
            },
            "columns": {
                "dscovr": list(dscovr_df.columns),
                "ace": list(ace_df.columns)
            },
            "date_range_actual": {
                "dscovr": {
                    "start": dscovr_df["start_time"].min() if not dscovr_df.empty else None,
                    "end": dscovr_df["start_time"].max() if not dscovr_df.empty else None
                },
                "ace": {
                    "start": ace_df["start_time"].min() if not ace_df.empty else None,
                    "end": ace_df["start_time"].max() if not ace_df.empty else None
                }
            }
        }
        
        print("\n=== Space Weather Pipeline Complete ===")
        print(f"DSCOVR Records: {report['records']['dscovr']}")
        print(f"ACE Records: {report['records']['ace']}")
        
        return {
            "dscovr_data": dscovr_df,
            "ace_data": ace_df,
            "report": report
        }
    
    def _save_data(self, df, data_type, date_range):
        """Save processed data to CSV"""
        data_dir = Path("nasa_data/space_weather")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{data_type}_{date_range.replace(':', '_')}.csv"
        filepath = data_dir / filename
        
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        
        return filepath


# Example usage
if __name__ == "__main__":
    pipeline = SpaceWeatherPipeline()
    result = pipeline.run_pipeline()