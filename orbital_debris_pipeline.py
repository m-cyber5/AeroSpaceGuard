"""
Orbital Debris Data Pipeline
Handles ORDEM 3.0 orbital debris data collection and processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class OrbitalDebrisPipeline:
    """
    Pipeline for orbital debris data (ORDEM 3.0)
    """
    
    def __init__(self, api_key=None):
        self.data_sources = {
            'ORDEM 3.0': 'Orbital Debris Engineering Model'
        }
    
    def fetch_ordem_data(self, start_date, end_date):
        """
        Fetch orbital debris data from ORDEM 3.0 (simulated)
        
        Args:
            start_date (str): Start date
            end_date (str): End date
        """
        altitudes = np.arange(200, 2000, 50)
        times = pd.date_range(start=start_date, end=end_date, freq='h')
        
        data = []
        for alt in altitudes:
            for time in times:
                # Simulate debris flux (peaks around 800-1000km)
                flux = 1e-4 * np.exp(-(alt - 900)**2 / 50000)
                
                # Spatial density
                density = flux * 1e-6
                
                # Velocity (typical orbital velocities)
                velocity = np.sqrt(3.986e14 / (6371 + alt) * 1e-3)  # km/s
                
                # Risk index (kinetic energy proxy)
                risk = 0.5 * density * velocity**2
                
                data.append({
                    'timestamp': time,
                    'altitude_km': alt,
                    'debris_flux': flux,
                    'spatial_density': density,
                    'velocity_kms': velocity,
                    'risk_index': risk
                })
        
        return pd.DataFrame(data)
    
    def process_ordem_data(self, df):
        """Process ORDEM data with enhanced features"""
        # Calculate additional risk metrics
        df["risk_index"] = df["debris_flux"] * df["velocity_kms"]**2 / 2
        
        # Identify high-risk altitude zones
        high_risk_threshold = df["risk_index"].quantile(0.9)
        df["high_risk_zone"] = df["risk_index"] > high_risk_threshold
        
        # Calculate collision probability
        df["collision_probability"] = df["spatial_density"] * df["velocity_kms"] * 1e-6
        
        # Risk categorization
        risk_bins = [0, 0.2, 0.5, 0.8, 1.0]
        risk_labels = ['Very Low', 'Low', 'Medium', 'High']
        df['risk_category'] = pd.cut(
            df['risk_index'].rank(pct=True),
            bins=risk_bins,
            labels=risk_labels
        )
        
        # Time-based analysis
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Altitude-based analysis
        df['altitude_category'] = pd.cut(
            df['altitude_km'],
            bins=[0, 400, 800, 1200, 1600, 2000],
            labels=['LEO', 'MEO_Low', 'MEO_High', 'GEO_Low', 'GEO_High']
        )
        
        return df
    
    def analyze_debris_trends(self, df):
        """Analyze orbital debris trends"""
        analysis = {}
        
        # Altitude distribution of risk
        altitude_risk = df.groupby('altitude_category')['risk_index'].mean().to_dict()
        analysis['altitude_risk'] = altitude_risk
        
        # Temporal trends
        hourly_risk = df.groupby('hour')['risk_index'].mean().to_dict()
        analysis['hourly_risk'] = hourly_risk
        
        # High-risk zones
        high_risk_zones = df[df['high_risk_zone']].groupby('altitude_km').size().to_dict()
        analysis['high_risk_zones'] = high_risk_zones
        
        # Risk distribution
        risk_dist = df['risk_category'].value_counts().to_dict()
        analysis['risk_distribution'] = risk_dist
        
        return analysis
    
    def run_pipeline(self, start_date=None, end_date=None, save_data=True):
        """
        Run the complete orbital debris pipeline
        
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
        
        print("=== Orbital Debris Pipeline Execution ===")
        print(f"Date Range: {start_date} to {end_date}")
        
        # Fetch data
        print("\nStep 1: Fetching orbital debris data...")
        raw_data = self.fetch_ordem_data(start_date, end_date)
        
        # Process data
        print("\nStep 2: Processing orbital debris data...")
        processed_data = self.process_ordem_data(raw_data)
        
        # Analyze trends
        print("\nStep 3: Analyzing debris trends...")
        trend_analysis = self.analyze_debris_trends(processed_data)
        
        # Save data if requested
        if save_data:
            print("\nStep 4: Saving processed data...")
            self._save_data(processed_data, "ordem", date_range)
        
        # Generate report
        report = {
            "data_sources": self.data_sources,
            "date_range": date_range,
            "records": len(processed_data),
            "altitude_range": {
                "min": processed_data['altitude_km'].min(),
                "max": processed_data['altitude_km'].max()
            },
            "risk_statistics": {
                "max_risk": processed_data['risk_index'].max(),
                "mean_risk": processed_data['risk_index'].mean(),
                "high_risk_records": processed_data['high_risk_zone'].sum()
            },
            "trend_analysis": trend_analysis
        }
        
        print("\n=== Orbital Debris Pipeline Complete ===")
        print(f"Total Records: {report['records']}")
        print(f"Altitude Range: {report['altitude_range']['min']} - {report['altitude_range']['max']} km")
        print(f"High Risk Records: {report['risk_statistics']['high_risk_records']}")
        
        return {
            "processed_data": processed_data,
            "trend_analysis": trend_analysis,
            "report": report
        }
    
    def _save_data(self, df, data_type, date_range):
        """Save processed data to CSV"""
        data_dir = Path("nasa_data/orbital_debris")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{data_type}_{date_range.replace(':', '_')}.csv"
        filepath = data_dir / filename
        
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        
        return filepath


# Example usage
if __name__ == "__main__":
    pipeline = OrbitalDebrisPipeline()
    result = pipeline.run_pipeline()