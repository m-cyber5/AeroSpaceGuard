"""
Atmospheric Data Pipeline
Handles GEOS-5 FP and MERRA-2 atmospheric data collection and processing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class AtmosphericPipeline:
    """
    Pipeline for atmospheric data (GEOS-5 and MERRA-2)
    """
  
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.data_sources = {
            'GEOS-5 FP': 'Global Earth Observing System Forward Processing',
            'MERRA-2': 'Modern-Era Retrospective analysis for Research and Applications, Version 2'
        }

    def fetch_geos5_data(self, variables, start_date, end_date, resolution="0.25x0.3125", pressure_levels=None):
        """
        Fetch atmospheric data from GEOS-5 FP (simulated)
        """
        if pressure_levels is None:
            pressure_levels = [1000, 850, 700, 500, 300, 200, 100]
        
        dates = pd.date_range(start=start_date, end=end_date, freq='3h')
        data = []
        
        variable_ranges = {
            'T': {'mean': 250, 'std': 30},
            'U': {'mean': 5, 'std': 15},
            'V': {'mean': 2, 'std': 10},
            'OMEGA': {'mean': 0, 'std': 0.1},
            'RH': {'mean': 50, 'std': 30},
            'QV': {'mean': 0.01, 'std': 0.01},
            'O3': {'mean': 100, 'std': 50},
            'TROPP': {'mean': 150, 'std': 30},
            'H': {'mean': 8000, 'std': 2000},
            'SLP': {'mean': 1013, 'std': 20}
        }
        
        # Coarse grid to improve performance
        sample_lats = np.arange(-90, 91, 10.0)
        sample_lons = np.arange(-180, 181, 10.0)
        total = len(dates) * len(sample_lats) * len(sample_lons)
        count = 0
        
        for date in dates:
            for lat in sample_lats:
                for lon in sample_lons:
                    count += 1
                    if count % 1000 == 0:
                        print(f"  • Generated {count}/{total} GEOS-5 points")
                    for var in variables:
                        if var in variable_ranges:
                            info = variable_ranges[var]
                            value = np.random.normal(info['mean'], info['std'])
                            if var == 'T':
                                lat_factor = np.cos(np.radians(lat))
                                season = np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                                value += 20 * lat_factor * season
                            elif var == 'O3':
                                lat_factor = abs(lat) / 90
                                value += 100 * lat_factor
                            data.append({
                                "timestamp": date,
                                "variable": var,
                                "value": value,
                                "latitude": lat,
                                "longitude": lon,
                                "pressure_level": np.random.choice(pressure_levels)
                            })
        return pd.DataFrame(data)


    def fetch_merra2_data(self, variables, start_date, end_date, resolution="0.5x0.625"):
        """
        Fetch atmospheric data from MERRA-2 (simulated)
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='6h')
        data = []
        
        variable_ranges = {
            'T': {'mean': 250, 'std': 25},
            'U': {'mean': 3, 'std': 12},
            'V': {'mean': 1, 'std': 8},
            'OMEGA': {'mean': 0, 'std': 0.05},
            'RH': {'mean': 60, 'std': 25},
            'QV': {'mean': 0.008, 'std': 0.008},
            'O3': {'mean': 120, 'std': 40},
            'TROPP': {'mean': 160, 'std': 25},
            'CLD': {'mean': 0.3, 'std': 0.3},
            'PRECTOT': {'mean': 0.0001, 'std': 0.0002}
        }
        
        sample_lats = np.arange(-90, 91, 10.0)
        sample_lons = np.arange(-180, 181, 10.0)
        total = len(dates) * len(sample_lats) * len(sample_lons)
        count = 0
        
        for date in dates:
            for lat in sample_lats:
                for lon in sample_lons:
                    count += 1
                    if count % 1000 == 0:
                        print(f"  • Generated {count}/{total} MERRA-2 points")
                    for var in variables:
                        if var in variable_ranges:
                            info = variable_ranges[var]
                            value = np.random.normal(info['mean'], info['std'])
                            if var == 'T':
                                lat_factor = np.cos(np.radians(lat))
                                season = np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                                value += 15 * lat_factor * season
                            elif var == 'O3':
                                lat_factor = abs(lat) / 90
                                value += 80 * lat_factor
                            elif var == 'CLD':
                                value = max(0, min(1, value))
                            elif var == 'PRECTOT':
                                value = max(0, value)
                            data.append({
                                "timestamp": date,
                                "variable": var,
                                "value": value,
                                "latitude": lat,
                                "longitude": lon,
                                "pressure_level": np.random.choice([1000,850,700,500,300,200,100])
                            })
        return pd.DataFrame(data)


    def process_geos5_data(self, df):
        """Process GEOS-5 data with enhanced features"""
        if df.empty:
            return df

        df_pivot = df.pivot_table(
            index=["timestamp","latitude","longitude","pressure_level"],
            columns="variable",
            values="value"
        ).reset_index()

        # Fill missing values
        for col in df_pivot.columns:
            if col not in ["timestamp","latitude","longitude","pressure_level"]:
                df_pivot[col] = df_pivot[col].fillna(df_pivot[col].mean())

        if "U" in df_pivot and "V" in df_pivot:
            df_pivot["wind_speed"] = np.sqrt(df_pivot["U"]**2 + df_pivot["V"]**2)
            df_pivot["wind_direction"] = (np.arctan2(df_pivot["V"], df_pivot["U"]) * 180/np.pi + 360) % 360
            df_pivot = df_pivot.sort_values(["latitude","longitude","timestamp","pressure_level"])
            df_pivot["wind_shear"] = df_pivot.groupby(["latitude","longitude","timestamp"])["wind_speed"].diff().abs().fillna(0)

        if "TROPP" in df_pivot and "wind_speed" in df_pivot:
            df_pivot["richardson_number"] = df_pivot["TROPP"]/(df_pivot["wind_speed"]**2+1)
            df_pivot["turbulence_index"] = (
                df_pivot["wind_speed"]*0.4 +
                df_pivot["wind_shear"]*0.3 +
                (1/(df_pivot["richardson_number"]+0.1))*0.3
            )

        if "O3" in df_pivot:
            df_pivot["ozone_concentration"] = df_pivot["O3"]
            df_pivot["day_of_year"] = df_pivot["timestamp"].dt.dayofyear
            mean_season = df_pivot.groupby("day_of_year")["ozone_concentration"].transform("mean")
            df_pivot["ozone_anomaly"] = df_pivot["ozone_concentration"]-mean_season
            thresh = df_pivot["ozone_concentration"].quantile(0.9)
            df_pivot["high_ozone_event"] = df_pivot["ozone_concentration"]>thresh

        if "T" in df_pivot and "RH" in df_pivot:
            df_pivot["lifted_index"] = df_pivot["T"]-273.15-df_pivot["RH"]*0.5
            df_pivot["k_index"] = df_pivot["T"]-273.15+df_pivot["RH"]-df_pivot["wind_speed"]*0.1

        if "wind_speed" in df_pivot:
            mask = (df_pivot["pressure_level"]<=300)&(df_pivot["wind_speed"]>30)
            df_pivot["jet_stream_core"] = mask
            df_pivot["jet_stream_strength"] = np.where(mask,df_pivot["wind_speed"],0)

        if "T" in df_pivot and "SLP" in df_pivot:
            df_pivot = df_pivot.sort_values(["timestamp","latitude"])
            df_pivot["temp_gradient"] = df_pivot.groupby(["timestamp","longitude"])["T"].diff().abs().fillna(0)
            df_pivot["pressure_gradient"] = df_pivot.groupby(["timestamp","longitude"])["SLP"].diff().abs().fillna(0)
            df_pivot["front_strength"] = df_pivot["temp_gradient"]*df_pivot["pressure_gradient"]

        return df_pivot


    def process_merra2_data(self, df):
        """Process MERRA-2 data with enhanced features"""
        if df.empty:
            return df

        df_pivot = df.pivot_table(
            index=["timestamp","latitude","longitude","pressure_level"],
            columns="variable",
            values="value"
        ).reset_index()

        for col in df_pivot.columns:
            if col not in ["timestamp","latitude","longitude","pressure_level"]:
                df_pivot[col] = df_pivot[col].fillna(df_pivot[col].mean())

        if "U" in df_pivot and "V" in df_pivot:
            df_pivot["wind_speed"] = np.sqrt(df_pivot["U"]**2 + df_pivot["V"]**2)
            df_pivot["wind_direction"] = (np.arctan2(df_pivot["V"], df_pivot["U"]) * 180/np.pi +360)%360

        if "T" in df_pivot and "RH" in df_pivot:
            df_pivot["dew_point"] = df_pivot["T"] - ((100-df_pivot["RH"])/5)
            df_pivot["rh_calculated"] = 100*(df_pivot["QV"]/0.022)

        if "PRECTOT" in df_pivot:
            df_pivot["precipitation_rate"] = df_pivot["PRECTOT"]*3600
            df_pivot["precip_event"] = df_pivot["precipitation_rate"]>0.1
            df_pivot["precip_intensity"] = pd.cut(
                df_pivot["precipitation_rate"],
                bins=[0,0.1,2.5,10,50,np.inf],
                labels=['None','Light','Moderate','Heavy','Violent']
            )

        if "CLD" in df_pivot:
            df_pivot["cloud_cover"] = pd.cut(
                df_pivot["CLD"],
                bins=[0,0.25,0.5,0.75,1],
                labels=['Clear','Scattered','Broken','Overcast']
            )
            df_pivot["overcast"] = df_pivot["CLD"]>0.75

        if "T" in df_pivot and "OMEGA" in df_pivot:
            df_pivot["vertical_stability"] = -df_pivot["OMEGA"]*df_pivot["T"]
            thr = df_pivot["vertical_stability"].quantile([0.25, 0.75])
            edges = [-np.inf, thr[0.25], thr[0.75], np.inf]
            edges = sorted(set(edges))
            labels = ['Stable', 'Neutral', 'Unstable'][:len(edges)-1]
            df_pivot["stability_category"] = pd.cut(
                df_pivot["vertical_stability"],
                bins=edges,
                labels=labels,
                include_lowest=True
            )

        if "O3" in df_pivot:
            df_pivot["ozone_concentration"]=df_pivot["O3"]
            df_pivot["ozone_hole"]=df_pivot["ozone_concentration"]<100
            df_pivot = df_pivot.sort_values(["latitude","longitude","timestamp"])
            df_pivot["ozone_column"] = df_pivot.groupby(["latitude","longitude","timestamp"])["ozone_concentration"].transform("sum")

        if "QV" in df_pivot:
            df_pivot["precipitable_water"] = df_pivot["QV"]*100
            thr = df_pivot["precipitable_water"].quantile([0.33,0.67])
            # Handle duplicate quantiles by ensuring unique bins
            bins = [-np.inf, thr[0.33], thr[0.67], np.inf]
            unique_bins = []
            for b in bins:
                if b not in unique_bins:
                    unique_bins.append(b)
            labels = ['Dry','Moderate','Moist'][:len(unique_bins)-1]
            df_pivot["moisture_category"] = pd.cut(
                df_pivot["precipitable_water"],
                bins=unique_bins,
                labels=labels,
                include_lowest=True
            )

        return df_pivot


    def integrate_datasets(self, geos5_df, merra2_df):
        """Integrate GEOS-5 and MERRA-2 datasets"""
        if geos5_df.empty and merra2_df.empty:
            return pd.DataFrame()

        if geos5_df.empty:
            merra2_df = merra2_df.add_prefix("merra2_").rename(columns={"merra2_timestamp":"timestamp"})
            return merra2_df
        if merra2_df.empty:
            geos5_df = geos5_df.add_prefix("geos5_").rename(columns={"geos5_timestamp":"timestamp"})
            return geos5_df

        geos5_df = geos5_df.add_prefix("geos5_").rename(columns={"geos5_timestamp":"timestamp"})
        merra2_df = merra2_df.add_prefix("merra2_").rename(columns={"merra2_timestamp":"timestamp"})
        merra2_df = merra2_df.rename(columns={"merra2_latitude":"geos5_latitude","merra2_longitude":"geos5_longitude"})

        merged = pd.merge(
            geos5_df, merra2_df,
            on=["timestamp","geos5_latitude","geos5_longitude"],
            how="outer", suffixes=("_geos5","_merra2")
        )

        merged["combined_wind_speed"] = (merged.get("geos5_wind_speed",0).fillna(0) + merged.get("merra2_wind_speed",0).fillna(0))/2
        merged["combined_turbulence_index"] = merged.get("geos5_turbulence_index",0).fillna(0)*0.6 + merged.get("merra2_vertical_stability",0).fillna(0)*0.4
        merged["combined_ozone_concentration"] = (merged.get("geos5_ozone_concentration",0).fillna(0)+merged.get("merra2_ozone_concentration",0).fillna(0))/2

        merged["atmospheric_risk_score"] = (
            merged["combined_wind_speed"]*0.3 +
            merged["combined_turbulence_index"]*0.4 +
            merged["combined_ozone_concentration"]*0.3
        )
        
        # Improved handling of potential division by zero
        max_risk = merged["atmospheric_risk_score"].max()
        if max_risk > 0:
            merged["atmospheric_risk_normalized"] = merged["atmospheric_risk_score"]/max_risk*100
        else:
            merged["atmospheric_risk_normalized"] = 0

        return merged


    def run_pipeline(self, start_date=None, end_date=None, save_data=True):
        """
        Run the complete atmospheric pipeline
        """
        if start_date is None:
            start_date = (datetime.now()-timedelta(days=7)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        date_range = f"{start_date}_to_{end_date}"
        print("=== Atmospheric Pipeline Execution ===")
        print(f"Date Range: {start_date} to {end_date}")

        print("\nStep 1: Fetching atmospheric data...")
        geos5_data = self.fetch_geos5_data(['T','U','V','OMEGA','RH','QV','O3','TROPP','H','SLP'], start_date, end_date)
        merra2_data = self.fetch_merra2_data(['T','U','V','OMEGA','RH','QV','O3','TROPP','CLD','PRECTOT'], start_date, end_date)

        print("\nStep 2: Processing atmospheric data...")
        geos5_df = self.process_geos5_data(geos5_data)
        merra2_df = self.process_merra2_data(merra2_data)

        print("\nStep 3: Integrating atmospheric datasets...")
        integrated_df = self.integrate_datasets(geos5_df, merra2_df)

        if save_data:
            print("\nStep 4: Saving processed data...")
            for df, name in [(geos5_df,"geos5"),(merra2_df,"merra2"),(integrated_df,"integrated_atmospheric")]:
                path = Path("nasa_data/atmospheric"); path.mkdir(parents=True,exist_ok=True)
                df.to_csv(path/f"{name}_{date_range.replace(':','_')}.csv",index=False)
                print(f"Data saved to {path/name}_{date_range.replace(':','_')}.csv")

        print("\n=== Atmospheric Pipeline Complete ===")
        print(f"GEOS-5 Records: {len(geos5_df)}")
        print(f"MERRA-2 Records: {len(merra2_df)}")
        print(f"Integrated Records: {len(integrated_df)}")

        return {
            "geos5_data": geos5_df,
            "merra2_data": merra2_df,
            "integrated_data": integrated_df
        }


if __name__ == "__main__":
    pipeline = AtmosphericPipeline()
    pipeline.run_pipeline()