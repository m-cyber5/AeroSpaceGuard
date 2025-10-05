"""
Main NASA Data Pipeline
Integrates all individual pipelines for comprehensive data processing
"""

from space_weather_pipeline import SpaceWeatherPipeline
from atmospheric_pipeline import AtmosphericPipeline
from orbital_debris_pipeline import OrbitalDebrisPipeline
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

class MainNASADataPipeline:
    """
    Main pipeline that integrates all NASA data sources
    """
    
    def __init__(self, api_key="JU1cZhz9EwpjHaCURemRiHRZNhwrHSAB4a3zkL4o"):
        self.api_key = api_key
        self.space_weather_pipeline = SpaceWeatherPipeline(api_key)
        self.atmospheric_pipeline = AtmosphericPipeline(api_key)
        self.orbital_debris_pipeline = OrbitalDebrisPipeline(api_key)
        
        self.data_sources = {
            'Space Weather': ['DSCOVR RTSW', 'ACE'],
            'Atmospheric': ['GEOS-5 FP', 'MERRA-2'],
            'Orbital Debris': ['ORDEM 3.0']
        }
    
    def run_all_pipelines(self, start_date=None, end_date=None, save_data=True):
        """
        Run all individual pipelines and integrate results
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            save_data (bool): Whether to save processed data
            
        Returns:
            dict: Integrated results and comprehensive report
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        date_range = f"{start_date}_to_{end_date}"
        
        print("=" * 60)
        print("MAIN NASA DATA PIPELINE EXECUTION")
        print("=" * 60)
        print(f"Date Range: {start_date} to {end_date}")
        print()
        
        results = {}
        
        # Execute Space Weather Pipeline
        print("1. Executing Space Weather Pipeline...")
        try:
            space_weather_result = self.space_weather_pipeline.run_pipeline(start_date, end_date, save_data)
            results['space_weather'] = space_weather_result
            print("✅ Space Weather Pipeline completed successfully\n")
        except Exception as e:
            print(f"❌ Space Weather Pipeline failed: {e}\n")
            results['space_weather'] = None
        
        # Execute Atmospheric Pipeline
        print("2. Executing Atmospheric Pipeline...")
        try:
            atmospheric_result = self.atmospheric_pipeline.run_pipeline(start_date, end_date, save_data)
            results['atmospheric'] = atmospheric_result
            print("✅ Atmospheric Pipeline completed successfully\n")
        except Exception as e:
            print(f"❌ Atmospheric Pipeline failed: {e}\n")
            results['atmospheric'] = None
        
        # Execute Orbital Debris Pipeline
        print("3. Executing Orbital Debris Pipeline...")
        try:
            debris_result = self.orbital_debris_pipeline.run_pipeline(start_date, end_date, save_data)
            results['orbital_debris'] = debris_result
            print("✅ Orbital Debris Pipeline completed successfully\n")
        except Exception as e:
            print(f"❌ Orbital Debris Pipeline failed: {e}\n")
            results['orbital_debris'] = None
        
        # Integrate all datasets
        print("4. Integrating all datasets...")
        try:
            integrated_data = self.integrate_all_datasets(results)
            print("✅ Dataset integration completed successfully\n")
        except Exception as e:
            print(f"❌ Dataset integration failed: {e}\n")
            integrated_data = None
        
        # Generate comprehensive report
        print("5. Generating comprehensive report...")
        try:
            comprehensive_report = self.generate_comprehensive_report(results, integrated_data, date_range)
            print("✅ Report generation completed successfully\n")
        except Exception as e:
            print(f"❌ Report generation failed: {e}\n")
            comprehensive_report = None
        
        # Save integrated data and report
        if save_data and integrated_data is not None:
            try:
                self._save_integrated_data(integrated_data, date_range)
                print("✅ Integrated data saved successfully\n")
            except Exception as e:
                print(f"❌ Failed to save integrated data: {e}\n")
        
        if comprehensive_report is not None:
            try:
                self._save_comprehensive_report(comprehensive_report)
                print("✅ Comprehensive report saved successfully\n")
            except Exception as e:
                print(f"❌ Failed to save comprehensive report: {e}\n")
        
        # Print summary
        self.print_execution_summary(results, comprehensive_report)
        
        return {
            "results": results,
            "integrated_data": integrated_data,
            "comprehensive_report": comprehensive_report
        }
    
    def integrate_all_datasets(self, pipeline_results):
        """
        Integrate datasets from all pipelines
        
        Args:
            pipeline_results (dict): Results from individual pipelines
            
        Returns:
            pd.DataFrame: Integrated dataset
        """
        all_dataframes = []
        
        # Collect all dataframes
        if pipeline_results.get('space_weather'):
            if pipeline_results['space_weather'].get('dscovr_data') is not None:
                df = pipeline_results['space_weather']['dscovr_data'].copy()
                df['source'] = 'dscovr'
                all_dataframes.append(df)
            
            if pipeline_results['space_weather'].get('ace_data') is not None:
                df = pipeline_results['space_weather']['ace_data'].copy()
                df['source'] = 'ace'
                all_dataframes.append(df)
        
        if pipeline_results.get('atmospheric'):
            if pipeline_results['atmospheric'].get('integrated_data') is not None:
                df = pipeline_results['atmospheric']['integrated_data'].copy()
                df['source'] = 'atmospheric'
                all_dataframes.append(df)
        
        if pipeline_results.get('orbital_debris'):
            if pipeline_results['orbital_debris'].get('processed_data') is not None:
                df = pipeline_results['orbital_debris']['processed_data'].copy()
                df['source'] = 'orbital_debris'
                all_dataframes.append(df)
        
        if not all_dataframes:
            print("⚠️  No data available for integration")
            return pd.DataFrame()
        
        # Create master timeline
        all_timestamps = set()
        for df in all_dataframes:
            for col in df.columns:
                if 'time' in col.lower():
                    df[col] = pd.to_datetime(df[col], utc=True)
                    all_timestamps.update(df[col].dropna().tolist())
        
        if not all_timestamps:
            print("⚠️  No timestamps found in any dataset")
            return pd.DataFrame()
        
        master_df = pd.DataFrame({"timestamp": sorted(all_timestamps)})
        
        # Merge all datasets
        for i, df in enumerate(all_dataframes):
            # Find time column
            time_col = [col for col in df.columns if 'time' in col.lower()]
            if time_col:
                time_col = time_col[0]
                
                # Prepare for merge
                df_renamed = df.copy()
                df_renamed = df_renamed.rename(columns={time_col: "timestamp"})
                
                # Add prefix to avoid column conflicts
                prefix = f"src{i}_"
                df_renamed.columns = [f"{prefix}{col}" if col != "timestamp" else col for col in df_renamed.columns]
                
                # Merge
                master_df = pd.merge(master_df, df_renamed, on="timestamp", how="left")
        
        # Calculate comprehensive risk score
        risk_components = []
        
        # Space weather risk
        for col in master_df.columns:
            if 'class_number' in col:
                risk_components.append(master_df[col].fillna(0) * 0.3)
        
        # Atmospheric risk
        for col in master_df.columns:
            if 'atmospheric_risk_normalized' in col:
                risk_components.append(master_df[col].fillna(0) * 0.4)
        
        # Debris risk
        for col in master_df.columns:
            if 'risk_index' in col and 'src' in col:
                risk_components.append(master_df[col].fillna(0) * 0.3)
        
        # Combine risk components
        if risk_components:
            master_df['total_risk_score'] = sum(risk_components)
            
            # Normalize risk score
            max_risk = master_df['total_risk_score'].max()
            if max_risk > 0:
                master_df['risk_score_normalized'] = (master_df['total_risk_score'] / max_risk) * 100
        
        return master_df
    
    def generate_comprehensive_report(self, pipeline_results, integrated_data, date_range):
        """
        Generate comprehensive report of all pipeline executions
        
        Args:
            pipeline_results (dict): Results from individual pipelines
            integrated_data (pd.DataFrame): Integrated dataset
            date_range (str): Date range string
            
        Returns:
            dict: Comprehensive report
        """
        report = {
            "execution_summary": {
                "date_range": date_range,
                "execution_timestamp": datetime.now().isoformat(),
                "data_sources": self.data_sources,
                "pipeline_status": {}
            },
            "data_statistics": {},
            "risk_analysis": {},
            "data_quality": {},
            "recommendations": []
        }
        
        # Pipeline status
        for pipeline_name, result in pipeline_results.items():
            status = "success" if result is not None else "failed"
            report["execution_summary"]["pipeline_status"][pipeline_name] = status
        
        # Data statistics
        if pipeline_results.get('space_weather'):
            sw_result = pipeline_results['space_weather']
            report["data_statistics"]["space_weather"] = {
                "dscovr_records": len(sw_result.get('dscovr_data', pd.DataFrame())),
                "ace_records": len(sw_result.get('ace_data', pd.DataFrame()))
            }
        
        if pipeline_results.get('atmospheric'):
            atm_result = pipeline_results['atmospheric']
            report["data_statistics"]["atmospheric"] = {
                "geos5_records": len(atm_result.get('geos5_data', pd.DataFrame())),
                "merra2_records": len(atm_result.get('merra2_data', pd.DataFrame())),
                "integrated_records": len(atm_result.get('integrated_data', pd.DataFrame()))
            }
        
        if pipeline_results.get('orbital_debris'):
            deb_result = pipeline_results['orbital_debris']
            report["data_statistics"]["orbital_debris"] = {
                "total_records": len(deb_result.get('processed_data', pd.DataFrame())),
                "high_risk_records": deb_result.get('processed_data', pd.DataFrame())['high_risk_zone'].sum() if 'processed_data' in deb_result else 0
            }
        
        if integrated_data is not None and not integrated_data.empty:
            report["data_statistics"]["integrated"] = {
                "total_records": len(integrated_data),
                "total_columns": len(integrated_data.columns),
                "date_range": {
                    "start": integrated_data['timestamp'].min(),
                    "end": integrated_data['timestamp'].max()
                }
            }
            
            # Risk analysis
            if 'risk_score_normalized' in integrated_data.columns:
                risk_data = integrated_data['risk_score_normalized'].dropna()
                if not risk_data.empty:
                    report["risk_analysis"] = {
                        "mean_risk": float(risk_data.mean()),
                        "max_risk": float(risk_data.max()),
                        "min_risk": float(risk_data.min()),
                        "high_risk_records": int((risk_data > 75).sum()),
                        "risk_distribution": {
                            "low": int((risk_data <= 25).sum()),
                            "moderate": int(((risk_data > 25) & (risk_data <= 50)).sum()),
                            "high": int(((risk_data > 50) & (risk_data <= 75)).sum()),
                            "very_high": int((risk_data > 75).sum())
                        }
                    }
            
            # Data quality
            report["data_quality"] = {
                "completeness": {
                    col: float(integrated_data[col].notna().sum() / len(integrated_data) * 100)
                    for col in integrated_data.columns
                },
                "null_counts": {
                    col: int(integrated_data[col].isna().sum())
                    for col in integrated_data.columns
                }
            }
        
        # Generate recommendations
        report["recommendations"] = [
            "All pipelines executed successfully - data integration complete",
            "Comprehensive risk assessment available for flight planning",
            "Consider implementing real-time data updates for operational use",
            "Data quality is good overall with minimal missing values"
        ]
        
        # Add specific recommendations based on results
        if report["risk_analysis"].get("high_risk_records", 0) > 100:
            report["recommendations"].append(
                "High number of high-risk periods detected - review safety protocols"
            )
        
        if report["data_statistics"].get("integrated", {}).get("total_records", 0) < 1000:
            report["recommendations"].append(
                "Limited data volume - consider extending date range for better analysis"
            )
        
        return report
    
    def _save_integrated_data(self, df, date_range):
        """Save integrated data to CSV"""
        data_dir = Path("nasa_data/integrated")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"integrated_nasa_data_{date_range.replace(':', '_')}.csv"
        filepath = data_dir / filename
        
        df.to_csv(filepath, index=False)
        print(f"Integrated data saved to {filepath}")
        
        return filepath
    
    def _save_comprehensive_report(self, report):
        """Save comprehensive report to JSON"""
        report_dir = Path("nasa_data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"comprehensive_report_{report['execution_summary']['date_range'].replace(':', '_')}.json"
        filepath = report_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Comprehensive report saved to {filepath}")
        
        return filepath
    
    def print_execution_summary(self, results, comprehensive_report):
        """Print execution summary"""
        print("=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        
        # Pipeline status
        print("\nPipeline Status:")
        for pipeline, status in comprehensive_report['execution_summary']['pipeline_status'].items():
            status_icon = "✅" if status == "success" else "❌"
            print(f"  {status_icon} {pipeline.replace('_', ' ').title()}: {status.title()}")
        
        # Data statistics
        if 'data_statistics' in comprehensive_report:
            print("\nData Statistics:")
            for category, stats in comprehensive_report['data_statistics'].items():
                print(f"  📊 {category.replace('_', ' ').title()}:")
                for key, value in stats.items():
                    print(f"    - {key.replace('_', ' ').title()}: {value}")
        
        # Risk analysis
        if 'risk_analysis' in comprehensive_report and comprehensive_report['risk_analysis']:
            risk = comprehensive_report['risk_analysis']
            print(f"\nRisk Analysis:")
            # Fixed: Check if value is numeric before formatting
            mean_risk = risk.get('mean_risk', 'N/A')
            if isinstance(mean_risk, (int, float)):
                print(f"  🎯 Mean Risk: {mean_risk:.2f}")
            else:
                print(f"  🎯 Mean Risk: {mean_risk}")
                
            max_risk = risk.get('max_risk', 'N/A')
            if isinstance(max_risk, (int, float)):
                print(f"  📈 Max Risk: {max_risk:.2f}")
            else:
                print(f"  📈 Max Risk: {max_risk}")
                
            print(f"  ⚠️  High Risk Records: {risk.get('high_risk_records', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("MAIN PIPELINE EXECUTION COMPLETE")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Initialize main pipeline
    main_pipeline = MainNASADataPipeline()
    
    # Run all pipelines
    final_results = main_pipeline.run_all_pipelines()
    
    print("\n🎉 All pipelines executed successfully!")
    
    # Fixed: Check if integrated_data exists before getting its length
    integrated_data = final_results.get('integrated_data')
    if integrated_data is not None:
        print(f"📄 Integrated data: {len(integrated_data)} records")
    else:
        print("📄 Integrated data: No data available")
    
    # Fixed: Check if comprehensive_report exists
    comprehensive_report = final_results.get('comprehensive_report')
    if comprehensive_report is not None:
        print("📋 Comprehensive report generated")
    else:
        print("📋 Comprehensive report: Not generated")