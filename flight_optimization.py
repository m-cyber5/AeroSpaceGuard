"""
Advanced Flight Path Optimization Module
Professional multi-objective optimization system for flight routing
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import warnings
from geopy.distance import geodesic
import networkx as nx
import folium
from folium import plugins

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flight_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedFlightPathOptimizer:
    """
    Simplified flight path optimization system
    """
    
    def __init__(self, turbulence_predictor, model_save_path="models/optimization"):
        """
        Initialize the flight path optimizer
        
        Args:
            turbulence_predictor: Trained turbulence predictor
            model_save_path (str): Path to save optimization models
        """
        self.turbulence_predictor = turbulence_predictor
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.graph = None
        self.waypoints = None
        self.ordem_data = None
        self.atmospheric_data = None
        
        # Optimization parameters
        self.optimization_params = {
            'safety_weight': 0.4,
            'fuel_weight': 0.3,
            'time_weight': 0.3
        }
        
        logger.info("Advanced Flight Path Optimizer initialized")
    
    def set_atmospheric_data(self, atmospheric_data):
        """
        Set atmospheric data for risk calculations
        
        Args:
            atmospheric_data (pd.DataFrame): Atmospheric data
        """
        self.atmospheric_data = atmospheric_data
        logger.info("Atmospheric data set for optimization")
    
    def set_ordem_data(self, ordem_data):
        """
        Set orbital debris data for risk calculations
        
        Args:
            ordem_data (pd.DataFrame): ORDEM debris data
        """
        self.ordem_data = ordem_data
        logger.info("ORDEM data set for optimization")
    
    def create_advanced_flight_network(self, origin, destination, grid_size=2.0, altitude_layers=3):
        """
        Create a simplified 3D flight network
        
        Args:
            origin (tuple): Origin coordinates (lat, lon)
            destination (tuple): Destination coordinates (lat, lon)
            grid_size (float): Grid resolution in degrees
            altitude_layers (int): Number of altitude layers
        """
        logger.info(f"Creating simplified flight network from {origin} to {destination}")
        
        self.waypoints = self._generate_3d_waypoints(origin, destination, grid_size, altitude_layers)
        
        self.graph = nx.Graph()
        
        # Add nodes with 3D coordinates
        for i, wp in enumerate(self.waypoints):
            self.graph.add_node(
                i, 
                pos=(wp['lat'], wp['lon']), 
                altitude=wp['alt'], 
                layer=wp['layer']
            )
        
        # Add edges with 3D distance calculations
        edge_count = 0
        for i in range(len(self.waypoints)):
            for j in range(i+1, len(self.waypoints)):
                # Calculate 3D distance
                dist_2d = geodesic(
                    (self.waypoints[i]['lat'], self.waypoints[i]['lon']),
                    (self.waypoints[j]['lat'], self.waypoints[j]['lon'])
                ).kilometers
                
                alt_diff = abs(self.waypoints[i]['alt'] - self.waypoints[j]['alt'])
                dist_3d = np.sqrt(dist_2d**2 + alt_diff**2)
                
                # Only connect waypoints within reasonable 3D distance
                if dist_3d < 1000:  # Increased maximum connection distance
                    self.graph.add_edge(
                        i, j, 
                        weight=dist_3d, 
                        distance_2d=dist_2d, 
                        altitude_change=alt_diff
                    )
                    edge_count += 1
        
        logger.info(f"Flight network created with {len(self.waypoints)} waypoints and {edge_count} edges")
    
    def _generate_3d_waypoints(self, origin, destination, grid_size, altitude_layers):
        """
        Generate 3D waypoints with multiple altitude layers
        
        Args:
            origin (tuple): Origin coordinates (lat, lon)
            destination (tuple): Destination coordinates (lat, lon)
            grid_size (float): Grid resolution in degrees
            altitude_layers (int): Number of altitude layers
            
        Returns:
            list: List of 3D waypoint dictionaries
        """
        waypoints = []
        
        # Create grid
        min_lat = min(origin[0], destination[0]) - 2
        max_lat = max(origin[0], destination[0]) + 2
        min_lon = min(origin[1], destination[1]) - 2
        max_lon = max(origin[1], destination[1]) + 2
        
        lats = np.arange(min_lat, max_lat, grid_size)
        lons = np.arange(min_lon, max_lon, grid_size)
        
        # Define altitude layers
        altitudes = np.linspace(9, 13, altitude_layers)  # 9km to 13km
        
        for layer, alt in enumerate(altitudes):
            for lat in lats:
                for lon in lons:
                    waypoints.append({
                        'lat': lat,
                        'lon': lon,
                        'alt': alt,
                        'layer': layer
                    })
        
        # Ensure origin and destination are included at all altitude layers
        for layer, alt in enumerate(altitudes):
            waypoints.insert(0, {'lat': origin[0], 'lon': origin[1], 'alt': alt, 'layer': layer})
            waypoints.append({'lat': destination[0], 'lon': destination[1], 'alt': alt, 'layer': layer})
        
        logger.info(f"Generated {len(waypoints)} 3D waypoints with {altitude_layers} altitude layers")
        
        return waypoints
    
    def calculate_comprehensive_risk(self, position, altitude, time, path_segment=None):
        """
        Calculate simplified comprehensive risk
        
        Args:
            position (tuple): Position (lat, lon)
            altitude (float): Altitude in km
            time (datetime): Time of flight
            path_segment (dict): Path segment information
            
        Returns:
            dict: Comprehensive risk assessment
        """
        risk_assessment = {
            'atmospheric_risk': 0,
            'debris_risk': 0,
            'total_risk': 0,
            'risk_factors': {}
        }
        
        # Simplified atmospheric risk calculation
        if self.atmospheric_data is not None:
            # Find nearest atmospheric data point
            min_dist = float('inf')
            nearest_data = None
            
            for _, row in self.atmospheric_data.iterrows():
                if 'geos5_latitude' in row and 'geos5_longitude' in row:
                    dist = geodesic(
                        position,
                        (row['geos5_latitude'], row['geos5_longitude'])
                    ).kilometers
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_data = row
            
            if nearest_data is not None:
                # Extract risk factors
                turbulence_index = nearest_data.get('combined_turbulence_index', 0)
                wind_speed = nearest_data.get('combined_wind_speed', 0)
                ozone_concentration = nearest_data.get('combined_ozone_concentration', 0)
                
                # Calculate risks
                risk_assessment['atmospheric_risk'] = min(turbulence_index, 1.0)
                risk_assessment['risk_factors']['turbulence_risk'] = min(turbulence_index, 1.0)
                risk_assessment['risk_factors']['wind_risk'] = min(wind_speed / 50, 1.0)
                risk_assessment['risk_factors']['ozone_risk'] = min(ozone_concentration / 300, 1.0)
        
        # Simplified debris risk calculation
        if self.ordem_data is not None:
            # Random debris risk for demonstration
            risk_assessment['debris_risk'] = np.random.uniform(0, 0.5)
            risk_assessment['risk_factors']['debris_risk'] = risk_assessment['debris_risk']
        
        # Total risk (weighted sum)
        risk_assessment['total_risk'] = (
            risk_assessment['atmospheric_risk'] * self.optimization_params['safety_weight'] +
            risk_assessment['debris_risk'] * 0.2
        )
        
        return risk_assessment
    
    def calculate_advanced_edge_cost(self, u, v, data, risk_data, flight_time):
        """
        Calculate simplified edge cost
        
        Args:
            u (int): Source node index
            v (int): Target node index
            data (dict): Edge data
            risk_data (pd.DataFrame): Risk data
            flight_time (datetime): Flight time
            
        Returns:
            dict: Comprehensive edge cost analysis
        """
        # Get positions and altitude
        u_pos = self.graph.nodes[u]['pos']
        v_pos = self.graph.nodes[v]['pos']
        altitude = self.graph.nodes[u]['altitude']
        
        # Calculate comprehensive risk
        u_risk = self.calculate_comprehensive_risk(u_pos, altitude, flight_time)
        v_risk = self.calculate_comprehensive_risk(v_pos, altitude, flight_time)
        
        # Use maximum risk along the segment
        max_risk = max(u_risk['total_risk'], v_risk['total_risk'])
        
        # Calculate fuel consumption (simplified model)
        base_fuel = data['weight'] * 1.2  # Base fuel consumption
        altitude_fuel = data['altitude_change'] * 0.1  # Fuel for altitude changes
        total_fuel = base_fuel + altitude_fuel
        
        # Calculate time
        time_factor = data['weight'] / 900  # Assuming average speed 900 km/h
        
        # Calculate total cost with weights
        total_cost = (
            self.optimization_params['safety_weight'] * max_risk +
            self.optimization_params['fuel_weight'] * total_fuel +
            self.optimization_params['time_weight'] * time_factor
        )
        
        return {
            'total_cost': total_cost,
            'risk_assessment': {
                'u': u_risk,
                'v': v_risk,
                'max_risk': max_risk
            },
            'fuel_consumption': total_fuel,
            'time_factor': time_factor,
            'distance': data['weight'],
            'altitude_change': data['altitude_change']
        }
    
    def optimize_path_advanced_astar(self, origin_idx, destination_idx, risk_data, flight_time):
        """
        Find optimal path using A* algorithm
        
        Args:
            origin_idx (int): Origin node index
            destination_idx (int): Destination node index
            risk_data (pd.DataFrame): Risk data
            flight_time (datetime): Flight time
            
        Returns:
            dict: Optimal path with detailed analysis
        """
        logger.info("Running A* optimization algorithm")
        
        def heuristic(u, v):
            # Use 3D distance as heuristic
            u_pos = self.graph.nodes[u]['pos']
            v_pos = self.graph.nodes[v]['pos']
            u_alt = self.graph.nodes[u]['altitude']
            v_alt = self.graph.nodes[v]['altitude']
            
            dist_2d = geodesic(u_pos, v_pos).kilometers
            alt_diff = abs(u_alt - v_alt)
            return np.sqrt(dist_2d**2 + alt_diff**2)
        
        # Update edge weights with comprehensive cost analysis
        edge_costs = {}
        for u, v, data in self.graph.edges(data=True):
            cost_analysis = self.calculate_advanced_edge_cost(u, v, data, risk_data, flight_time)
            edge_costs[(u, v)] = cost_analysis
            data['cost'] = cost_analysis['total_cost']
        
        # Find shortest path using A*
        path = nx.astar_path(
            self.graph, 
            source=origin_idx, 
            target=destination_idx,
            heuristic=heuristic,
            weight='cost'
        )
        
        # Analyze the path
        path_analysis = self._analyze_path(path, edge_costs, flight_time)
        
        logger.info(f"A* optimization completed. Path length: {len(path)} waypoints")
        
        return {
            'path': path,
            'analysis': path_analysis,
            'edge_costs': edge_costs
        }
    
    def _analyze_path(self, path, edge_costs, flight_time):
        """
        Analyze the optimized path with detailed metrics
        
        Args:
            path (list): Path as list of node indices
            edge_costs (dict): Edge cost analysis
            flight_time (datetime): Flight time
            
        Returns:
            dict: Path analysis
        """
        total_distance = 0
        total_fuel = 0
        total_time = 0
        risk_profile = []
        
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            
            if (u, v) in edge_costs:
                cost_info = edge_costs[(u, v)]
                total_distance += cost_info['distance']
                total_fuel += cost_info['fuel_consumption']
                total_time += cost_info['time_factor']
                
                risk_profile.append({
                    'segment': f"{u}-{v}",
                    'max_risk': cost_info['risk_assessment']['max_risk'],
                    'atmospheric_risk': cost_info['risk_assessment']['u']['atmospheric_risk'],
                    'debris_risk': cost_info['risk_assessment']['u']['debris_risk']
                })
        
        # Calculate overall metrics
        avg_risk = np.mean([rp['max_risk'] for rp in risk_profile]) if risk_profile else 0
        avg_comfort = 1 - avg_risk  # Simplified comfort calculation
        max_risk_segment = max(risk_profile, key=lambda x: x['max_risk']) if risk_profile else None
        
        return {
            'total_distance': total_distance,
            'total_fuel': total_fuel,
            'total_time': total_time,
            'average_risk': avg_risk,
            'average_comfort': avg_comfort,
            'max_risk_segment': max_risk_segment,
            'risk_profile': risk_profile,
            'num_segments': len(path) - 1
        }
    
    def compare_algorithms(self, origin, destination, risk_data=None, flight_time=None):
        """
        Compare optimization algorithms (simplified)
        
        Args:
            origin (str): Origin airport code
            destination (str): Destination airport code
            risk_data (pd.DataFrame): Risk data
            flight_time (datetime): Flight time
            
        Returns:
            dict: Comparison results
        """
        logger.info(f"Comparing optimization algorithms for {origin} to {destination}")
        
        airports = {
            'JFK': (40.6413, -73.7781),
            'DXB': (25.2532, 55.3657),
            'LAX': (34.0522, -118.2437),
            'LHR': (51.4700, -0.4543),
            'SIN': (1.3592, 103.9898),
            'SYD': (-33.8688, 151.2093)
        }
        
        origin_coords = airports.get(origin, origin)
        dest_coords = airports.get(destination, destination)
        
        if flight_time is None:
            flight_time = datetime.now()
        
        # Create flight network
        self.create_advanced_flight_network(origin_coords, dest_coords, grid_size=2.0, altitude_layers=3)
        
        # Find origin and destination indices (use middle altitude layer)
        origin_idx = 1  # Middle layer
        destination_idx = len(self.waypoints) - 2  # Middle layer at destination
        
        # Run A* algorithm
        result = self.optimize_path_advanced_astar(
            origin_idx, destination_idx, risk_data, flight_time
        )
        
        # Calculate direct path for comparison
        direct_distance = geodesic(origin_coords, dest_coords).kilometers
        
        # Generate results
        results = {
            'advanced_astar': {
                'origin': origin,
                'destination': destination,
                'algorithm': 'advanced_astar',
                'direct_path': {
                    'distance_km': direct_distance,
                    'estimated_time': direct_distance / 900  # hours
                },
                'optimized_path': {
                    'distance_km': result['analysis']['total_distance'],
                    'estimated_time': result['analysis']['total_time'],
                    'fuel_consumption': result['analysis']['total_fuel'],
                    'num_waypoints': len(result['path']),
                    'num_segments': result['analysis']['num_segments']
                },
                'performance_metrics': {
                    'distance_increase': ((result['analysis']['total_distance'] - direct_distance) / direct_distance) * 100,
                    'average_risk': result['analysis']['average_risk'],
                    'average_comfort': result['analysis']['average_comfort'],
                    'max_risk_segment': result['analysis']['max_risk_segment']
                },
                'path_analysis': result['analysis'],
                'risk_profile': result['analysis']['risk_profile'],
                'path': result['path']
            }
        }
        
        comparison_report = {
            'origin': origin,
            'destination': destination,
            'algorithms_tested': ['advanced_astar'],
            'results': results,
            'best_algorithm': 'advanced_astar',
            'comparison_metrics': {
                'distance_increase': {'advanced_astar': results['advanced_astar']['performance_metrics']['distance_increase']},
                'average_risk': {'advanced_astar': results['advanced_astar']['performance_metrics']['average_risk']},
                'average_comfort': {'advanced_astar': results['advanced_astar']['performance_metrics']['average_comfort']}
            }
        }
        
        logger.info(f"Algorithm comparison completed. Best algorithm: advanced_astar")
        
        return comparison_report
    
    def visualize_route(self, optimization_result, origin, destination):
        """
        Create simplified visualization of the optimized route
        
        Args:
            optimization_result (dict): Route optimization result
            origin (str): Origin airport code
            destination (str): Destination airport code
            
        Returns:
            folium.Map: Interactive map with visualization
        """
        airports = {
            'JFK': (40.6413, -73.7781),
            'DXB': (25.2532, 55.3657),
            'LAX': (34.0522, -118.2437),
            'LHR': (51.4700, -0.4543),
            'SIN': (1.3592, 103.9898),
            'SYD': (-33.8688, 151.2093)
        }
        
        origin_coords = airports.get(origin, origin)
        dest_coords = airports.get(destination, destination)
        
        # Calculate center point
        center_lat = (origin_coords[0] + dest_coords[0]) / 2
        center_lon = (origin_coords[1] + dest_coords[1]) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=3)
        
        # Add direct path
        direct_coords = [
            [origin_coords[0], origin_coords[1]],
            [dest_coords[0], dest_coords[1]]
        ]
        folium.PolyLine(
            direct_coords,
            color='red',
            weight=3,
            opacity=0.7,
            tooltip='Direct Path'
        ).add_to(m)
        
        # Add optimized path
        path = optimization_result.get('path', [])  # Use .get() with default empty list
        if path:  # Check if path exists and is not empty
            optimized_coords = []
            for idx in path:
                wp = self.waypoints[idx]
                optimized_coords.append([wp['lat'], wp['lon']])
            
            folium.PolyLine(
                optimized_coords,
                color='green',
                weight=3,
                opacity=0.7,
                tooltip='Optimized Path'
            ).add_to(m)
        else:
            logger.warning("No path found in optimization result")
        
        # Add markers for origin and destination
        folium.Marker(
            origin_coords,
            popup=f'<b>Origin: {origin}</b>',
            icon=folium.Icon(color='blue', icon='plane', prefix='fa')
        ).add_to(m)
        
        folium.Marker(
            dest_coords,
            popup=f'<b>Destination: {destination}</b>',
            icon=folium.Icon(color='green', icon='plane', prefix='fa')
        ).add_to(m)
        
        # Save map
        map_file = f"optimized_route_{origin}_{destination}.html"
        m.save(map_file)
        logger.info(f"Route visualization saved to {map_file}")
        print(f"📍 Map saved to: {map_file}")
        
        return m
    
    def _calculate_risk_distribution(self, risk_profile):
        """
        Calculate risk distribution across path segments
        
        Args:
            risk_profile (list): Risk profile data
            
        Returns:
            dict: Risk distribution
        """
        risk_levels = {'low': 0, 'medium': 0, 'high': 0}
        
        for segment in risk_profile:
            risk = segment['max_risk']
            if risk <= 0.3:
                risk_levels['low'] += 1
            elif risk <= 0.7:
                risk_levels['medium'] += 1
            else:
                risk_levels['high'] += 1
        
        total = sum(risk_levels.values())
        if total > 0:
            return {level: count/total for level, count in risk_levels.items()}
        else:
            return {'low': 0, 'medium': 0, 'high': 0}
    
    def _calculate_fuel_efficiency_rating(self, optimization_result):
        """
        Calculate fuel efficiency rating
        
        Args:
            optimization_result (dict): Optimization result
            
        Returns:
            str: Fuel efficiency rating
        """
        distance_increase = optimization_result['performance_metrics']['distance_increase']
        fuel_consumption = optimization_result['optimized_path']['fuel_consumption']
        
        if distance_increase < 5 and fuel_consumption < 1000:
            return "Excellent"
        elif distance_increase < 10 and fuel_consumption < 1500:
            return "Good"
        elif distance_increase < 15 and fuel_consumption < 2000:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_optimization_recommendations(self, optimization_result):
        """
        Generate recommendations based on optimization results
        
        Args:
            optimization_result (dict): Optimization result
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # Safety recommendations
        avg_risk = optimization_result['performance_metrics']['average_risk']
        if avg_risk > 0.7:
            recommendations.append({
                "priority": "HIGH",
                "category": "Safety",
                "message": "High average risk detected - consider alternative routing"
            })
        elif avg_risk > 0.4:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Safety",
                "message": "Moderate risk levels - monitor conditions during flight"
            })
        
        # Fuel efficiency recommendations
        distance_increase = optimization_result['performance_metrics']['distance_increase']
        if distance_increase > 15:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Fuel Efficiency",
                "message": "Significant distance increase - evaluate fuel impact"
            })
        
        return recommendations