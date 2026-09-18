# AeroSpaceGuard

> An AI-assisted aerospace software project for analysing space weather, atmospheric conditions, orbital debris and flight-path optimisation.

## Overview

AeroSpaceGuard is a Python-based aerospace project that combines multiple data-processing pipelines to support aviation and aerospace analysis.

The project brings together space-weather data, atmospheric information and orbital-debris data, with additional components for turbulence prediction and flight-path optimisation.

The project was developed as part of the NASA Space Apps Challenge.

## Features

- **Space Weather Analysis** — Processes space-weather information and considers potential impacts on aviation.
- **Atmospheric Data Pipeline** — Processes atmospheric conditions relevant to flight-path analysis.
- **Orbital Debris Pipeline** — Incorporates orbital-debris information for aerospace situational awareness.
- **Turbulence Prediction** — Uses machine-learning concepts to analyse and predict turbulence conditions.
- **Flight-Path Optimisation** — Generates optimised routes using the processed data.
- **Route Visualisation** — Includes interactive HTML visualisations for selected international routes.

## System Components

The project is organised around several Python components:

```text
AeroSpaceGuard/
│
├── AeroSpaceGuard.py
├── main_pipeline.py
├── main_ai_ml_system.py
│
├── space_weather_pipeline.py
├── atmospheric_pipeline.py
├── orbital_debris_pipeline.py
├── turbulence_prediction.py
├── flight_optimization.py
│
├── optimized_route_JFK_DXB.html
├── optimized_route_LAX_LHR.html
├── optimized_route_SIN_SYD.html
│
├── Report for AeroSpaceGuard.pdf
└── README.md
