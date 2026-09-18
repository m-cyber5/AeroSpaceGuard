# AeroSpaceGuard

> AI-assisted aerospace software for space-weather, atmospheric and orbital-debris analysis with flight-path optimisation and route visualisation.

## Overview

AeroSpaceGuard is a Python-based aerospace software project that integrates multiple data-processing pipelines with machine-learning and flight-path optimisation components.

The project combines space-weather, atmospheric and orbital-debris data to create an integrated dataset and risk-analysis workflow. It also includes a Random Forest turbulence classification model and an A* flight-path optimisation system operating on a simplified 3D flight network.

The project was developed as part of the NASA Space Apps Challenge.

---

## Key Features

### 🌌 Space Weather Analysis

Processes space-weather information from multiple data sources to support aerospace and aviation analysis.

Supported sources include:

- DSCOVR RTSW
- ACE

---

### 🌍 Atmospheric Data Processing

Processes atmospheric datasets used as part of the project's flight and risk-analysis workflow.

Supported sources include:

- GEOS-5 FP
- MERRA-2

---

### 🛰️ Orbital Debris Analysis

Integrates orbital-debris data into the project's aerospace situational-awareness and risk-analysis pipeline.

Supported source:

- ORDEM 3.0

---

### 🤖 Turbulence Machine Learning

Includes a machine-learning pipeline for turbulence classification using a scikit-learn Random Forest model.

The model uses engineered features derived from:

- Wind speed
- Turbulence-related measurements
- Ozone
- Temperature
- Humidity
- Time-based features

The pipeline also includes:

- Feature engineering
- Feature selection
- Data scaling
- Train/test splitting
- Model evaluation
- Feature importance analysis
- Model saving and loading

---

### ✈️ Flight-Path Optimisation

Uses A* pathfinding over a simplified 3D flight network.

The optimisation system:

- Generates 3D flight waypoints
- Uses multiple altitude layers
- Builds a NetworkX graph
- Calculates route distances using geodesic calculations
- Incorporates risk, fuel and travel-time factors into route costs
- Produces route metrics for analysis

The current implementation uses three altitude layers across approximately 9–13 km.

---

### 🗺️ Route Visualisation

Generates interactive HTML maps using Folium.

Example routes included in the repository:

- JFK → DXB
- LAX → LHR
- SIN → SYD

The visualisations provide a way to inspect direct and optimised routes geographically.

---

### 📊 Integrated NASA Data Pipeline

The main data pipeline coordinates the individual data-processing components and combines their outputs into an integrated dataset.

The pipeline:

1. Executes the space-weather pipeline
2. Executes the atmospheric pipeline
3. Executes the orbital-debris pipeline
4. Integrates the resulting datasets
5. Creates a master timestamp timeline
6. Calculates a combined risk score
7. Generates data statistics
8. Performs data-quality analysis
9. Generates a comprehensive JSON report
10. Saves integrated data as CSV

---

## System Architecture

```text
                         AeroSpaceGuard
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
          NASA Data Pipeline              AI / Optimisation
                 │                             │
       ┌─────────┼─────────┐          ┌────────┴────────┐
       │         │         │          │                 │
       ▼         ▼         ▼          ▼                 ▼
   Space      Atmospheric Orbital  Turbulence      Flight Path
   Weather       Data      Debris   Prediction     Optimisation
       │         │         │          │                 │
       │         │         │      Random Forest         │
       │         │         │          │              A* / NetworkX
       └─────────┼─────────┘          │                 │
                 ▼                    └────────┬────────┘
          Data Integration                     │
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                         Route Analysis
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
               Route Metrics          Visualisation
                    │                       │
                    ▼                       ▼
               Risk / Fuel /          Interactive
                Time Analysis          HTML Maps
