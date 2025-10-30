# Aerodrome Movement Monitoring Using ADS-B Data:  
## A Case Study at Lommis Airfield

### Presented at:
#### The 13th OpenSky Symposium

**Authors:**  
- Alex Fustagueras ([ORCID 0009-0005-7063-6295](https://orcid.org/0009-0005-7063-6295))  
  - *Centre for Aviation, Zurich University of Applied Sciences, Winterthur, Switzerland*  
  - <alex.fustagueras@zhaw.ch>  
- Manuel Waltert ([ORCID 0000-0001-7649-6581](https://orcid.org/0000-0001-7649-6581))  
  - *Centre for Aviation, Zurich University of Applied Sciences, Winterthur, Switzerland*

---

## Project Overview

This repository provides the full workflow and code for the OpenSky Symposium paper titled:

> **Aerodrome Movement Monitoring Using ADS-B Data: A Case Study at Lommis Airfield**

The methods and tools here enable detection, quantification, and reporting of aerodrome movements (departures, arrivals, and traffic circuits) at Lommis Airfield, using ADS-B surveillance data.

---

## Directory Structure

```
main.ipynb                   # Main workflow: from raw data to reporting and visualization
model/
    model_training.ipynb     # Machine learning model training
    model_evaluation.ipynb   # Quantitative evaluation of trained models
    labeling_app.py          # Streamlit tool for interactive flight labeling
data/
    comparison_analysis.ipynb# Compare outputs to reference data
    dashboard.ipynb          # Missing flights or QC dashboard
    plots.ipynb              # Additional figures for analysis/paper
lommis_func.py               # Project-specific functions and utilities
LICENSE                      # License file
```

---

## Main Workflow

- **Start with [`main.ipynb`](main.ipynb)**
  - This notebook demonstrates the entire process:
    - Loading and pre-processing ADS-B data
    - Detecting and classifying flight events (departures, arrivals, circuits)
    - Saving monthly reports in BAZL-compliant Excel format
    - Visualizing key statistics

- **Model Development & Evaluation**
  - All model-related code is under `model/`: training, evaluation, and labeling workflows.

- **Data Analysis & Visualization**
  - Summary dashboards and comparison analyses are under `data/`.