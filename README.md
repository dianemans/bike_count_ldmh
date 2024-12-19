# Kaggle Challenge: Predicting Cyclist Traffic in Paris

This repository contains all the code, data, and documentation for our participation in the Kaggle competition *Predicting Cyclist Traffic in Paris*. The objective is to forecast the number of cyclists at various locations across Paris based on temporal and external data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## Project Overview

Cyclist traffic in Paris is influenced by a variety of factors, including time of year, weather conditions, holidays, and external disruptions like the COVID-19 pandemic. Our analysis and models aim to capture these influences to accurately predict bike traffic.

### Highlights:
- *Dataset*: Hourly bike counts from 30 counters in Paris (September 2020 to September 2021).
- *Features*: Includes weather data, holidays, and COVID-19 measures.
- *Models Tested*:
  - Linear Regression (baseline)
  - Facebook Prophet (time-series)
  - XGBoost (tree-based, best performer)
- *Evaluation Metric*: Root Mean Square Error (RMSE).

---

## Repository Structure

```plaintext
.
├── data/                      # Raw and processed datasets
├── external_data/             # External data sources (e.g., weather, holidays)
├── eda.ipynb                  # Exploratory Data Analysis
├── crossval.ipynb             # Cross-validation experiments
├── optuna.ipynb               # Hyperparameter tuning
├── prophet.ipynb              # Prophet experiments
├── mainfile.py                # Main script for predictions
├── utils.py                   # Utility functions (encoding, preprocessing, pipeline)
├── models.pkl                 # Models from the Prophet experiments
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation (this file)
└── pictures/                  # Generated plots and figures