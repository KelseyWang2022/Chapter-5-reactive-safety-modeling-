Crash Injury Severity Modeling (Chapter 5)

This repository contains the code for the modeling framework used in the study:

“A Hybrid Approach to Investigating Factors Associated with Crash Injury Severity: Integrating Interpretable Machine Learning with Logit Model.” 

applsci-15-10417-v2 (3)

The project combines machine learning and statistical modeling to analyze factors associated with traffic crash injury severity.

Methods

The framework includes three main components:

Random Forest (RF) for crash severity prediction

SHAP (Shapley Additive Explanations) for model interpretability

Logit-based models (MNLogit / Nested Logit / PPO) for statistical inference

The goal is to identify key factors influencing crash injury severity while maintaining model interpretability.

Repository Structure
.
├── 1.nest_logit(all).py
├── 2. nested_logit(minor+no_injury).py
├── MNLogit.py
├── PPO.ipynb
└── random forest + SHAP.py

Description

random forest + SHAP.py – Random Forest model and SHAP interpretation

MNLogit.py – Multinomial logit model

1.nest_logit(all).py – Nested logit model with full categories

2. nested_logit(minor+no_injury).py – Nested logit model with merged categories

PPO.ipynb – Partial proportional odds model

Dataset

The dataset is obtained from the French national road safety open data platform (ONISR).

Source:
https://www.onisr.securite-routiere.gouv.fr/outils-statistiques/open-data

The data include traffic crash records in France (2019–2024), covering crash characteristics, road conditions, vehicles, and road users. 
