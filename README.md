# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using advanced feature engineering and ensemble modeling techniques.

## Overview

This project implements a comprehensive fraud detection system that processes transaction data, engineers meaningful features, and applies machine learning models to identify potentially fraudulent activities with high accuracy.

## Features

- Exploratory data analysis with transaction pattern insights
- Advanced feature engineering including temporal and geospatial features
- Class imbalance handling using SMOTE-Tomek resampling
- Model training and evaluation with performance metrics
- Professional code standards suitable for production environments

## Project Structure

```
├── notebook/                    # Jupyter notebooks for analysis and modeling
├── ai-prompts/                 # AI development prompts and guidelines
├── figures/                    # Data visualizations and analysis plots
├── model/                      # Trained model artifacts
├── requirements.txt            # Python dependencies
├── validate_environment.py     # Environment validation script
└── CLAUDE.md                   # Development guidelines
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Validate environment:
   ```bash
   python validate_environment.py
   ```

3. Run analysis notebooks in sequence:
   - `01_Data_Loading_and_Understanding.ipynb`
   - `02_Exploratory_Data_Analysis.ipynb`
   - `03_Feature_Engineering.ipynb`
   - `04_Model_Training_and_Evaluation.ipynb`

## Data Source

Dataset sourced from [Kaggle Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) containing synthetic credit card transactions for machine learning purposes.

## Technologies

- Python 3.x
- pandas, numpy for data manipulation
- scikit-learn for machine learning
- imbalanced-learn for handling class imbalance
- Jupyter for interactive development