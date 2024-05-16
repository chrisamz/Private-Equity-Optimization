pip install sec-edgar-downloader

from sec_edgar_downloader import Downloader

# Initialize the downloader
dl = Downloader("path/to/save/files")

# Download filings for a specific company (e.g., Apple Inc., CIK: 0000320193)
dl.get("10-K", "0000320193", amount=5)  # Get the last 5 annual reports (10-K)

import re

def extract_financial_metrics(filing_path):
    with open(filing_path, 'r') as file:
        content = file.read()
    
    # Example regex to extract financial metrics (you may need to adjust these based on the filing format)
    revenue = re.findall(r"Total Revenue\s*\$\s*([\d,]+\.?\d*)", content)
    profit = re.findall(r"Net Income\s*\$\s*([\d,]+\.?\d*)", content)
    assets = re.findall(r"Total Assets\s*\$\s*([\d,]+\.?\d*)", content)
    liabilities = re.findall(r"Total Liabilities\s*\$\s*([\d,]+\.?\d*)", content)
    
    return {
        "revenue": revenue[0] if revenue else None,
        "profit": profit[0] if profit else None,
        "assets": assets[0] if assets else None,
        "liabilities": liabilities[0] if liabilities else None
    }

# Example usage
metrics = extract_financial_metrics("path/to/filing.txt")
print(metrics)

import pandas as pd

def clean_data(metrics):
    for key, value in metrics.items():
        if value:
            metrics[key] = float(value.replace(",", ""))
    return metrics

# Example usage
cleaned_metrics = clean_data(metrics)
print(cleaned_metrics)

def compare_to_industry(company_metrics, industry_metrics):
    comparison = {}
    for metric in company_metrics.keys():
        if metric in industry_metrics:
            comparison[metric] = {
                "company": company_metrics[metric],
                "industry_avg": industry_metrics[metric],
                "comparison": company_metrics[metric] / industry_metrics[metric]
            }
    return comparison

# Example industry average metrics
industry_metrics = {
    "revenue": 500000000,  # Example industry average revenue
    "profit": 50000000,    # Example industry average profit
    "assets": 1000000000,  # Example industry average assets
    "liabilities": 400000000  # Example industry average liabilities
}

# Example usage
comparison = compare_to_industry(cleaned_metrics, industry_metrics)
print(comparison)

from sklearn.linear_model import LinearRegression

def predict_future_growth(company_metrics, industry_metrics, past_data):
    # Create a DataFrame for the past data
    df = pd.DataFrame(past_data)
    
    # Fit a linear regression model to predict future revenue based on past revenue
    model = LinearRegression()
    model.fit(df[['year']], df['revenue'])
    
    # Predict future revenue for the next year
    next_year = df['year'].max() + 1
    predicted_revenue = model.predict([[next_year]])
    
    # Compare predicted growth with industry growth rate
    industry_growth_rate = (industry_metrics['revenue'] / past_data[-1]['revenue']) - 1
    company_growth_rate = (predicted_revenue[0] / company_metrics['revenue']) - 1
    
    return {
        "predicted_revenue": predicted_revenue[0],
        "company_growth_rate": company_growth_rate,
        "industry_growth_rate": industry_growth_rate,
        "undervalued": company_growth_rate > industry_growth_rate
    }

# Example past data
past_data = [
    {"year": 2018, "revenue": 400000000},
    {"year": 2019, "revenue": 450000000},
    {"year": 2020, "revenue": 480000000},
    {"year": 2021, "revenue": 490000000}
]

# Example usage
valuation = predict_future_growth(cleaned_metrics, industry_metrics, past_data)
print(valuation)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_valuation_model(data):
    # Prepare the dataset
    df = pd.DataFrame(data)
    X = df.drop(columns=['valuation'])
    y = df['valuation']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score:.2f}")
    
    return model

# Example dataset (with synthetic valuation data)
data = [
    {"revenue": 500000000, "profit": 50000000, "assets": 1000000000, "liabilities": 400000000, "valuation": 1.2},
    {"revenue": 600000000, "profit": 60000000, "assets": 1100000000, "liabilities": 450000000, "valuation": 1.3},
    # Add more historical data points...
]

# Example usage
valuation_model = train_valuation_model(data)
