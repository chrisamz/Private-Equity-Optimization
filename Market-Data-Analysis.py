pip install alpha_vantage pandas

from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Alpha Vantage API key
api_key = 'your_alpha_vantage_api_key'

# Initialize Alpha Vantage TimeSeries
ts = TimeSeries(key=api_key, output_format='pandas')

# Get daily market data for a specific company (e.g., Apple Inc., symbol: AAPL)
market_data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

# Display the first few rows of the market data
print(market_data.head())

def extract_market_features(market_data):
    # Calculate moving averages
    market_data['SMA_50'] = market_data['4. close'].rolling(window=50).mean()
    market_data['SMA_200'] = market_data['4. close'].rolling(window=200).mean()

    # Calculate daily returns
    market_data['daily_return'] = market_data['4. close'].pct_change()

    # Fill missing values
    market_data.fillna(0, inplace=True)

    return market_data

# Extract features from market data
market_features = extract_market_features(market_data)
print(market_features.head())

def combine_features(financial_metrics, market_features):
    # Example financial metrics (replace with actual extracted data)
    financial_data = pd.DataFrame([financial_metrics])

    # Combine financial data with market features
    combined_data = pd.concat([financial_data, market_features], axis=1)
    return combined_data

# Example financial metrics
financial_metrics = {
    "revenue": 500000000,
    "profit": 50000000,
    "assets": 1000000000,
    "liabilities": 400000000
}

# Combine features
combined_features = combine_features(financial_metrics, market_features)
print(combined_features.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Prepare the dataset
X = combined_features.drop(columns=['valuation'])
y = combined_features['valuation']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Mean Absolute Error: {mae:.2f}")

# Example usage: Predict future growth and valuation for a new set of features
new_features = X_test.iloc[0]
predicted_valuation = model.predict([new_features])
print(f"Predicted Valuation: {predicted_valuation[0]:.2f}")

from sklearn.model_selection import cross_val_score

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print(f"Cross-validated Mean Absolute Error: {-cv_scores.mean():.2f}")

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
    "revenue": 600000000,  # Example industry average revenue
    "profit": 60000000,    # Example industry average profit
    "assets": 1100000000,  # Example industry average assets
    "liabilities": 450000000  # Example industry average liabilities
}

# Example usage
comparison = compare_to_industry(financial_metrics, industry_metrics)
print(comparison)

