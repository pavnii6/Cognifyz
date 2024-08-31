import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load the Dataset
file_path = r'C:\Users\pavni\OneDrive\Documents\Dataset .csv'
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing

# Handle Missing Values
numerical_cols = df.select_dtypes(include=['number']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Identify and drop non-relevant or problematic columns
# Replace these column names with actual column names if different
df = df.drop(columns=['Restaurant Name', 'Address', 'Locality', 'Locality Verbose', 'Currency'])  # Dropping 'Currency' column

# Encode Categorical Variables
df_encoded = pd.get_dummies(df, columns=['Country Code', 'City', 'Cuisines'], drop_first=True)

# Step 3: Define Features and Target Variable
X = df_encoded.drop(columns=['Aggregate rating'])
y = df_encoded['Aggregate rating']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training and Evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

results = {}
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results[name] = {'MAE': mae, 'RMSE': rmse}
    
    print(f'{name}:')
    print(f'  Mean Absolute Error (MAE): {mae:.4f}')
    print(f'  Root Mean Squared Error (RMSE): {rmse:.4f}\n')

# Step 6: Model Performance Comparison
results_df = pd.DataFrame(results).T
print("Model Performance Comparison:")
print(results_df)

results_df[['MAE', 'RMSE']].plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Error')
plt.xticks(rotation=0)
plt.legend(title='Metrics')
plt.tight_layout()
plt.show()

# Step 7: Additional Visualizations

for name, y_pred in predictions.items():
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    sns.lineplot(x=y_test, y=y_test, color='red')  # Perfect prediction line
    plt.title(f'Actual vs Predicted Ratings ({name})')
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.tight_layout()
    plt.show()
    
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'Residuals Plot ({name})')
    plt.xlabel('Predicted Ratings')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.show()

# Step 8: Feature Importance (for Tree-Based Models)
for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest']:
        feature_importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        
        top_features = importance_df.sort_values(by='Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        plt.title(f'Top 10 Feature Importances ({name})')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
