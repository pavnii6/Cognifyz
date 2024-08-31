import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Step 1: Load the dataset
file_path = r'C:\Users\pavni\OneDrive\Documents\Dataset .csv'
df = pd.read_csv(file_path)

# Step 2: Identify the number of rows and columns
rows, columns = df.shape
print(f'The dataset contains {rows} rows and {columns} columns.')

# Step 3: Check for missing values and handle them
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Fill missing values for numeric columns
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Handle missing values for non-numeric columns if any
# df['non_numeric_column'] = df['non_numeric_column'].fillna('default_value')

# Step 4: Perform data type conversion if necessary
print("Data types before conversion:")
print(df.dtypes)

# Example conversion: Convert a specific column to numeric if necessary
# df['some_column'] = pd.to_numeric(df['some_column'], errors='coerce')

print("Data types after conversion:")
print(df.dtypes)

# Step 5: Analyze the distribution of the target variable
# Check if the target variable is continuous or categorical
target_col = 'Aggregate rating'

# Determine if the target column is continuous
if pd.api.types.is_numeric_dtype(df[target_col]):
    # For continuous target variable (regression)
    print("Target variable is continuous.")
    
    # Plot distribution for continuous target
    sns.histplot(df[target_col], bins=30, kde=True)
    plt.title('Distribution of Aggregate Rating')
    plt.show()

    # No SMOTE for regression; handle imbalance through other means
else:
    # For categorical target variable (classification)
    print("Target variable is categorical.")
    
    # Convert target variable to categorical if it's continuous
    df[target_col] = pd.cut(df[target_col], bins=[0, 1, 2, 3, 4, 5], labels=[1, 2, 3, 4, 5])
    
    # Plot distribution for categorical target
    sns.countplot(x=target_col, data=df)
    plt.title('Distribution of Aggregate Rating')
    plt.show()
    
    # Check for class imbalance
    class_counts = df[target_col].value_counts()
    print("Class distribution of Aggregate Rating:")
    print(class_counts)
    
    # Handle class imbalance using SMOTE
    X = df.drop(columns=[target_col])
    y = df[target_col]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    print("Distribution of Aggregate Rating after SMOTE:")
    print(pd.Series(y_res).value_counts())
