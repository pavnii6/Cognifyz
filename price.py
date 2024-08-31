import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Dataset
file_path = r'C:\Users\pavni\OneDrive\Documents\Dataset .csv'
df = pd.read_csv(file_path)

# Step 2: Data Preprocessing

# Handle Missing Values
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Convert 'Yes'/'No' to 1/0 for Table Booking and Online Delivery columns
df['Has Table booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Has Online delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert Price Range to categorical if it's not already
df['Price range'] = df['Price range'].astype('category')

# Step 3: Analyze Price Range Data

# Determine the most common price range
most_common_price_range = df['Price range'].mode()[0]
print(f"Most common price range: {most_common_price_range}")

# Calculate the average rating for each price range
avg_rating_by_price_range = df.groupby('Price range')['Aggregate rating'].mean()

# Print the average rating for each price range
print("Average rating for each price range:")
print(avg_rating_by_price_range)

# Identify the color that represents the highest average rating
max_avg_rating = avg_rating_by_price_range.max()
color_with_highest_rating = avg_rating_by_price_range.idxmax()

print(f"Color representing the highest average rating: {color_with_highest_rating} with an average rating of {max_avg_rating:.2f}")

# Step 4: Visualization

# Visualization: Average Ratings by Price Range
sns.barplot(x=avg_rating_by_price_range.index,
            y=avg_rating_by_price_range.values,
            palette='viridis')  # Using a colormap to represent different colors
plt.title('Average Ratings by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Average Rating')
plt.show()
