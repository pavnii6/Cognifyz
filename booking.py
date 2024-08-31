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

# Step 3: Percentage of Restaurants Offering Table Booking and Online Delivery

table_booking_percentage = df['Has Table booking'].mean() * 100
online_delivery_percentage = df['Has Online delivery'].mean() * 100

print(f"Percentage of restaurants offering table booking: {table_booking_percentage:.2f}%")
print(f"Percentage of restaurants offering online delivery: {online_delivery_percentage:.2f}%")

# Step 4: Compare Average Ratings of Restaurants with and without Table Booking

avg_rating_with_table_booking = df[df['Has Table booking'] == 1]['Aggregate rating'].mean()
avg_rating_without_table_booking = df[df['Has Table booking'] == 0]['Aggregate rating'].mean()

print(f"Average rating with table booking: {avg_rating_with_table_booking:.2f}")
print(f"Average rating without table booking: {avg_rating_without_table_booking:.2f}")

# Visualization: Average Ratings with and without Table Booking
sns.barplot(x=['With Table Booking', 'Without Table Booking'],
            y=[avg_rating_with_table_booking, avg_rating_without_table_booking],
            palette='Blues')
plt.title('Average Ratings: With vs. Without Table Booking')
plt.ylabel('Average Rating')
plt.show()

# Step 5: Analyze Availability of Online Delivery Among Different Price Ranges

online_delivery_by_price_range = df.groupby('Price range')['Has Online delivery'].mean() * 100

print("Online delivery availability by price range:")
print(online_delivery_by_price_range)

# Visualization: Online Delivery by Price Range
sns.barplot(x=online_delivery_by_price_range.index,
            y=online_delivery_by_price_range.values,
            palette='Greens')
plt.title('Online Delivery Availability by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Percentage of Restaurants Offering Online Delivery')
plt.show()

# Step 6: Compare Ratings for Restaurants with and without Online Delivery

avg_rating_with_online_delivery = df[df['Has Online delivery'] == 1]['Aggregate rating'].mean()
avg_rating_without_online_delivery = df[df['Has Online delivery'] == 0]['Aggregate rating'].mean()

print(f"Average rating with online delivery: {avg_rating_with_online_delivery:.2f}")
print(f"Average rating without online delivery: {avg_rating_without_online_delivery:.2f}")

# Visualization: Average Ratings with and without Online Delivery
sns.barplot(x=['With Online Delivery', 'Without Online Delivery'],
            y=[avg_rating_with_online_delivery, avg_rating_without_online_delivery],
            palette='Oranges')
plt.title('Average Ratings: With vs. Without Online Delivery')
plt.ylabel('Average Rating')
plt.show()
