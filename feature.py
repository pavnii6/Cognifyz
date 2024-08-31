import pandas as pd
import matplotlib.pyplot as plt

# Load the Dataset
file_path = r'C:\Users\pavni\OneDrive\Documents\Updated_Dataset.csv'
df = pd.read_csv(file_path)

# Plot length of restaurant names
plt.figure(figsize=(12, 6))
plt.hist(df['Name Length'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Restaurant Name Lengths')
plt.xlabel('Length of Restaurant Name')
plt.ylabel('Number of Restaurants')
plt.grid(True)
plt.show()

# Plot length of addresses
plt.figure(figsize=(12, 6))
plt.hist(df['Address Length'], bins=30, color='salmon', edgecolor='black')
plt.title('Distribution of Address Lengths')
plt.xlabel('Length of Address')
plt.ylabel('Number of Restaurants')
plt.grid(True)
plt.show()

# Compare the average ratings based on table booking availability
avg_rating_table_booking = df.groupby('Has Table Booking')['Aggregate rating'].mean()
plt.figure(figsize=(8, 6))
avg_rating_table_booking.plot(kind='bar', color=['orange', 'lightgreen'])
plt.title('Average Ratings: With vs. Without Table Booking')
plt.xlabel('Has Table Booking')
plt.ylabel('Average Rating')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.grid(axis='y')
plt.show()

# Compare the average ratings based on online delivery availability
avg_rating_online_delivery = df.groupby('Has Online Delivery')['Aggregate rating'].mean()
plt.figure(figsize=(8, 6))
avg_rating_online_delivery.plot(kind='bar', color=['purple', 'yellow'])
plt.title('Average Ratings: With vs. Without Online Delivery')
plt.xlabel('Has Online Delivery')
plt.ylabel('Average Rating')
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], rotation=0)
plt.grid(axis='y')
plt.show()

# Plot the distribution of restaurants by price range
price_range_columns = [col for col in df.columns if col.startswith('Price range')]
price_range_counts = df[price_range_columns].sum()

plt.figure(figsize=(10, 6))
price_range_counts.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Number of Restaurants by Price Range')
plt.xlabel('Price Range')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
