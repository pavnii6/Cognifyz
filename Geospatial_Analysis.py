import pandas as pd
import geopandas as gpd
import folium
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = r'C:\Users\pavni\OneDrive\Documents\Dataset .csv'
df = pd.read_csv(file_path)

# Step 2: Visualize the Locations of Restaurants on a Map using Latitude and Longitude
# Ensure the dataset has 'Latitude' and 'Longitude' columns

# Create a folium map centered around the mean latitude and longitude
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)

# Add markers for each restaurant
for _, row in df.iterrows():
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Restaurant Name']).add_to(m)

# Save map as an HTML file and display
m.save('restaurants_map.html')

# Step 3: Analyze the Distribution of Restaurants Across Different Cities or Countries
city_counts = df['City'].value_counts()
country_counts = df['Country Code'].value_counts()

# Visualize the distribution across cities
plt.figure(figsize=(10, 6))
sns.barplot(x=city_counts.values, y=city_counts.index, palette='viridis')
plt.title('Distribution of Restaurants Across Cities')
plt.xlabel('Number of Restaurants')
plt.ylabel('City')
plt.show()

# Visualize the distribution across countries
plt.figure(figsize=(10, 6))
sns.barplot(x=country_counts.values, y=country_counts.index, palette='plasma')
plt.title('Distribution of Restaurants Across Countries')
plt.xlabel('Number of Restaurants')
plt.ylabel('Country Code')
plt.show()

# Step 4: Correlation Analysis Between Location and Rating
# Visualizing the relationship between Latitude/Longitude and Aggregate Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='Aggregate rating', data=df, palette='coolwarm')
plt.title('Correlation Between Restaurant Location and Rating')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Calculate correlation between Latitude, Longitude, and Aggregate Rating
correlation = df[['Latitude', 'Longitude', 'Aggregate rating']].corr()
print("Correlation Matrix:")
print(correlation)
