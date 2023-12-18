#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats 


# In[2]:


path = "/Users/neal/Desktop/Jupyter/movies.csv"
data = pd.read_csv(path)


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.describe


# # DATA EXPLORATION

# In[6]:


# Set the style for seaborn
sns.set(style="whitegrid")

# Distribution of Ratings
plt.figure(figsize=(12, 6))
sns.countplot(x='rating', data=data, order=data['rating'].value_counts().index, palette='viridis')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Distribution of Scores
plt.figure(figsize=(12, 6))
sns.histplot(data['score'].dropna(), kde=True, color='skyblue')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()

# Distribution of Genres
plt.figure(figsize=(14, 8))
sns.countplot(y='genre', data=data, order=data['genre'].value_counts().index, palette='muted')
plt.title('Distribution of Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Distribution of Countries
plt.figure(figsize=(14, 12))
sns.countplot(y='country', data=data, order=data['country'].value_counts().index, palette='pastel')
plt.title('Distribution of Countries')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()

# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Budget vs. Gross Relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='budget', y='gross', data=data, color='purple', alpha=0.7)
plt.title('Budget vs. Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')
plt.show()


# ### Distribution of Ratings
# 
#     Chart Type: Bar Chart
#     Purpose: Shows the count of movies in each rating category (e.g., R, PG).
#     Insight: Most movies fall into specific rating categories, providing an overview of the distribution of movie ratings.
# 
# ### Distribution of Scores
# 
#     Chart Type: Histogram
#     Purpose: Displays the distribution of movie scores.
#     Insight: Helps understand the spread and concentration of scores, identifying common score ranges.
# 
# ### Distribution of Genres
# 
#     Chart Type: Horizontal Bar Chart
#     Purpose: Illustrates the count of movies in each genre.
#     Insight: Highlights popular genres and their relative representation in the dataset.
# 
# ### Distribution of Countries
# 
#     Chart Type: Horizontal Bar Chart
#     Purpose: Shows the count of movies from different countries.
#     Insight: Reveals the distribution of movies across countries, indicating the prevalence of films from specific regions.
# 
# ### Correlation Matrix
# 
#     Chart Type: Heatmap
#     Purpose: Displays the correlation between numeric variables (e.g., score, votes, budget).
#     Insight: Helps identify relationships between variables; positive values indicate positive correlations, while negative values indicate negative correlations.
# 
# ### Budget vs. Gross Earnings Relationship
# 
#     Chart Type: Scatter Plot
#     Purpose: Illustrates the relationship between movie budgets and gross earnings.
#     Insight: Provides insights into how movie budgets correlate with their gross earnings; helps identify trends or outliers.
# 
# #### These charts collectively offer a comprehensive view of the dataset, allowing for insights into ratings, scores, genres, countries, and relationships between numeric variables. Feel free to ask if you have specific questions or if there's anything else you'd like to explore!

# # DATA CLEANING

# In[7]:


# Check for missing values in each column
missing_values = data.isnull().sum()
# Display columns with missing values
columns_with_missing_values = missing_values[missing_values > 0]
print("Columns with Missing Values:")
print(columns_with_missing_values)


# ### Handling 'Rating' column

# In[8]:


data['rating'].fillna('N/A', inplace=True)


# ### Handling 'Released' column

# In[9]:


data['released'].fillna('Unknown', inplace=True)


# ### Handling 'Score' column

# In[10]:


data['score'].fillna(data['score'].mean(), inplace=True)


# ### Handling 'Votes' column

# In[11]:


data['votes'].fillna(data['votes'].median(), inplace=True)


# ### Handling 'Writer, Star, Country' column

# In[12]:


data['writer'].fillna('Unknown', inplace=True)
data['star'].fillna('Unknown', inplace=True)
data['country'].fillna('Unknown', inplace=True)


# ### Handling 'Budget & Gross' column

# In[13]:


data['budget'].fillna(0, inplace=True)
data['gross'].fillna(0, inplace=True)


# ### Handling 'Company' column

# In[14]:


data['company'].fillna('Unknown', inplace=True)


# ### Handling 'Run time' column

# In[15]:


data['runtime'].fillna(data['runtime'].mean(), inplace=True)


# In[16]:


# Check for missing values after handling
remaining_missing_values = data.isnull().sum()
print("Remaining Missing Values:")
print(remaining_missing_values)


# In[17]:


# Replace missing values in numerical columns with the mean
data['score'].fillna(data['score'].mean(), inplace=True)
data['votes'].fillna(data['votes'].mean(), inplace=True)
data['budget'].fillna(data['budget'].mean(), inplace=True)
data['gross'].fillna(data['gross'].mean(), inplace=True)
data['runtime'].fillna(data['runtime'].mean(), inplace=True)

# Replace missing values in categorical columns with a constant or "N/A"
data['rating'].fillna('N/A', inplace=True)
data['released'].fillna('N/A', inplace=True)
data['writer'].fillna('N/A', inplace=True)
data['star'].fillna('N/A', inplace=True)
data['country'].fillna('N/A', inplace=True)

# Drop rows with missing values in critical columns (e.g., 'name', 'director')
data.dropna(subset=['name', 'director'], inplace=True)

# Drop the 'company' column if missing values are deemed acceptable
data.drop('company', axis=1, inplace=True)

# Check the remaining missing values
remaining_missing = data.isnull().sum()

# Display the remaining missing values
print("Remaining Missing Values:")
print(remaining_missing)


# In[18]:


from matplotlib.ticker import FuncFormatter

# Select numerical columns with potential outliers
numerical_columns = ['score', 'votes', 'budget', 'gross', 'runtime']

# Create box plots to visualize potential outliers
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[column])
    
    # Format the y-axis labels using FuncFormatter
    formatter = FuncFormatter(lambda x, _: '{:,.0f}'.format(x))  # Format as comma-separated integers
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.title(f'Box Plot for {column}')
    plt.show()


# Example: Winsorizing (capping extreme values)
winsorize_columns = ['budget', 'gross']
for column in winsorize_columns:
    data[column] = scipy.stats.mstats.winsorize(data[column], limits=[0.05, 0.05])

# Check the distribution after winsorizing
for column in winsorize_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data[column])
    plt.title(f'Box Plot for {column} after Winsorizing')
    plt.show()


# # Statistic Analysis

# In[19]:


descriptive_stats = data.describe()
print(descriptive_stats)


# In[20]:


grouped_stats_genre = data.groupby('genre')['score'].mean()
grouped_stats_country = data.groupby('country')['score'].mean()
print(grouped_stats_genre)
print()
print(grouped_stats_country)


# ## Hypothesis Testing
# 

# In[21]:


from scipy.stats import ttest_ind

# Assuming 'data' is your DataFrame
genre1_scores = data[data['genre'] == 'Action']['score']
genre2_scores = data[data['genre'] == 'Adventure']['score']

t_stat, p_value = ttest_ind(genre1_scores, genre2_scores, equal_var=False)
print(f'T-statistic: {t_stat}\nP-value: {p_value}')


# # Machine Learning - Recommendation

# In[22]:


pip install scikit-surprise


# ## Type in the name of movie - Recommendation

# In[35]:


import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Assuming 'data' is your DataFrame with columns 'name', 'genre', and 'score'
# Modify as needed based on your actual DataFrame

# Drop the 'rating' column if it still exists
data = data[['name', 'genre', 'score']]

# Create a pivot table to convert the data to a user-item matrix
pivot_table = pd.pivot_table(data, values='score', index='name', columns='genre', fill_value=0)

# Fit the k-NN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
knn_model.fit(pivot_table)

# Choose a movie to find recommendations for
movie_name = input("What is your favorite movie")#'Star Wars: Episode V - The Empire Strikes Back'

# Find the index of the chosen movie in the pivot table
movie_index = pivot_table.index.get_loc(movie_name)

# Get the distances and indices of the k-nearest neighbors
distances, indices = knn_model.kneighbors(pivot_table.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=6)

# Display the recommended movies
recommended_movies = [pivot_table.index[i] for i in indices.flatten() if i != movie_index]
print(f"Recommended movies for '{movie_name}': {recommended_movies}")


# ## Full list of movie with score higher than 8

# In[31]:


print(data[data.apply(lambda row: row['score'] > 8, axis=1)]['name'].tolist())


# ## Full list of movie with score higher than 8 - Reorganize for better look

# In[34]:


for index, row in data.iterrows():
    if row['score'] > 8:
        print(row['name'])


# ## Scroll the list of movie - Recommendation

# In[39]:


import pandas as pd
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import ttk


data = data[['name', 'genre', 'score']]

# Create a pivot table to convert the data to a user-item matrix
pivot_table = pd.pivot_table(data, values='score', index='name', columns='genre', fill_value=0)

# Fit the k-NN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
knn_model.fit(pivot_table)

# Create a tkinter window
window = tk.Tk()
window.title("Movie Recommendation")

# Choose a movie using a dropdown menu
selected_movie = tk.StringVar()
movie_dropdown = ttk.Combobox(window, textvariable=selected_movie, values=pivot_table.index.tolist())
movie_dropdown.set("Choose a movie")
movie_dropdown.grid(column=0, row=0)

# Function to get recommendations based on the selected movie
def get_recommendations():
    movie_name = selected_movie.get()
    movie_index = pivot_table.index.get_loc(movie_name)
    distances, indices = knn_model.kneighbors(pivot_table.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=6)
    recommended_movies = [pivot_table.index[i] for i in indices.flatten() if i != movie_index]
    print(f"Recommended movies for '{movie_name}': {recommended_movies}")

# Button to trigger recommendation
recommend_button = tk.Button(window, text="Get Recommendations", command=get_recommendations)
recommend_button.grid(column=1, row=0)

# Run the tkinter event loop
window.mainloop()


# In[ ]:




