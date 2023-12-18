# Movie Recommendation System
## Overview

### This project implements a movie recommendation system using collaborative filtering. The system is based on the k-Nearest Neighbors algorithm, specifically designed to recommend movies based on user preferences and similarity to other movies.
Contents

    Introduction
    Data Exploration
    Data Cleaning
    Statistic Analysis
    Machine Learning - Recommendation
    Usage
    Dependencies
    Contributing

## Introduction

The goal of this project is to provide users with movie recommendations based on their preferences. The system analyzes a dataset containing information about movies, such as genre, score, and user ratings, to generate personalized recommendations.

## Data Exploration

Explore the dataset to gain insights into movie ratings, scores, genres, and other relevant information. Visualizations are included to provide a better understanding of the data distribution.

### Distribution of Ratings

```python
# Set the style for seaborn
sns.set(style="whitegrid")

# Distribution of Ratings
plt.figure(figsize=(12, 6))
sns.countplot(x='rating', data=data, order=data['rating'].value_counts().index, palette='viridis')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
```

### Distribution of Score

```python
plt.figure(figsize=(12, 6))
sns.histplot(data['score'].dropna(), kde=True, color='skyblue')
plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Count')
plt.show()
```
### Distribution of Genres

```python
plt.figure(figsize=(14, 8))
sns.countplot(y='genre', data=data, order=data['genre'].value_counts().index, palette='muted')
plt.title('Distribution of Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()
```
### Distribution of Country

```python
plt.figure(figsize=(14, 12))
sns.countplot(y='country', data=data, order=data['country'].value_counts().index, palette='pastel')
plt.title('Distribution of Countries')
plt.xlabel('Count')
plt.ylabel('Country')
plt.show()
```
### Correlation Matrix

```python
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```
### Budget vs. Gross Relationship
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='budget', y='gross', data=data, color='purple', alpha=0.7)
plt.title('Budget vs. Gross Earnings')
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')
plt.show()
```
## Data Cleaning
Handle missing values, outliers, and perform necessary data cleaning procedures to ensure the accuracy and reliability of the recommendation system.

## Statistic Analysis

Conduct statistical analysis to derive meaningful insights from the dataset. This may include descriptive statistics, group statistics by genre or country, and hypothesis testing.

## Machine Learning - Recommendation

Implement a movie recommendation system using the k-Nearest Neighbors algorithm. Users can input their favorite movie, and the system will suggest similar movies based on the algorithm's analysis.

## Usage

    Clone the repository to your local machine.
    Install the required dependencies.
    Run the recommendation system script.
    Enter your favorite movie when prompted.
    Receive personalized movie recommendations.

## Dependencies

    Pandas
    Matplotlib
    Seaborn
    Scikit-learn
    Tkinter (for GUI)
    Surprise (for collaborative filtering)


## Contributing

Contributions are welcome! If you have ideas for improvements or find any issues, feel free to open an issue or submit a pull request.
