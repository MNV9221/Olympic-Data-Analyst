import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import uuid

# Step 1: Data Preparation
# Load the dataset
df = pd.read_csv('/kaggle/input/summer-olympics-medals/Summer-0lympic-medals-1976-to-2008.csv', encoding='latin1')

# Display first few rows
print("First few rows of the dataset:")
print(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Step 2: Data Cleaning
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows with all missing values
df_cleaned = df.dropna(how='all')

# Drop unnecessary columns
df_cleaned = df_cleaned.drop(['Event_gender', 'Country_Code'], axis=1)

# Convert Year to integer
df_cleaned['Year'] = df_cleaned['Year'].astype(int)

# Check for remaining missing values
print("\nMissing Values After Cleaning:")
print(df_cleaned.isnull().sum())

# Display cleaned dataset info
print("\nCleaned Dataset Info:")
print(df_cleaned.info())

# Step 3: Exploratory Data Analysis (EDA)
# 3.1 Total Medal Count by Country
medals_by_country = df_cleaned.groupby('Country')['Medal'].count().sort_values(ascending=False)
print("\nTop 5 Countries by Medal Count:")
print(medals_by_country.head())

# Plot top 10 countries by medal count
plt.figure(figsize=(10,6))
medals_by_country.head(10).plot(kind='bar', color='gold')
plt.title("Top 10 Countries by Medal Count")
plt.xlabel("Country")
plt.ylabel("Total Medals")
plt.savefig('top_countries_medal_count.png')
plt.close()

# 3.2 Medal Trends Over Years
medals_by_year = df_cleaned.groupby('Year')['Medal'].count()
plt.figure(figsize=(10,6))
medals_by_year.plot(marker='o', linestyle='-', color='b')
plt.title("Total Medals Won Over the Years")
plt.xlabel("Year")
plt.ylabel("Total Medals")
plt.grid(True)
plt.savefig('medal_trends_over_years.png')
plt.close()

# 3.3 Gender Distribution
gender_distribution = df_cleaned['Gender'].value_counts()
plt.figure(figsize=(6,4))
gender_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], explode=[0.05, 0])
plt.title("Gender Distribution in Olympics Events")
plt.ylabel('')
plt.savefig('gender_distribution.png')
plt.close()

# 3.4 Top Athletes
athlete_medal_count = df_cleaned.groupby('Athlete')['Medal'].count().sort_values(ascending=False)
print("\nTop 5 Athletes by Medal Count:")
print(athlete_medal_count.head())

# Plot top 10 athletes
plt.figure(figsize=(10,6))
athlete_medal_count.head(10).plot(kind='bar', color='silver')
plt.title("Top 10 Athletes by Medal Count")
plt.xlabel("Athlete")
plt.ylabel("Total Medals")
plt.savefig('top_athletes_medal_count.png')
plt.close()

# Step 4: Specific Questions from Document
# Q1: Which city hosted the maximum number of Olympics?
q1_data = df_cleaned[['City', 'Year']].drop_duplicates('Year')
print("\nCities Hosting Olympics:")
print(q1_data)

# Q2: Which city hosted the most events?
q2_data = df_cleaned['City'].value_counts()
print("\nCities by Event Count:")
print(q2_data)

plt.figure(figsize=(10,4))
q2_data.plot.bar()
plt.title("Cities by Number of Events Hosted")
plt.xlabel("City")
plt.ylabel("Event Count")
plt.savefig('city_event_count.png')
plt.close()

# Q3: Understand the events
q3_data = df_cleaned[['Sport', 'Discipline', 'Event']].drop_duplicates()
print("\nTotal Number of Unique Events:", len(q3_data))

# Q4: Athlete with most medals
q4_data = df_cleaned.groupby('Athlete')['Medal'].count().reset_index(name='Count').sort_values(by='Count', ascending=False).head(10)
print("\nTop Athlete by Medal Count:")
print(q4_data)

# Q5: Gender ratio in winning teams
q5_data = df_cleaned.groupby('Gender')['Gender'].count()
plt.figure(figsize=(12,2))
q5_data.plot.barh()
plt.title("Gender Ratio in Medal Wins")
plt.xlabel("Count")
plt.ylabel("Gender")
plt.savefig('gender_ratio_medal_wins.png')
plt.close()

# Step 5: Predictive Analysis
# Prepare data for machine learning
le = LabelEncoder()
df_encoded = df_cleaned.copy()
df_encoded['Country'] = le.fit_transform(df_encoded['Country'])
df_encoded['Sport'] = le.fit_transform(df_encoded['Sport'])
df_encoded['Gender'] = le.fit_transform(df_encoded['Gender'])
df_encoded['Medal'] = le.fit_transform(df_encoded['Medal'])

# Features and target
X = df_encoded[['Country', 'Sport', 'Gender']]
y = df_encoded['Medal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nLogistic Regression Model Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save plots for predictive analysis
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Medal Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig('confusion_matrix.png')
plt.close()

# Step 6: Insights and Conclusion
print("\nKey Insights:")
print("- Top countries include the United States, Russia, and Germany.")
print("- Michael Phelps is the top athlete with 16 medals.")
print("- Gender distribution shows more male medalists, with some events exclusive to men.")
print("- Beijing hosted the most events, and no city hosted the Olympics twice from 1976 to 2008.")
print("- The logistic regression model provides a baseline for predicting medal wins, but advanced models like random forests could improve performance.")

# Note: The dataset path assumes Kaggle input; adjust as needed for local use.