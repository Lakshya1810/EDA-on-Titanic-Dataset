Titanic Data Analysis Project

This project explores the famous Titanic dataset to understand the characteristics of passengers and analyze the factors that influenced survival rates. It involves basic data cleaning, visualization, and exporting a cleaned version of the dataset.
📂 Dataset

The dataset used is the Titanic dataset provided by Data Science Dojo. It includes information about the passengers such as age, sex, ticket class, fare, and whether they survived the Titanic disaster.
🛠️ Libraries Used

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

🔍 Project Workflow
1. Load the Dataset

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

2. Display Basic Information

df.head()
df.info()
df.describe()

3. Handle Missing Values

    Fill missing values in Age with the median.

    Drop rows with missing Embarked values.

    Fill missing Cabin entries with 'Unknown'.

df['Age'] = df['Age'].fillna(df['Age'].median())
df.dropna(subset=['Embarked'], inplace=True)
df['Cabin'] = df['Cabin'].fillna('Unknown')

4. Data Visualization

    Survival Count

    Survival by Gender

    Age Distribution

sns.countplot(x='Survived', data=df)
sns.countplot(x='Survived', hue='Sex', data=df)
sns.histplot(df['Age'], bins=30, kde=True)

5. Save Cleaned Dataset

df.to_csv("cleaned_titanic.csv", index=False)

🔮 Future Updates

Here are some planned improvements and updates to this project:

Perform feature engineering to create new insights (e.g., title extraction from names, family size).

Encode categorical variables for machine learning.

Build predictive models using Logistic Regression, Random Forest, and KNN to predict survival.

Evaluate model performance using confusion matrix, ROC curve, and classification report.

Use cross-validation to improve model generalization.

Visualize correlations using a heatmap.

    Deploy the model using Flask or Streamlit for interactive prediction.

📁 Output

    cleaned_titanic.csv: The cleaned version of the dataset after handling missing values.

📌 Notes

Make sure to install the required Python packages if not already installed:

pip install pandas matplotlib seaborn
