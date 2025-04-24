# 🚢 Titanic Survival Analysis

This project explores the **Titanic dataset** to understand the characteristics of passengers and analyze the factors that influenced survival rates. It includes **data cleaning**, **exploratory visualizations**, and saving a cleaned dataset for further modeling.

---

## 📂 Dataset

The dataset is sourced from [Data Science Dojo](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).  
It includes features such as:

- **PassengerId**: Unique ID
- **Survived**: Survival (0 = No, 1 = Yes)
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name, Sex, Age**: Personal info
- **SibSp, Parch**: Number of siblings/spouses & parents/children aboard
- **Fare**: Ticket price
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation

---

## 🛠️ Libraries Used

```python
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

    ✅ Fill missing Age values with the median.

    ✅ Drop rows where Embarked is missing.

    ✅ Replace missing Cabin entries with 'Unknown'.

df['Age'] = df['Age'].fillna(df['Age'].median())
df.dropna(subset=['Embarked'], inplace=True)
df['Cabin'] = df['Cabin'].fillna('Unknown')

4. Data Visualization
🔹 Survival Count

sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

🔹 Survival by Gender

sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival by Gender')
plt.show()

🔹 Age Distribution

sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

5. Save Cleaned Dataset

df.to_csv("cleaned_titanic.csv", index=False)

🔮 Future Updates

Planned improvements to enhance the project:

✨ Feature Engineering (e.g., title from Name, family size)

🔢 Encode categorical variables for ML

🤖 Build predictive models (Logistic Regression, Random Forest, KNN)

📈 Evaluate models (Confusion Matrix, ROC Curve, Classification Report)

🔁 Apply Cross-Validation

📊 Correlation heatmap and feature importance

    🌐 Deploy the model using Flask or Streamlit

📁 Output

    cleaned_titanic.csv: Cleaned dataset after handling missing values

⚙️ Setup Instructions

Make sure to install the required packages:

pip install pandas matplotlib seaborn

