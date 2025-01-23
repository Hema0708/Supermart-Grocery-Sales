# Supermart-Grocery-Sales

This project involves analyzing a fictional dataset that simulates grocery sales through a delivery application in Tamil Nadu, India. The dataset is intended for data analysts and data scientists to practice exploratory data analysis (EDA) and machine learning techniques.

Tools Used

Programming Languages : Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

Database: SQL

Spreadsheet Software: Excel

Dataset Description
The dataset comprises various attributes related to customer orders, including:
Order ID
Customer Name
Category
Sub Category
City
Order Date
Region
Sales
Discount
Profit
State
Month Number
Month Name
Year

Project Steps

Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

 Step 2: Load the Dataset
data = pd.read_csv('supermart_grocery_sales.csv')
print(data.head())

Step 3: Data Preprocessing
1.Check for Missing Values
   print(data.isnull().sum())
   data.dropna(inplace=True)
   data.drop_duplicates(inplace=True)

2. Convert Date Columns
   data['Order Date'] = pd.to_datetime(data['Order Date'])
   data['Order Day'] = data['Order Date'].dt.day
   data['Order Month'] = data['Order Date'].dt.month
   data['Order Year'] = data['Order Date'].dt.year

3. Label Encoding for Categorical Variables
   le = LabelEncoder()
   categorical_columns = ['Category', 'Sub Category', 'City', 'Region', 'State', 'Month']
    for col in categorical_columns:
       data[col] = le.fit_transform(data[col])
   print(data.head())

 Step 4: Exploratory Data Analysis (EDA)
1. Sales Distribution by Category:
   plt.figure(figsize=(10, 6))
   sns.boxplot(x='Category', y='Sales', data=data, palette='Set2')
   plt.title('Sales Distribution by Category')
   plt.xlabel('Category')
   plt.ylabel('Sales')
   plt.show()

2. Sales Trends Over Time:
   plt.figure(figsize=(12, 6))
   data.groupby('Order Date')['Sales'].sum().plot()
   plt.title('Total Sales Over Time')
   plt.xlabel('Date')
   plt.ylabel('Total Sales')
   plt.show()

3. Correlation Heatmap:
   plt.figure(figsize=(12, 6))
   corr_matrix = data.corr()
   sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
   plt.title('Correlation Heatmap')
   plt.show()

Step 5: Feature Selection and Model Building
features = data.drop(columns=['Order ID', 'Customer Name', 'Order Date', 'Sales'])
target = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Step 6: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

Step 7: Evaluate the Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
 Step 8: Visualize the Results - Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

Conclusion and Next Steps
The linear regression model effectively predicts sales based on selected features with an R-squared value indicating a good fit. Future enhancements may include:

1. Experimenting with more complex models (e.g., Random Forest or XGBoost).
2. Conducting further feature engineering to improve model accuracy.
3. Deploying the model into a real-time analytics dashboard.

This project serves as a practical introduction to retail sales analysis using machine learning techniques.

Insights from EDA

1. The category of "Eggs, Meat & Fish" contributes significantly to total sales.
2. Monthly sales trends indicate consistent growth over time.
3. Yearly sales show a positive trend with substantial contributions from 2017 and 2018.
This report provides a comprehensive overview of the dataset and outlines actionable insights for future business strategies in retail analytics.
