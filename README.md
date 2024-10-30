# Vietnam gold price prediction
This gold prediction system aims to predict future gold prices in Vietnam by analyzing past data and applying machine learning models and this code uses libraries like Pandas, NumPy, Matplotlib, Plotly and Scikit-learn for data processing, visualization and modeling.
The dataset used from 12/1989 to 04/2024

# Key Components
- Data Preprocessing:
The code starts by loading historical gold price data from a CSV file (gooh.csv), then cleans it by removing any rows with missing values.
The date column is converted into a datetime format, ensuring accurate time-based processing.
Data is filtered to include records from 2015 onwards, and each date is converted to the number of days since the start date, creating a continuous numeric feature for modeling.

- Data Visualization:
An initial line chart is created using Plotly to visualize the historical gold prices over time, giving insights into the overall trend.

- Modeling:
The code applies a Linear Regression model as well as a Random Forest Regression to predict gold prices. These models are evaluated using Mean Squared Error (MSE) to assess prediction accuracy.

- System Goals:
Predict Future Prices: Provide reliable gold price forecasts to aid investment decisions.
Visualize Trends: Offer intuitive visualizations to help users understand market trends over time.
Evaluate Model Performance: Compare different models to ensure accurate and reliable predictions.

# Result
### Linear Regression:
![newplot](https://github.com/user-attachments/assets/856399e6-d85a-4d81-8934-faaa60fb189f)

### Random Forest Regression
![newplot (1)](https://github.com/user-attachments/assets/9e8a397e-09b4-49f1-bc46-f16244a60c58)

