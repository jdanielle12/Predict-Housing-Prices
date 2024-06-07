# Ames Housing Price Prediction and Forecasting

## Table of Contents
1. Overview
2. Dataset
3. Dependencies
4. Data Preprocessing
5. Exploratory Data Analysis (EDA)
6. Model Training and Evaluation
7. Hyperparameter Tuning
8. Forecasting Future Trends
9. Results
10. Conclusion
11. Usage

## Overview
This project aims to analyze the Ames Housing dataset to predict house prices using various regression models and to forecast future trends using the Facebook Prophet model. The project involves data preprocessing, exploratory data analysis (EDA), model training and evaluation, hyperparameter tuning, and time series forecasting.

## Dataset
The dataset used in this project is the Ames Housing dataset from Kaggle, which includes numeroud features describing houses in Ames, Iowa, along with their sale prices. The dataset can be accessed [here](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

## Dependencies
The project requires the following Python libraries:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* prophet

These libraries can be installed using pip. Ensure you have them installed before running the code.

## Data Preprocessing

### Loading the Data
The first step involves loading the dataset and examining its structure. This includes checking the data types of each column and displaying the first few rows of the dataset to get an initial understanding of the data.

### Handling Missing Values
Missing values are identified and handled appropriately. Numerical features are imputed with their mean values to ensure there are no gaps in the data, which could otherwise affect the performance of the machine learning models.

### Feature Engineering
New features are created to enhance the dataset. For example, `TotalSF` is the sum of the first floor, second floor, and basement square footage, and `Age` is the difference between the year sold and the year built. These features can potentially improve the predictive power of the models. 

### Selecting Important Features
Key features relevant to predicting the house prices are selected. This step reduces the dimensionality of the dataset and focuses on the more impactful variables.

## Exploratory Data Analysis
EDA helps to understand the underlying patterns and distributions in the data. Several visualization techniques are employed:

### Summary Statistics
Summary statistics provide an overview of the dataset, including measures of central tendency and variability. 
![Summary Statistics](images/SummaryStatistics.png)

### Histograms
Histograms are plotted for numerical features to visualize their distributions. This helps in understanding the spread and skewness of the data.
![Histograms](images/histogram.png)

### Scatter Plots
Scatter plots are used to visualize the relationships between numerical features and the target variable `SalePrice`. This helps in identifying any correlations.
![Scatter Plots](images/SummaryStatistics.png)

### Correlation Heatmap
A heatmap is generated to visualize the correlation between different features. Features with high correlation to the target variable are of particular interest.
![Correlation Heatmap](images/heatmap.png)

### Box Plots
Box plots are used to identify outliers in numerical features. Outliers can significantly affect the performance of machine learning models and may need to be handled appropriately. 
![Box Plots](images/boxplots.png)

## Model Training and Evaluation

### Splitting the Data
The dataset is split into training and testing sets to evaluate the models' performance. The training set is used to train the models, and the testing set is used to evaluate how well the models generalize to unseen data.



### Contributors
* Walter Hickman Jr. 
* Jamie McGraner
* Anand Punwani
* Anthony Wilson
