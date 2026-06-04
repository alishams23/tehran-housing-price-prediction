# A Machine Learning Approach to Tehran Housing Price Prediction Using Ensemble Regressors


## Abstract

Automated real estate valuation is a critical challenge in computational economics, requiring robust pipelines to handle data heterogeneity and non-linear feature relationships. This project demonstrates an **end-to-end machine learning framework** designed to predict housing prices in Tehran using structural and neighborhood data.

A comprehensive dataset from Kaggle was used. Extensive Exploratory Data Analysis (EDA) was performed, followed by a robust data preprocessing pipeline including:

- Text cleaning using **Regular Expressions (Regex)**
- Missing value imputation
- Outlier detection and removal using **Interquartile Range (IQR)**
- One-Hot Encoding for categorical variables

Multiple models were trained and evaluated: Multiple Linear Regression (baseline), K-Neighbors Regressor, XGBoost Regressor, and Random Forest Regressor. Performance was measured using **R² Score** and **Root Mean Squared Error (RMSE)** with cross-validation.

**The Random Forest Regressor achieved the best performance:**
- **Training R²:** 94.05%
- **Testing R²:** **77.58%**

The final optimized pipeline was serialized using **joblib** for deployment.

---

## 1. Introduction and Dataset Description

### 1.1 Introduction

Real estate valuation plays a pivotal role in urban planning, financial forecasting, and macroeconomic stability. Traditional appraisal methods often rely on subjective assessments or rigid linear assumptions, which fail to capture the complex socio-economic patterns in housing markets.

This project leverages **machine learning** to build an accurate, scalable, and objective pricing model for residential properties in Tehran.

### 1.2 Dataset Description

The dataset contains **3,479** residential property records from Tehran.

#### Features:

| Feature Name   | Data Type          | Description                                      |
|----------------|--------------------|--------------------------------------------------|
| `Area`         | Continuous Numeric | Built-up area in square meters                   |
| `Room`         | Discrete Integer   | Number of bedrooms                               |
| `Parking`      | Boolean            | Availability of parking space                    |
| `Warehouse`    | Boolean            | Availability of storage/warehouse                |
| `Elevator`     | Boolean            | Presence of elevator                             |
| `Address`      | Categorical Text   | Neighborhood/District (high cardinality)         |
| `Price`        | Continuous Numeric | Property price in **Tomans** (Target Variable)   |

#### Statistical Summary:

| Statistic      | Area          | Room   | Price (Tomans)     | Price (USD)    |
|----------------|---------------|--------|--------------------|----------------|
| count          | 3479          | 3479   | 3479               | 3479           |
| mean           | 8,744,000     | 2.08   | 5.36e9             | 178,634        |
| std            | 316,726,629   | 0.76   | 8.10e9             | 269,998        |
| min            | 30            | 0      | 3,600,000          | 120            |
| 25%            | 69            | 2      | 1.42e9             | 47,275         |
| 50%            | 90            | 2      | 2.90e9             | 96,667         |
| 75%            | 120           | 2      | 6.00e9             | 200,000        |
| max            | 16,160,000,000| 5      | 92.4e9             | 3,080,000      |

---

## 2. Exploratory Data Analysis and Data Preprocessing

### 2.1 Exploratory Data Analysis (EDA)

- Analyzed distribution of features and target variable
- Observed **right-skewed** distribution in `Price`
- Computed Pearson correlation matrix
- Found strong positive correlation between `Area` and `Price`

### 2.2 Data Preprocessing & Feature Engineering

1. **Text Cleaning**: Used Regex to clean `Area` column (removed commas and non-numeric characters)
2. **Missing Values**: Imputed using neighborhood-specific median values
3. **Outlier Handling**: Applied IQR method (1.5 × IQR) to remove extreme values
4. **Categorical Encoding**: One-Hot Encoding on `Address` (high-cardinality feature)

---

## 3. Methodology and Model Training

### 3.1 Models Evaluated

- **Multiple Linear Regression** (Baseline)
- **K-Neighbors Regressor**
- **XGBoost Regressor**
- **Random Forest Regressor** (Best Performer)
- Additional models: Ridge, Lasso, ElasticNet, Decision Tree

### 3.2 Evaluation Strategy

- Train/Test split
- Cross-validation
- Metrics: **R² Score** and **RMSE**

---

## 4. Results and Discussion

### Model Performance Comparison

| Model                    | Training R² | Testing R² | RMSE (Tomans)      |
|--------------------------|-------------|------------|--------------------|
| Linear Regression        | 60.40%      | 53.35%     | 6,357,973,680      |
| Ridge                    | 60.40%      | 53.36%     | 6,357,553,946      |
| Lasso                    | 60.40%      | 53.35%     | 6,357,953,518      |
| ElasticNet               | 59.51%      | 55.36%     | 6,219,838,183      |
| K-Neighbors Regressor    | 60.39%      | 63.99%     | 5,586,254,776      |
| Decision Tree            | 97.91%      | 71.12%     | 5,002,784,481      |
| XGBoost Regressor        | 90.19%      | 75.87%     | 4,573,220,543      |
| **Random Forest**        | **94.05%**  | **77.58%** | **4,407,752,061**  |

### Discussion

- Linear models performed poorly due to the **non-linear** nature of real estate pricing.
- Tree-based ensemble models significantly outperformed others.
- **Random Forest** provided the best balance between accuracy and generalization.
- Decision Tree suffered from overfitting (high training score, lower testing score).

---

## 5. Conclusion

This research successfully developed an end-to-end machine learning pipeline for Tehran housing price prediction. Through careful data preprocessing and ensemble modeling, we achieved strong predictive performance.

The **Random Forest Regressor** was selected as the final model and saved using `joblib` for future use and deployment.


