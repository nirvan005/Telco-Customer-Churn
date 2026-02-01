# Telco Customer Churn Prediction

A machine learning project to predict customer churn in a telecommunications company using **Random Forest** and **Logistic Regression** models.

---

## 📊 Dataset

**Source:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) by Blastchar  
Focused customer retention programs

**Dataset Details:**

- **Total Records:** 7,043 customers
- **Features:** 21 columns including demographics, services subscribed, account information, and churn status
- **Target Variable:** `Churn` (Yes/No)

---

## 📁 Directory Structure

```
Telco Customer Churn/
│
├── 01_Telco_Customer_Churn_Random_Forest.ipynb   # Main Jupyter notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Dataset
├── output.html                                    # Data profiling report
├── churn_predict_rf.pkl                           # Trained Random Forest model (generated)
├── churn_predict_lr.pkl                           # Trained Logistic Regression model (generated)
├── preprocess_telco.pkl                           # Preprocessing pipeline (generated)
└── README.md                                      # Project documentation
```

---

## 🔍 Project Overview

This project analyzes customer churn patterns and builds predictive models to identify customers likely to churn. The analysis includes:

1. **Data Cleaning & Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering & Transformation**
4. **Model Training & Hyperparameter Tuning**
5. **Model Evaluation & Comparison**

---

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - Machine learning models and preprocessing
- **ydata-profiling** - Automated EDA report generation

---

## 🧹 Data Preprocessing

### Missing Values

- **TotalCharges** column contained 11 missing values (represented as whitespace)
- Since missing values represented only 0.16% of the dataset, they were safely dropped

### Feature Transformation

- **TotalCharges:** Applied Yeo-Johnson Power Transformation to reduce skewness
- **MonthlyCharges:** No transformation needed (transformation didn't improve skewness)

### Encoding

- **Target Variable (Churn):** Label encoded (Yes=1, No=0)
- **Categorical Features:** One-Hot encoded with `drop='first'` to avoid multicollinearity
- **Dropped Features:**
  - `customerID` - Unique identifier with no predictive value
  - `PhoneService` - Information already captured in `MultipleLines` feature

---

## 📈 Exploratory Data Analysis - Key Findings

### 1. **Numerical Features Impact on Churn**

#### Monthly Charges

- **Customers paying higher monthly charges are more likely to churn**
- Higher monthly costs may indicate dissatisfaction with value-for-money

#### Total Charges

- **Customers with higher total charges tend to stay**
- Higher total charges indicate longer customer relationship and loyalty

#### Tenure

- **Customers with longer tenure are more likely to stay**
- Tenure is inversely related to churn probability

### 2. **Categorical Features Impact on Churn**

Key categorical features analyzed for their effect on churn rates:

- **Gender:** Minimal impact on churn
- **SeniorCitizen:** Significant impact - senior citizens have higher churn rates
- **Partner:** Customers without partners show higher churn rates
- **Dependents:** Customers without dependents are more likely to churn
- **InternetService:** Fiber optic customers show significantly higher churn rates
- **Contract Type:** Month-to-month contracts have the highest churn rates
- **PaymentMethod:** Electronic check users show higher churn tendency

### 3. **Feature Correlations**

- TotalCharges and tenure are positively correlated
- Monthly charges show moderate positive correlation with churn
- Contract type and tenure are strongly related to customer retention

---

## 🤖 Machine Learning Models

### Model 1: Random Forest Classifier

**Initial Model Performance:**

- Showed signs of **overfitting** (high train accuracy, lower test accuracy)

**Hyperparameter Tuning Approaches:**

1. **Manual Tuning** - Testing different parameter combinations
2. **GridSearchCV** - Exhaustive search over specified parameter values
3. **RandomizedSearchCV** - Efficient search over parameter distributions

**Final Random Forest Model:**

```python
RandomForestClassifier(
    n_estimators=250,
    max_depth=9,
    max_features=0.2,
    max_samples=0.7,
    min_samples_split=5,
    max_leaf_nodes=100,
    n_jobs=4
)
```

**Performance Metrics:**

- **Cross-validated Accuracy:** ~79-80%
- **Precision:** High precision for churn prediction
- **F1 Score:** Balanced performance across classes

### Model 2: Logistic Regression

**Final Logistic Regression Model:**

```python
LogisticRegression(
    max_iter=150,
    n_jobs=4
)
```

**Performance:**

- Achieved similar performance to Random Forest (~79% accuracy)

### 🎯 Key Insight:

**Logistic Regression performs almost as well as Random Forest, indicating that the data is approximately linear.** This suggests that the relationship between features and churn can be captured well with a simple linear decision boundary.

---

## 🔄 Model Pipeline

The final models use a complete preprocessing pipeline:

```python
Pipeline([
    ('preprocess', ColumnTransformer([
        ('yeo_trf', PowerTransformer(), ['TotalCharges']),
        ('drop_unwanted', 'drop', ['PhoneService', 'customerID']),
        ('ohe', OneHotEncoder(drop='first', dtype='int8'), [categorical_features])
    ], remainder='passthrough')),
    ('model', <RandomForest or LogisticRegression>)
])
```

---

## 💾 Model Persistence

Trained models are saved using `pickle` for deployment:

- `churn_predict_rf.pkl` - Random Forest model
- `churn_predict_lr.pkl` - Logistic Regression model
- `preprocess_telco.pkl` - Preprocessing pipeline

---

## 🚀 Usage

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ydata-profiling
```

### 2. Run the Notebook

```bash
jupyter notebook 01_Telco_Customer_Churn_Random_Forest.ipynb
```

### 3. Make Predictions

```python
import pickle

# Load model
model = pickle.load(open('churn_predict_rf.pkl', 'rb'))

# Predict for new customer
prediction = model.predict(customer_data)
```

---

## 📊 Model Evaluation

Both models were evaluated using:

- **10-fold Cross-Validation** (for robust performance estimation)
- **Accuracy Score**
- **Precision Score**
- **F1 Score (Macro)**
- **Train/Test Accuracy** (to detect overfitting)

---

## 🎓 Insights & Conclusions

1. **Data is approximately linear** - Logistic Regression performs competitively with Random Forest
2. **Tenure is the strongest predictor** - Longer customer relationships reduce churn
3. **Contract type matters** - Month-to-month contracts are high-risk
4. **Service quality perception** - High monthly charges without proportional value leads to churn
5. **Customer profile** - Senior citizens without partners/dependents are at higher risk
6. **Internet service type** - Fiber optic customers show higher churn (possibly due to competition or pricing)

---

## 🔮 Business Recommendations

1. **Focus on early retention** - First few months are critical
2. **Incentivize long-term contracts** - Reduce month-to-month churn
3. **Review fiber optic pricing** - Address competitive pressure
4. **Target senior citizens** - Special retention programs
5. **Value-based pricing** - Ensure monthly charges align with perceived value

---

## 📝 Notes

- **Random State:** Set to 40 for reproducibility
- **Train-Test Split:** 80-20 split
- **Data Transformation:** Skewness transformation was applied primarily to enable fair comparison between Random Forest and Logistic Regression, as Random Forest doesn't strictly require normally distributed features

---

## 👤 Author

**Nirvan**

---

## 📄 License

This project is for educational purposes. Dataset credit: Blastchar on Kaggle.

---

## 🙏 Acknowledgments

- Dataset source: [Kaggle - Telco Customer Churn by Blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Focus: Customer retention programs
