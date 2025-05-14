# 🚙 SUV Purchase Prediction Using K-Nearest Neighbors (KNN)

This project uses the **K-Nearest Neighbors (KNN)** machine learning algorithm to predict whether a user is likely to purchase an SUV based on features such as age and estimated salary.

---

## 🎯 Objective

To build a machine learning model that can predict whether a person will purchase an SUV based on their demographic details using the **KNN classification algorithm**.

---

## 🧾 Dataset Description

The dataset typically includes:

* `User ID` (not used in modeling)
* `Gender`
* `Age`
* `EstimatedSalary`
* `Purchased` (Target: 0 = No, 1 = Yes)

> Example source: [Social\_Network\_Ads.csv](https://www.kaggle.com/datasets/rakeshrau/social-network-ads)

---

## 🛠️ Tools & Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn (for visualization)

---

## 🔍 Project Workflow

### 1. 📊 Data Preprocessing

* Load the dataset
* Handle missing or irrelevant columns
* Encode categorical variables (e.g., Gender)
* Feature scaling using StandardScaler or MinMaxScaler
* Split the dataset into training and testing sets

### 2. 🧠 Apply KNN Classifier

* Train the **K-Nearest Neighbors** classifier
* Use `n_neighbors` parameter tuning
* Predict results on test data

### 3. 📈 Evaluate Model

Use the following evaluation metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Confusion Matrix**

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
```

### 4. ⚙️ Hyperparameter Tuning

Improve performance by:

* Tuning `n_neighbors`
* Trying different distance metrics (Euclidean, Manhattan)
* Using GridSearchCV or cross-validation

---

## ✅ Results

Model outputs include:

* Classification report
* Confusion matrix plot
* Accuracy and F1 Score

---

## 🔧 Future Improvements

* Compare with other classifiers (Logistic Regression, SVM, Decision Trees)
* Deploy as a web app using Streamlit
* Collect more features (device usage, past purchases, etc.)


