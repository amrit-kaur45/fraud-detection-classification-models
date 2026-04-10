# Fraud Detection using Machine Learning Models

## Overview  
This project focuses on detecting fraudulent credit card transactions using multiple machine learning classification models. It compares model performance using standard evaluation metrics. 

---

## Dataset  
The dataset contains anonymized transaction features and a target variable `Class`:  
- `0` → Normal transaction  
- `1` → Fraudulent transaction  
- Highly imbalanced dataset
---

## Workflow  

### 1. Data Preprocessing  
- Checked for missing values  
- Train-test split with stratification to maintain class distribution  

```python
from sklearn.model_selection import train_test_split

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 2. Models Implemented
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree
- Random Forest
- XGBoost

### 3. Random Forest (Best Model)

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
```

### 4. Evaluation Metrics

```python
from sklearn.metrics import f1_score, confusion_matrix, classification_report

print("F1 Score:", f1_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
```

### 5. ROC Curve

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

probs = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, probs)

plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

print("AUC:", auc(fpr, tpr))
```

## Key Insights
- Dataset is highly imbalanced
- Tree-based ensemble models performed better
- Random Forest achieved the highest F1-score
- Ensemble methods captured complex patterns effectively
