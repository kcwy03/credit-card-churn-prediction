#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -U imbalanced-learn


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# In[28]:


# Load and prepare data

df = pd.read_csv('BankChurners.csv')
print("Dataset loaded successfully.")


# In[4]:


df.info()


# In[5]:


print(df.head())


# In[6]:


print(df.describe())


# In[7]:


# 檢查缺失值
print(df.isnull().sum())


# In[8]:


# Remove unnecessary columns
df = df.drop(columns=[
    'CLIENTNUM',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
])


# In[10]:


print({df.shape})


# In[11]:


# Convert target variable to numeric
df['Attrition_Flag'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)


# In[12]:


# (EDA)
print("\nChurn Distribution:")
print(df['Attrition_Flag'].value_counts())


# In[14]:


# Visualizations
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(x='Attrition_Flag', data=df, palette='viridis')
plt.title('Distribution of Customer Churn')
plt.show()


# In[15]:


plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Total_Trans_Ct', hue='Attrition_Flag', kde=True)
plt.title('Total Transaction Count by Churn Status')
plt.show()


# In[16]:


# Data Preprocessing and Splitting
# Define features (X) and target (y)
X = df.drop('Attrition_Flag', axis=1)
y = df['Attrition_Flag']


# In[17]:


# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()


# In[20]:


# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
    ])


# In[21]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# In[22]:


# Model Training
# Create pipelines with SMOTE and classifier
log_reg_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
])

rf_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1))
])


# In[23]:


# Train models
print("\nTraining models...")
log_reg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)
print("Model training complete.")


# In[24]:


# Model Evaluation
models = {
    "Logistic Regression": log_reg_pipeline,
    "Random Forest": rf_pipeline
}

print("\n--- Model Evaluation Results ---")
for name, model in models.items():
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Existing (0)', 'Attrited (1)']))


# In[25]:


# Feature Importance Analysis using Random Forest
print("\n--- Random Forest Feature Importances (Top 15) ---")

try:
    ohe_feature_names = rf_pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
except AttributeError: 
    ohe_feature_names = []
    for i, col in enumerate(categorical_features):
        categories = rf_pipeline.named_steps['preprocessor'].transformers_[1][1].categories_[i]
        ohe_feature_names.extend([f"{col}_{cat}" for cat in categories])

all_feature_names = numerical_features + list(ohe_feature_names)
classifier = rf_pipeline.named_steps['classifier']


# In[26]:


# Create a DataFrame for feature importances
importances = classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df.head(15))


# In[27]:


# Visualize feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('Top 15 Feature Importances from Random Forest')
plt.tight_layout()
plt.show()


# In[ ]:




