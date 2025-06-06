# Credit-card-churn-prediction
A machine learning project to predict which credit card customers are likely to churn.

This project analyzes a Kaggle dataset to build a model that identifies at-risk customers. The goal is to provide data-driven insights that can help a business develop proactive retention strategies, saving costs and protecting revenue.      




1. The Business Problem
   
The cost of acquiring a new customer is often significantly higher than the cost of retaining an existing one. In the competitive financial services industry, high churn rates can severely impact profitability and market share. Therefore, understanding the reasons behind churn and identifying at-risk customers early are paramount.



This project aims to answer two fundamental questions:

Who is likely to churn? (Prediction)  

Why are they likely to churn? (Interpretation)



	 
2. Exploratory Data Analysis (EDA)

Initial analysis revealed key patterns. Visualizations showed that customers who churned generally had significantly lower transaction counts and amounts, a lower revolving balance, and a higher number of contacts with the bank in the preceding months. The class imbalance was also confirmed during this phase.




3. To prepare the data for modeling, a ColumnTransformer was used to apply two main steps:
   

StandardScaler: Applied to all numerical features to normalize their scale, ensuring that no single feature dominates due to its large values.

OneHotEncoder: Applied to all categorical features to convert them into a numerical format that the model can process.  




4. Handling Class Imbalance (SMOTE)

Given that only 16% of customers churned, the dataset is imbalanced. To prevent the model from becoming biased towards the majority class (non-churners), the SMOTE was applied. This was done carefully inside a modeling pipeline to ensure that SMOTE was only applied to the training data and leave the test data untouched to provide an unbiased evaluation of real-world performance.




5. Model Training


Logistic Regression: A linear model used as a baseline for its simplicity and interpretability.

Random Forest: A powerful ensemble model known for its high performance and ability to capture complex non-linear relationships.  




6. Results & Evaluation

Models were evaluated on the unseen test set. The primary metric for success was Recall for the churn class, as the business cost of failing to identify a churner (a False Negative) is much higher than mistakenly flagging a loyal customer.



![Screenshot 2025-06-06 143218](https://github.com/user-attachments/assets/9b1ed1b9-ccd3-4458-9499-6854b8ff2798)  


The Random Forest model demonstrated superior performance across all key metrics. A Recall of 0.825 means the model can successfully identify 82.5% of all customers who are genuinely at risk of leaving, providing the business with a large and accurate pool of customers to target with retention efforts.



![Screenshot 2025-06-06 144837](https://github.com/user-attachments/assets/35dcf373-8f9b-4b2f-ab60-1f644570407c)



By analyzing the feature importances from the Random Forest model, we were able to determine why customers churn.	


7. Conclusion
This project successfully demonstrates the power of machine learning in solving a critical business problem. We have built a robust Random Forest model that can effectively predict customer churn with a high degree of accuracy and recall.

More importantly, the analysis moves beyond simple prediction to provide deep, actionable insights. The findings clearly show that declining customer engagement, primarily visible through transaction data, is the most powerful driver of churn. This knowledge allows a business to focus its retention efforts not on static demographic data, but on monitoring dynamic customer behavior and intervening at the first sign of disengagement.


