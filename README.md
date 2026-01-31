# Titanic-Survival-Prediction-

Objective
The goal of this project is to predict passenger survival on the Titanic using the training dataset and submit predictions for the test dataset. A Logistic Regression model was used due to its interpretability and suitability for binary classification.

Data Cleaning and Preparation:
 The training dataset (train.csv) was loaded using pandas.
Columns that are not predictive or have too many missing values were dropped:
Cabin (too many missing values)
Ticket, Name, PassengerId (irrelevant for prediction)
The remaining features were separated into:
Numerical features: Age, Fare, SibSp, Parch
Categorical features: Sex, Embarked

Key principle:
  All missing value handling, scaling, and encoding are done inside the pipeline to ensure consistency between training and test data and avoid data leakage.
  
Pipeline and Model
  A Scikit-learn Pipeline was built to automate preprocessing and model training.
  
Numerical pipeline:
 SimpleImputer(strategy="median") → fills missing numerical values with the median (calculated from training data)
 StandardScaler() → scales numerical features to zero mean and unit variance

Categorical pipeline:
   SimpleImputer(strategy="most_frequent") → fills missing categorical values with the most frequent category
OneHotEncoder(handle_unknown="ignore") → converts categorical variables into binary columns and handles unseen categories in the test set.

ColumnTransformer: Combines numerical and categorical pipelines into a single feature matrix.

Logistic Regression:
 Maximum iterations set to 1000 for convergence. Trained on preprocessed features
 
Model Training and Evaluation:
 The dataset was split into 80% training and 20% validation.
Model accuracy was evaluated using:
Accuracy (score): Overall correctness on the validation set
Confusion matrix & classification report: Optional metrics for precision, recall, and F1-score
Feature importance was extracted from the Logistic Regression coefficients:
Positive coefficient → increases probability of survival
Negative coefficient → decreases probability of survival
Feature importance was visualized in feature_importance.png for interpretability.

Test Data Predictions
  Test dataset (test.csv) was loaded.
Only structural cleaning was applied (dropped Cabin) — all statistical cleaning is handled by the pipeline.
Predictions were generated using model.predict() and combined with PassengerId to form the submission file:
Output file:
Palazzo's Titanic_survival prediction.csv

Columns:
PassengerId
Survived

Notes:

Pipeline ensures consistency:
  The pipeline automatically applies the same preprocessing to the test set as it learned from training, preventing inconsistencies or data leakage.
  
Feature Importance Insights:
 Sex_female and Pclass_1 likely have the strongest positive impact on survival
Pclass_3 and higher age negatively influence survival
(See feature_importance.png for full ranking).

Reproducibility:
  Random seed (random_state=42) ensures consistent train-test splits
Logistic Regression’s max_iter=1000 ensures convergence.

Conclusion;
  The submitted predictions reflect the model’s understanding of passenger survival based on patterns in the training dataset. This approach:
Uses a robust pipeline to prevent data leakage
Enables interpretability via coefficients.
  This methodology is scalable, reproducible, and aligns with standard machine learning best practices.
