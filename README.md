# Loan Prediction Status Application

This Loan Prediction Status Application is a machine learning project that predicts the approval status of a loan application based on various input parameters. The application is built using Python and various libraries such as pandas, scikit-learn, and joblib. It includes a GUI for user interaction and allows users to input their data to get instant loan approval predictions.
The goal of this project is to automate the loan approval process by predicting whether a loan application will be approved or not. The prediction is made using various machine learning models trained on historical loan data.

Features
1. Data Preprocessing: Handles missing values, encodes categorical variables, and scales numerical features.
2. Model Training: Includes Logistic Regression, Support Vector Classifier, Decision Tree, Random Forest, and Gradient Boosting models.
3. Hyperparameter Tuning: Uses RandomizedSearchCV for tuning hyperparameters of the Logistic Regression, SVC, and Random Forest models.
4. Prediction: Predicts loan approval status based on user input.
5. GUI Implementation: Allows users to input their data and get instant predictions using a simple command-line interface.

Dataset
The dataset used for training the models is Loan_Prediction_train.csv. It contains various attributes of loan applicants, such as:
1. Gender
2. Married status
3. Number of dependents
4. Education level
5. Employment status
6. Applicant's income
7. Co-applicant's income
8. Loan amount
9. Loan amount term
10. Credit history
11. Property area
12. Loan status (target variable)

Preprocessing
Data preprocessing steps include:
1. Handling missing values by imputing or dropping them.
2. Encoding categorical variables using mapping.
3. Scaling numerical features using StandardScaler.

Modeling
The following models were trained and evaluated:
1. Logistic Regression
2. Support Vector Classifier (SVC)
3. Decision Tree Classifier
4. Random Forest Classifier
5. Gradient Boosting Classifier

Each model's performance is evaluated using accuracy and cross-validation scores.

Hyperparameter Tuning
RandomizedSearchCV is used to perform hyperparameter tuning for:
1. Logistic Regression: Tuning the regularization parameter C.
2. Support Vector Classifier: Tuning the regularization parameter C and the kernel.
3. Random Forest Classifier: Tuning the number of estimators, maximum depth, minimum samples split, and minimum samples leaf.

GUI Implementation
A simple command-line GUI is implemented to allow users to input their data and get a prediction. The model trained with the best hyperparameters is used for prediction.
