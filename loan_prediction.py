import pandas as pd
import numpy as np
import joblib
from tkinter import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv('Loan_Prediction_train.csv')

# Displaying Top 5 rows of the dataset
data.head()
# Displaying Last 5 rows of teh dataset
data.tail()

# Finding the shape of the dataset that is the number of rows and the number of columns
data.shape # shape is an attribute of the panads not a method
print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])

# Gathering information about the dataset
data.info()

# Handling Null values in the dataset
data.isnull().sum()

# Changing in percentages
data.isnull().sum()*100/len(data)

# Handling missing values by droping the rows with less than 5%
data = data.drop('Loan_ID',axis=1)

data.head(1)

columns = ['Gender','Dependents','LoanAmount','Loan_Amount_Term']

data = data.dropna(subset=columns)

data.isnull().sum()*100 / len(data)

data['Self_Employed'].mode()[0]

data['Self_Employed'] =data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])

data.isnull().sum()*100 / len(data)

data['Gender'].unique()

data['Self_Employed'].unique()

data['Credit_History'].mode()[0]

data['Credit_History'] =data['Credit_History'].fillna(data['Credit_History'].mode()[0])

data.isnull().sum()*100 / len(data)

# Handling Categorical Columns
data.sample(5)

data['Dependents'] =data['Dependents'].replace(to_replace="3+",value='4')

data['Dependents'].unique()

data['Loan_Status'].unique()

data['Gender'] = data['Gender'].map({'Male':1,'Female':0}).astype('int')
data['Married'] = data['Married'].map({'Yes':1,'No':0}).astype('int')
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
data['Property_Area'] = data['Property_Area'].map({'Rural':0,'Semiurban':2,'Urban':1}).astype('int')
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0}).astype('int')

data.head()

# Storing Feature Matrix in X and Response(Target) in vector y
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']

y

# Feature Scaling
data.head()

cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']

st = StandardScaler()
X[cols]=st.fit_transform(X[cols])

X

# Splitting the Dataset into the training Set and Test Set
# To apply K-Fold Cross Validation
model_df={}
def model_val(model,X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.20, random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print(f"{model} accuracy is {accuracy_score(y_test,y_pred)}")

    score = cross_val_score(model,X,y,cv=5)
    print(f"{model} Avg cross val score is {np.mean(score)}")
    model_df[model]=round(np.mean(score)*100,2)

# Using Logistic Regression
model = LogisticRegression()
model_val(model,X,y)

# Support Vector Classifier
model = svm.SVC()
model_val(model,X,y)

# Using Decision Tree Classifier
model = DecisionTreeClassifier()
model_val(model,X,y)

# Using Random Forest Classifier
model =RandomForestClassifier()
model_val(model,X,y)

# Using Gradient Boosting Classifier
model =GradientBoostingClassifier()
model_val(model,X,y)

# Using Hyperparameter Tuning
# Logistic Regression
log_reg_grid={"C":np.logspace(-4,4,20),
             "solver":['liblinear']}

rs_log_reg=RandomizedSearchCV(LogisticRegression(),
                   param_distributions=log_reg_grid,
                  n_iter=20,cv=5,verbose=True)

rs_log_reg.fit(X,y)

rs_log_reg.best_score_

rs_log_reg.best_params_

# Again using SVC
svc_grid = {'C':[0.25,0.50,0.75,1],"kernel":["linear"]}

rs_svc=RandomizedSearchCV(svm.SVC(),
                  param_distributions=svc_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)

rs_svc.fit(X,y)

rs_svc.best_score_

rs_svc.best_params_

# Random Forest Classifier
RandomForestClassifier()
rf_grid={'n_estimators':np.arange(10,1000,10),
  'max_features':['auto','sqrt'],
 'max_depth':[None,3,5,10,20,30],
 'min_samples_split':[2,5,20,50,100],
 'min_samples_leaf':[1,2,5,10]
 }

rs_rf=RandomizedSearchCV(RandomForestClassifier(),
                  param_distributions=rf_grid,
                   cv=5,
                   n_iter=20,
                  verbose=True)

rs_rf.fit(X,y)

rs_rf.best_score_

rs_rf.best_params_

# Saving the model
X = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']

rf = RandomForestClassifier(n_estimators=270,
 min_samples_split=5,
 min_samples_leaf=5,
 max_features='sqrt',
 max_depth=5)

rf.fit(X,y)

joblib.dump(rf,'loan_status_predict')

model = joblib.load('loan_status_predict')

df = pd.DataFrame({
    'Gender':1,
    'Married':1,
    'Dependents':2,
    'Education':0,
    'Self_Employed':0,
    'ApplicantIncome':2889,
    'CoapplicantIncome':0.0,
    'LoanAmount':45,
    'Loan_Amount_Term':180,
    'Credit_History':0,
    'Property_Area':1
},index=[0])

df

result = model.predict(df)

if result==1:
    print("Loan Approved")
else:
    print("Loan Not Approved")

# GUI

def show_entry():
    # Collect input data from the user
    p1 = float(input("Enter Gender [1:Male, 0:Female]: "))
    p2 = float(input("Enter Married [1:Yes, 0:No]: "))
    p3 = float(input("Enter Dependents [1,2,3,4]: "))
    p4 = float(input("Enter Education [1:Graduate, 0:Not Graduate]: "))
    p5 = float(input("Enter Self_Employed [1:Yes, 0:No]: "))
    p6 = float(input("Enter ApplicantIncome: "))
    p7 = float(input("Enter CoapplicantIncome: "))
    p8 = float(input("Enter LoanAmount: "))
    p9 = float(input("Enter Loan_Amount_Term: "))
    p10 = float(input("Enter Credit_History [1:Yes, 0:No]: "))
    p11 = float(input("Enter Property_Area [1:Urban, 2:Semiurban, 3:Rural]: "))

    # Load the pre-trained model
    model = joblib.load('loan_status_predict')

    # Create a DataFrame with the input data
    df = pd.DataFrame({
        'Gender': p1,
        'Married': p2,
        'Dependents': p3,
        'Education': p4,
        'Self_Employed': p5,
        'ApplicantIncome': p6,
        'CoapplicantIncome': p7,
        'LoanAmount': p8,
        'Loan_Amount_Term': p9,
        'Credit_History': p10,
        'Property_Area': p11
    }, index=[0])

    # Make a prediction
    result = model.predict(df)

    # Display the result
    if result == 1:
        print("Loan approved")
    else:
        print("Loan Not Approved")

# Call the function to execute it
show_entry()