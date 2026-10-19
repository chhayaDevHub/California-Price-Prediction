from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import joblib

# Load dataset
data = fetch_california_housing()

# inpute features and target variable
X=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target, name="target")

print(f"total records:{X.shape[0]}")

# split dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# create models 
models={'Linear Regression':LinearRegression()
        ,
        'Decision Tree':DecisionTreeRegressor
        (random_state=42),

        'Random Forest':RandomForestRegressor
        (n_estimators=100,
         random_state=42)
         }


# train and evaluate models
for model_name,model in models.items():

    print("\n----------")
    print(model_name)

    #train model
    model.fit(X_train,y_train)

    # prediction on test set
    y_pred=model.predict(X_test)

    MAE=mean_absolute_error(y_test,y_pred)
    MSE=mean_squared_error(y_test,y_pred)
    R2=r2_score(y_test,y_pred)

    print(f"average error:{MAE *100000:,.0f}")
    print(f"mean squared error:{MSE *100000:,.0f}")
    print(f"R2 score:{R2 *100:,.0f}%")

# save the Random Forest Model 
joblib.dump(model,"california_housing_model.joblib")
joblib.dump(list(X.columns),"california_housing_features.joblib")

print("model and  features saved successfully")