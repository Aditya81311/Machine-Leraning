import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit , cross_val_score
from sklearn.preprocessing import OrdinalEncoder , OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def pipelines(data_):

    # Transforming categorical columns
    housing_num = data_.drop("ocean_proximity",axis = 1)
    num_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),('std_scaler',StandardScaler()),])
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    num_attr = list(housing_num.columns)
    cat_attr = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
         ('num', num_pipeline, num_attr),
        ('cat', OneHotEncoder(), cat_attr)
    ])
    
    prepared = full_pipeline.fit_transform(data_)
    return prepared
    


data = pd.read_csv("/home/dark/Python/Meachine-Learning/ML_Projects/Super_vised/datasets/housing/housing.csv")
# print(data.head())
# print(data.describe())
# print(data.info())

# train_set,test_set = train_test_split(data, test_size=0.2,random_state= 42)  We will use stratifiedshuffelsplit becausse even splits

# print(train_set.head())

data["income_cat"] = pd.cut(data["median_income"],bins = [0.,1.5,3.0,4.5,6.,np.inf],labels = [1,2,3,4,5])

split = StratifiedShuffleSplit()

for train_index , test_index in split.split(data,data["income_cat"]):
    strat_train_set = data.iloc[train_index]
    strat_test_set = data.iloc[test_index]

train_set = strat_train_set.copy()

# seprating features and labels from data 
# train_cat = train_set["ocean_proximity"]
features = train_set.drop("median_house_value",axis = 1)
labels = train_set["median_house_value"]
    
features_prep = pipelines(features)

# test_set = strat_test_set.copy()
# test_prep = test_set.drop("median_house_value",axis = 1)
# test_labels = test_set["median_house_value"]

model = RandomForestRegressor()
# model  = LinearRegression()
# model = DecisionTreeRegressor()
model.fit(features_prep,labels)

predected = model.predict(features_prep)
mse = mean_squared_error(labels,predected)
rmse = np.sqrt(mse)

scores = cross_val_score(model,features_prep,labels,scoring="neg_mean_squared_error")
fore_rmse_scores = np.sqrt(-scores)

print(f"Mean Squred Error = {mse} \n Root Mean Squred Error = {rmse} \n Cross validation Score  = {fore_rmse_scores.mean()}\n {fore_rmse_scores.std()}")

import joblib
joblib.dump(model,"prediction_model.joblib")