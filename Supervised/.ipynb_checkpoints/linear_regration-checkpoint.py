import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets , linear_model
from sklearn.metrics import mean_squared_error as mse
dbt = datasets.load_diabetes()
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])


dbt_X = dbt.data[:,np.newaxis,2]

# dbt_X = dbt.data
dbt_X_train = dbt_X[:-30]
dbt_X_test = dbt_X[-20:]

dbt_y_train = dbt.target[:-30]
dbt_y_test = dbt.target[-20:]

model = linear_model.LinearRegression()
model.fit(dbt_X_train,dbt_y_train)
model.coef_
model.intercept_
dbt_y_predict = model.predict(dbt_X_test)

print(mse(dbt_y_test,dbt_y_predict))

print(model.coef_,model.intercept_)

plt.scatter(dbt_X_test,dbt_y_test)
plt.plot(dbt_X_test,dbt_y_predict)
plt.show()