# Mental_Fitness_Tracker
The Mental Fitness Tracker analyzes the data and uses various regression techniques to predict mental fitness levels based on the parameters. The users can enter their particular disorder levels, which in turn predict their mental fitness in percentage. 
# Code
"""Mental_fitness_tracker.ipynb"""

1.Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

2.Data Selecection
df1 = pd.read_csv('/content/drive/MyDrive/mental-and-substance-use-as-share-of-disease.csv')
df2 = pd.read_csv('/content/drive/MyDrive/prevalence-by-mental-and-substance-use-disorder.csv')

df1.head()

df2.head()

data = df2.merge(df1)

data.head()

data.tail()

3.Data Preprocessing

data.isnull().sum()

data.drop('Code',axis=1,inplace=True)

data.isna().all()

data.shape

data.describe()

data.info()

data.duplicated()

data.columns = ['Country','Year', 'Schizophrenia',
                'Bipolar disorder','Eating disorder','Anxiety disorder','Drug use disorder'
                ,'Depressive disorder','Alcohol use disorder','mental disorder']

data.columns

data.corr()

4.Exploratory Data Analysis

plt.figure(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,cmap='mako')
plt.plot()

sns.pairplot(data,hue='mental disorder')

sns.jointplot(x='Schizophrenia',y='mental disorder',data=data)

sns.distplot(data['mental disorder'],hist=True, bins=10)

fig = px.pie(data, values='mental disorder', names='Year')
fig.show()

import plotly.express as px
fig = px.area(data, x="Year", y="mental disorder", color='Country',line_group="Country")
fig.show()

px.bar(data,x='Year',y='mental disorder')

5.Data Conversion
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
data.Country=l.fit_transform(data.Country)
print(data)

X= data.drop('mental disorder',axis=1)
y= data['mental disorder']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=2)

6.Model Selection and Evaluation

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lg = LinearRegression()
lg.fit(X_train,y_train)

#evaluating on training data
y_pred = lg.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2score = r2_score(y_train, y_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2score))
print("\n")

#evaluating on testing set
y_pred = lg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train,y_train)

#evaluating on training data
y_pred = knn_model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2score = r2_score(y_train, y_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2score))
print("\n")

#evaluating on testing set
y_pred = knn_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

from xgboost import XGBRegressor
X = XGBRegressor()
X.fit(X_train,y_train)

#evaluating on training data
y_pred = X.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2score = r2_score(y_train, y_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2score))
print("\n")

#evaluating on testing set
y_pred = X.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

from sklearn.tree import DecisionTreeRegressor
D = DecisionTreeRegressor()
D.fit(X_train,y_train)

#evaluating on training data
y_pred = D.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2score = r2_score(y_train, y_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2score))
print("\n")

#evaluating on testing set
y_pred = D.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

from sklearn.ensemble import RandomForestRegressor
R = RandomForestRegressor()
R.fit(X_train,y_train)

#evaluating on training data
y_pred = R.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
rmse = (np.sqrt(mean_squared_error(y_train, y_pred)))
r2score = r2_score(y_train, y_pred)

print("The model performance for training set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2score))
print("\n")

#evaluating on testing set
y_pred = R.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = r2_score(y_test, y_pred)

print("The model performance for testing set")
print("--------------------------------------")
print('MSE is {}'.format(mse))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

"""# User Input"""

print("Welcome to Mental Fitness Tracker!\nFill the detail to check your mental fitness!")
country= l.fit_transform([input('Enter Your country Name:')])
year= int(input("Enter the Year:"))
schi= (float(input("Enter your Schizophrenia rate in %:")))
bipo_dis= (float(input("Enter your Bipolar disorder rate in %:")))
eat_dis= (float(input("Enter your Eating disorder rate in %:")))
anx= (float(input("Enter your Anxiety rate in %:")))
drug_use= (float(input("Enter your Drug Usage rate in per year %:")))
depr= (float(input("Enter your Depression rate in %:")))
alch= (float(input("Enter your Alcohol Consuming rate per year in %:")))


prediction= R.predict([[country,year,schi,bipo_dis,eat_dis,anx,drug_use,depr,alch]])
print("Your Mental Fitness is {}%".format(prediction*10))


