#Impoting libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn .metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

#Reading data
dataset = pd.read_csv('../data/external/FuelConsumption.csv')
#print(dataset.head())
#print(dataset.describe())

#Exploring data
ds = dataset[['MAKE','MODEL','VEHICLECLASS','ENGINESIZE','CYLINDERS','TRANSMISSION','FUELTYPE','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']]
print(ds.head())
ds.hist()
plt.draw()

#Plotting fuel consumption_comb vs CO2 emissions
fig2 = plt.figure()
plt.scatter(ds.FUELCONSUMPTION_COMB, ds.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("EMISSIONS")
plt.draw()

#Plotting engine size vs CO2 emissions
fig3 = plt.figure()
plt.scatter(ds.ENGINESIZE, ds.CO2EMISSIONS, color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")
plt.draw()

#Plotting cylinders size vs CO2 emissions
fig4 = plt.figure()
plt.scatter(ds.CYLINDERS, ds.CO2EMISSIONS, color='blue')
plt.xlabel("CYLINDERS")
plt.ylabel("EMISSIONS")
plt.draw()


#Splitting data into training set and test set
msk = np.random.rand(len(ds)) < 0.8
train = ds[msk]
test = ds[~msk]

## Simple Linear Regression

#Modeling linear regression
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print('Coeffs: ', regr.coef_)
print('Intercept: ', regr.intercept_)

#regr.predict()

#Plotting the data
fig5 = plt.figure()
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

#Evaluating the model
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


## Multiple Linear Regression

#Modeling
regr_m = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr_m.fit (x, y)

print ('Coefficients: ', regr_m.coef_)

#Predicting the results
y_hat = regr_m.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x_ = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y_ = np.asanyarray(test[['CO2EMISSIONS']])

print("Mean Squared Error (MSE): %.2f" % np.mean((y_hat - y_) ** 2))
print('Variance score; %.2f' % regr_m.score(x_, y_))


##Polynomial Regression

#Modeling
poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
regr_p = linear_model.LinearRegression()
train_y_ = regr_p.fit(train_x_poly, train_y)
print('Coeff.: ', regr_p.coef_)
print('Inter.: ', regr_p.intercept_)

#Plotting polynomial regresion
fig6 = plt.figure()
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = regr_p.intercept_[0]+ regr_p.coef_[0][1]*XX+ regr_p.coef_[0][2]*np.power(XX, 2) + regr_p.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

#Evaluation
test_x_poly = poly.transform(test_x)
test_y_ = regr_p.predict(test_x_poly)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y,test_y_ ))


#Showing th data
plt.show()