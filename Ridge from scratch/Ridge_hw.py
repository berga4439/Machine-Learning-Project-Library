import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import RidgeLib as rl

df = pd.read_csv("austinHousingData.csv")

df.dropna(inplace=True)

df = df.drop(df["lotSizeSqFt"].idxmax())
df = df.drop(df["livingAreaSqFt"].idxmax())
df = df.drop(df["latestPrice"].idxmax())

y = df["latestPrice"]
x = df[["propertyTaxRate", "parkingSpaces","yearBuilt","numPriceChanges","latest_salemonth", "latest_saleyear", "numOfAppliances", "numOfParkingFeatures", "numOfPatioAndPorchFeatures", "numOfSecurityFeatures", "numOfWaterfrontFeatures", "numOfWindowFeatures", "numOfCommunityFeatures", "lotSizeSqFt", "livingAreaSqFt", "numOfPrimarySchools", "numOfElementarySchools", "numOfMiddleSchools", "numOfHighSchools", "avgSchoolDistance", "avgSchoolRating", "avgSchoolSize", "MedianStudentsPerTeacher", "numOfBathrooms", "numOfBedrooms", "numOfStories"]]

x_std = (x - np.average(x, axis=0))/np.std(x, axis=0)
y_std = (y - np.average(y))/np.std(y)

x_train, x_test, y_train, y_test = train_test_split(x_std, y_std, test_size=0.9)



linModel = LinearRegression().fit(x_train, y_train)
ridgeModel = Ridge(alpha=0.05).fit(x_train, y_train)
myRidgeModel = rl.RidgeRegression(lam=0.05).fit(x_train, y_train)


print("Linear (train):", linModel.score(x_train, y_train))
print("Linear (test):", linModel.score(x_test, y_test))

print("SkLearn Ridge (train):", ridgeModel.score(x_train, y_train))
print("SkLearn Ridge (test):", ridgeModel.score(x_test, y_test))

print("My Ridge (train):", myRidgeModel.score(x_train, y_train))
print("My Ridge (test):", myRidgeModel.score(x_test, y_test))



