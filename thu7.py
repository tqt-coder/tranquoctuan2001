import numpy as np
import pandas as pd

df=pd.read_csv('./weatherAUS.csv')
print('Do lon cua bang [frame] du lieu thoi tiet: ',df.shape)

print(df[0:5])

print(df.count().sort_values())

df=df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','Date','RISK_MM'],axis=1) #df =
print(df.shape)

df = df.dropna(how='any')
print(df.shape)

from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df= df[(z < 3).all(axis=1)]
print(df.shape)

df['RainToday'].replace({'No':0,'Yes':1},inplace=True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

print(df[['RainToday','RainTomorrow']])

print(df['RainToday'].array)

df.describe()

df['WindDir9am'].value_counts().plot(kind='bar')

df['WindDir3pm'].replace({'S':0,'SW':1,'SE':2,'WSW':3,
           'ENE':4,'W':5,'E':6,'SSW':7,'ESE':8,'N':9,'NE':10,'SSE':11,
           'NNW':12,'WNW':13,'NW':14,'NNE':15},inplace=True)
df['WindGustDir'].replace({'S':0,'SW':1,'SE':2,'WSW':3,
           'ENE':4,'W':5,'E':6,'SSW':7,'ESE':8,'N':9,'NE':10,'SSE':11,
           'NNW':12,'WNW':13,'NW':14,'NNE':15},inplace=True)
df['WindDir9am'].replace({'S':0,'SW':1,'SE':2,'WSW':3,
           'ENE':4,'W':5,'E':6,'SSW':7,'ESE':8,'N':9,'NE':10,'SSE':11,
           'NNW':12,'WNW':13,'NW':14,'NNE':15},inplace=True)

df['WindDir9am'].value_counts().plot(kind='bar')

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
df.iloc[4:10]
print(df)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
print (scaled_data)

from sklearn.feature_selection import SelectKBest, chi2
X = df.loc[:,df.columns!='RainTomorrow']
y = df[['RainTomorrow']]
selector = SelectKBest(chi2, k=3)
selector.fit(X, y)
X_new = selector.transform(X)
#Index(['Rainfall', 'Humidity3pm', 'RainToday'], dtype='object')
print(X.columns[selector.get_support(indices=True)])

print(X.columns[selector.get_support(indices=True)])


df = df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]
X = df[['Humidity3pm']]
y = df[['RainTomorrow']]
selector = SelectKBest(chi2, k='all')
selector.fit(X, y)
X_new = selector.transform(X)
#Index(['Rainfall', 'Humidity3pm', 'RainToday'], dtype='object')
print(X.columns[selector.get_support(indices=True)])


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
t1 = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.25) 
clf_logreg = LogisticRegression(random_state=0)
clf_logreg.fit(X_train, y_train)
y_pred = clf_logreg.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('Accuracy using Logistic Regression:', score)
print('Time taken using Logistic Regression:',time.time() - t1)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
t0=time.time()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf_rf= RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
clf_rf.fit(X_train,y_train)
y_pred = clf_rf.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Accuracy using Random Forest Classifier:',score)
print('Time taken using Random Forest Classifier:' ,time.time()-t0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
t0=time.time()
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.25) 
clf_dt =DecisionTreeClassifier(random_state=0)
clf_dt.fit(X_train,y_train)
y_pred = clf_dt.predict(X_test)
score = accuracy_score(y_test,y_pred)
print('Accuracy using Decision Tree Classifier:',score)
print('Time taken using Decision Tree Classifier:' ,time.time()-t0)