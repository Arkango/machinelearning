import pandas as pd
import quandl ,math,datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style 
import pickle

style.use('ggplot')

ds = quandl.get('WIKI/GOOGL')

#print(ds.head())

ds = ds[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
ds ['HL_PCT'] = (ds['Adj. High'] - ds['Adj. Close'] ) / ds['Adj. Close'] * 100.0
ds ['CHANGE_PCT'] = (ds['Adj. Close'] - ds['Adj. Open'] ) / ds['Adj. Open'] * 100.0

ds = ds[['Adj. Close','HL_PCT','CHANGE_PCT','Adj. Volume']]

#print(ds.head())

forecast_col = 'Adj. Close'

#fill not available
ds.fillna(-99999,inplace=True)

# we use ten days (ten percent in this case only) to predict the proce of today
forecast_out = int(math.ceil(0.01*len(ds)))

ds['label'] = ds[forecast_col].shift(-forecast_out)


#print(ds.head())

x = np.array(ds.drop(['label'],1))

#scale based on all data
x = preprocessing.scale(x)
x = x[:-forecast_out]
x_lately = x[-forecast_out:]
ds.dropna(inplace=True)
y = np.array(ds['label'])

x_train , x_test , y_train , y_test = model_selection.train_test_split(x,y,test_size=0.2)

# con regressione lineare semplice clf = LinearRegression()
##clf = LinearRegression(n_jobs= 10) # con regressione lineare semplice multithread (velocizza solo la parte di training)
#support vector defualt kernel liner clf = svm.SVR()   
#kernel polinomiale clf = svm.SVR(kernel = 'poly')  
##clf.fit(x_train,y_train)


#with open ('linearregression.pickle','wb') as f:
#    pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

confidence = clf.score(x_test,y_test)

#print(confidence)

forecast_set = clf.predict(x_lately)

#print(forecast_set, confidence, forecast_out)

ds['Forecast'] = np.nan

last_date = ds.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    ds.loc[next_date] = [np.nan for _ in range(len(ds.columns) -1)] +  [i]

ds['Adj. Close'].plot()
ds['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
#plt.savefig('graph.jpg')
plt.show()

"""
y = mx +b 

xS = x segnato
yS = y segnato
xyS = xy segnato

m = ((xS X yS) - xyS ) / xS**2 - x**2S

b = y - mxS
"""