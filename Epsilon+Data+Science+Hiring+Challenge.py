
# coding: utf-8

# # EPSILON DATA SCIENCE HIRING CHALLENGE - DEBADRI

# ## Importing dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
from scipy.signal import lfilter


# In[2]:


import chardet
import pandas as pd

with open('stockdata.csv', 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large


df=pd.read_csv('stockdata.csv', encoding=result['encoding'])
df.head()


# ## Data Pre-processing

# In[3]:


df.dtypes


# In[4]:


df.shape


# In[5]:


df['StockCode'].nunique()


# In[6]:


df.isnull().sum()


# In[7]:


df['CustomerID'].fillna(-1,inplace=True)


# In[8]:


df.shape


# In[9]:


df['Description'].fillna('Not Available',inplace=True)


# In[10]:


df.shape


# In[11]:


df['InvoiceDate'] =  pd.to_datetime(df['InvoiceDate'], infer_datetime_format=True)


# In[12]:


df.dtypes


# In[13]:


df.index=df['InvoiceDate']


# In[14]:


df.head()


# In[15]:


#plt.figure(figsize=(15,8))
#n = 1200  # the larger n is, the smoother curve will be
#b = [1.0 / n] * n
#a = 1
#yy = lfilter(b,a,pd.rolling_mean(df['Quantity'],1))
#plt.plot(df['InvoiceDate'], yy, linewidth=2, linestyle="-", c="b") 
#plt.xlabel('Month-Year')
#plt.ylabel('Count')
#plt.title('No. of Units per Stock sold')


# In[16]:


#plt.figure(figsize=(15,8))
#n = 1200  # the larger n is, the smoother curve will be
#b = [1.0 / n] * n
#a = 1
#yy = lfilter(b,a,pd.rolling_mean(df['UnitPrice'],1))
#plt.plot(df['InvoiceDate'], yy, linewidth=2, linestyle="-", c="b") 
#plt.xlabel('Month-Year')
#plt.ylabel('Price')
#plt.title('Price of Units per Stock')


# ## Time Series analysis & Variation of the prices of goods according to the change in date and time for Top 5 highest selling stocks

# In[17]:


x=df['StockCode'].value_counts()


# In[18]:


x


# In[19]:


df2 = x.rename_axis('StockCode').reset_index(name='counts')
pd.DataFrame(df2)
print (df2)


# In[20]:


df2['StockCode'][:5]


# In[21]:


serial = [1,2,3,4,5]

plt.figure(figsize=(12,10))
           
LABELS=["85123A", "22423", "85099B","47566","20725"]

plt.bar(serial, df2['counts'][:5], align='center')
plt.xticks(serial, LABELS)
plt.ylabel('No. of Units')
plt.xlabel('StockCode')
plt.title("Top 5 Highest Selling Stocks")
plt.show()


# ## Time-Series Forecasting, Price Variation for stock 85123A

# In[22]:


foo = df.ix[(df['StockCode']=="85123A")]


# In[23]:


foo


# In[24]:


plt.figure(figsize=(15,8))
n = 1200  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,pd.rolling_mean(foo['UnitPrice'],1))
plt.plot(foo['InvoiceDate'], yy, linewidth=2, linestyle="-", c="b") 
plt.xlabel('Month-Year')
plt.ylabel('Price per Unit')
plt.title('Price of Units of Stock 85123A')


# In[25]:


from statsmodels.tsa.arima_model import ARIMA


# In[26]:


ts_stock1=foo[['InvoiceDate','UnitPrice']]


# In[27]:


ts_stock1 = ts_stock1.reset_index(drop=True)


# In[28]:


ts_stock1.head()


# In[29]:


ts_stock1.dtypes


# In[30]:


ts_stock1.isnull().any()


# In[31]:


data = pd.Series(ts_stock1.UnitPrice.values, index=ts_stock1.InvoiceDate)


# In[32]:


data


# In[33]:


X=data.values


# In[34]:


size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))



# In[35]:


# plot
plt.figure(figsize=(15,8))
plt.plot(test)
plt.plot(predictions, color='red')
plt.title('Forecasted Values')
plt.show()


# ## Time-Series Forecasting, Price Variation for stock 22423

# In[36]:


foo = df.ix[(df['StockCode']=="22423")]


# In[37]:


plt.figure(figsize=(15,8))
n = 1200  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,pd.rolling_mean(foo['UnitPrice'],1))
plt.plot(foo['InvoiceDate'], yy, linewidth=2, linestyle="-", c="b") 
plt.xlabel('Month-Year')
plt.ylabel('Price per Unit')
plt.title('Price of Units of Stock 22423')


# In[38]:


ts_stock1=foo[['InvoiceDate','UnitPrice']]


# In[39]:


ts_stock1 = ts_stock1.reset_index(drop=True)


# In[40]:


data = pd.Series(ts_stock1.UnitPrice.values, index=ts_stock1.InvoiceDate)


# In[41]:


X=data.values


# In[42]:


size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


# In[43]:


# plot
plt.figure(figsize=(15,8))
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# ## Time-Series Forecasting, Price Variation for stock 85099B

# In[44]:


foo = df.ix[(df['StockCode']=="85099B")]


# In[45]:


plt.figure(figsize=(15,8))
n = 1200  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,pd.rolling_mean(foo['UnitPrice'],1))
plt.plot(foo['InvoiceDate'], yy, linewidth=2, linestyle="-", c="b") 
plt.xlabel('Month-Year')
plt.ylabel('Price per Unit')
plt.title('Price of Units of Stock 85099B')


# In[46]:


ts_stock1=foo[['InvoiceDate','UnitPrice']]


# In[47]:


ts_stock1 = ts_stock1.reset_index(drop=True)


# In[48]:


data = pd.Series(ts_stock1.UnitPrice.values, index=ts_stock1.InvoiceDate)


# In[49]:


X=data.values


# In[50]:


size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))



# In[51]:


# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.title('Forecasted Values')
plt.show()


# ## Time-Series Forecasting, Price Variation for stock 47566

# In[52]:


foo = df.ix[(df['StockCode']=="47566")]


# In[53]:


plt.figure(figsize=(15,8))
n = 1200  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,pd.rolling_mean(foo['UnitPrice'],1))
plt.plot(foo['InvoiceDate'], yy, linewidth=2, linestyle="-", c="b") 
plt.xlabel('Month-Year')
plt.ylabel('Price per Unit')
plt.title('Price of Units of Stock 47566')


# In[54]:


ts_stock1=foo[['InvoiceDate','UnitPrice']]


# In[55]:


ts_stock1 = ts_stock1.reset_index(drop=True)


# In[56]:


data = pd.Series(ts_stock1.UnitPrice.values, index=ts_stock1.InvoiceDate)


# In[57]:


X=data.values


# In[58]:


size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))


# In[59]:


plt.figure(figsize=(15,8))
plt.plot(test)
plt.plot(predictions, color='red')
plt.title('Forecasted Values')
plt.show()


# ## Time-Series Forecasting, Price Variation for stock 20725

# In[60]:


foo = df.ix[(df['StockCode']=="20725")]


# In[61]:


foo


# In[62]:


plt.figure(figsize=(15,8))
n = 1200  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
yy = lfilter(b,a,pd.rolling_mean(foo['UnitPrice'],1))
plt.plot(foo['InvoiceDate'], yy, linewidth=2, linestyle="-", c="b") 
plt.xlabel('Month-Year')
plt.ylabel('Price per Unit')
plt.title('Price of Units of Stock 20725')


# In[63]:


ts_stock1=foo[['InvoiceDate','UnitPrice']]


# In[64]:


ts_stock1 = ts_stock1.reset_index(drop=True)


# In[65]:


data = pd.Series(ts_stock1.UnitPrice.values, index=ts_stock1.InvoiceDate)


# In[66]:


X=data.values


# In[67]:


size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))



# In[68]:


plt.figure(figsize=(15,8))
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.title('Forecasted Values')
plt.show()


# ## Correlation between unit price and quantity sold.

# In[ ]:


print("Correlation b/w Unit Price and Quantity Sold",df['Quantity'].corr(df['UnitPrice']))


# They are not highly correlated but it is evident that if one increases then the other will decrease, since correlation value is negative

# ## Checking if description has any effect on the price or sale of goods.

# In[ ]:


print("Correlation b/w Description and Quantity Sold",df['Quantity'].corr(df['Description']))


# In[ ]:


df['Description'][0]


# In[ ]:


df['Description'][0]="Not Available"
df['Description'][1]="Not Available"
df['Description'][2]="Not Available"


# In[ ]:


print("Correlation b/w Description and Quantity Sold",df['Quantity'].corr(df['Description']))


# Due to slow laptop and lack of submission time, I wasn't able to execute the last part,

# ## Tools Used: Pandas, Numpy
# ## For Visualization: Matplotlib
# ## For Forecasting: ARIMA Model

# In[ ]:




