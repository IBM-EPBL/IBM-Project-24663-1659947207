#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot

#for offline plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[2]:


oil = pd.read_csv('COPP.csv')
oil.head()


# In[3]:


oil.info()


# In[4]:


oil['Date'] = pd.to_datetime(oil['Date'])


# In[5]:


print(f'Dataframe contains crude oil prices between {oil.Date.min()} {oil.Date.max()}') 
print(f'Total days = {(oil.Date.max()  - oil.Date.min()).days} days')


# In[6]:


oil.describe()


# In[7]:


oil[['Open','High','Low','Close','Adj Close']].plot(kind='box')


# In[8]:


layout = go.Layout(
    title=' Prices of crude oil ',
    xaxis=dict(
        title='Date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

oil_data = [{'x':oil['Date'], 'y':oil['Close']}]
plot = go.Figure(data = oil_data, layout=layout)


# In[9]:


iplot(plot)


# LSTM

# In[10]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


# In[11]:


data = pd.read_csv('COPP TRAIN.csv')
data.head()


# In[12]:


data.info()


# In[13]:


data["Close"]=pd.to_numeric(data.Close,errors='coerce')
data = data.dropna()
trainData = data.iloc[:,4:5].values


# In[14]:


data.info()


# In[15]:


sc = MinMaxScaler(feature_range=(0,1))
trainData = sc.fit_transform(trainData)
trainData.shape


# In[16]:


X_train = []
y_train = []

for i in range (60,165): 
    X_train.append(trainData[i-60:i,0]) 
    y_train.append(trainData[i,0])

X_train,y_train = np.array(X_train),np.array(y_train)


# In[17]:


X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1)) #adding the batch_size axis
X_train.shape


# In[18]:


model = Sequential()

model.add(LSTM(units=100, return_sequences = True, input_shape =(X_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units =1))
model.compile(optimizer='adam',loss="mean_squared_error")


# In[19]:


hist = model.fit(X_train, y_train, epochs = 70, batch_size = 32, verbose=2)


# In[20]:


plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


# In[21]:


testData = pd.read_csv('COPP TEST.csv')

testData["Close"]=pd.to_numeric(testData.Close,errors='coerce')
testData = testData.dropna()
testData = testData.iloc[:,4:5]
y_test = testData.iloc[60:,0:].values 

#input array for the model
inputClosing = testData.iloc[:,0:].values 
inputClosing_scaled = sc.transform(inputClosing)
inputClosing_scaled.shape
X_test = []
length = len(testData)
timestep = 60

for i in range(timestep,length):  
    X_test.append(inputClosing_scaled[i-timestep:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
X_test.shape


# In[22]:


y_pred = model.predict(X_test)
y_pred


# In[23]:


predicted_price = sc.inverse_transform(y_pred)


# In[ ]:





# In[24]:


plt.plot(predicted_price, color = 'green', label = 'Predicted crude oil Price')
plt.title('Crude Oil price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[25]:


print(predicted_price)


# In[ ]:




