#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
d1= pd.read_csv("Ecommerce_Customers.csv")
columns = ["Email","Address","Avatar","Avg Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent"]
d1= pd.read_table("Ecommerce_Customers.csv", sep=',', header=None, names=columns)
x= d1[["Avg Session Length", "Time on App","Time on Website", "Length of Membership"]]
y1= d1['Yearly Amount Spent']


# In[25]:


d1


# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y1_train,y1_test=train_test_split(x,y1,test_size=.3,random_state=20)
from sklearn.linear_model import LinearRegression
r=LinearRegression()
r.fit(x_train,y1_train)
Predict = r.predict(x_test)


# In[28]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y1_test,Predict))
print('MSE:', metrics.mean_squared_error(y1_test,Predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y1_test,Predict)))


# In[29]:


plt.scatter(y1_test,Predict)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[31]:


x= d1['Avg Session Length']
y = d1['Yearly Amount Spent']
plt.xlabel("Avg Session Length")
plt.ylabel("Yearly Amount Spent")
plt.scatter(x,y)
plt.show()


# In[32]:


x= d1['Time on App']
y = d1['Yearly Amount Spent']
plt.xlabel("Time on App")
plt.ylabel("Yearly Amount Spent")
plt.scatter(x,y)
plt.show()


# In[34]:


x= d1['Time on Website']
y = d1['Yearly Amount Spent']
plt.xlabel("Time on Website")
plt.ylabel("Yearly Amount Spent")

plt.scatter(x,y)
plt.show()


# In[35]:


x= d1["Length of Membership"]
y = d1['Yearly Amount Spent']
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")

plt.scatter(x,y)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




