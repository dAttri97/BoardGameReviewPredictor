#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Reading DataFrame

# In[43]:


df = pd.read_csv('games.csv')
# calculating size of the DataFrame
print(df.shape)
# reviewing the first 5 rows of the data
df.head(5)


# ### Creating a DataFrame to read the data types,unique count and null count

# In[44]:



lis=df.isnull().sum()
temp = pd.DataFrame({'Types':df.dtypes,'Unique Count':df.nunique(),'Null Count':lis},index=df.columns)
print(temp)


# In[45]:


df['average_rating'].plot.hist(bins=25)


# In[46]:


# very high number of games have an average rating of zero which is unusual
df['average_rating'].value_counts()


# In[47]:


# it will be best to remove these rows with average rating zero because they don't help in our predictive model
df[df['average_rating']==0].iloc[0]


# The game basically doesn't exists and this is just a false entry.
# Thus it is necessary to remove all such entries to get a higher accuracy in our prediction.

# In[48]:


df=df[df['users_rated']>0]
df['average_rating'].plot.hist(bins=25)


# In[49]:


# removing rows with null entries
df.dropna(axis=0,inplace = True)
df.isnull().sum()


# In[50]:


# Creating a Co-relation Matrix
coMat = df.corr()
plt.figure(figsize=(7,6),dpi=120)
sns.heatmap(coMat)


# In[51]:


# removing unnecassry columns from the DataFrame as they don't influence our Predictions
df.drop(columns=['id','name','type','bayes_average_rating'],axis=1,inplace=True)
df.head(1)
df.dtypes


# In[52]:


# creating our target variable
y = df['average_rating']
x = df.drop(['average_rating'],axis=1)
x.shape,y.shape


# In[53]:


# creating training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=97,test_size=.2)
x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[54]:


# importing the necessary library and funtions
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_log_error as mle


# In[55]:


# predicting the review through Linear Regression
lr = LinearRegression()
lr.fit(x_train,y_train)
predict_lr = lr.predict(x_test)
error_lr = mle(predict_lr,y_test)
error_lr


# In[56]:


# predicting the review through K-Neighbors Regression
# we need to compute the optimal value of n_neighbors inorder to make the best prediction
def elbow_curve(K):
    error=[]
    for i in K:
        kn = KNeighborsRegressor(n_neighbors=i)
        kn.fit(x_train,y_train)
        predict = kn.predict(x_test)
        error.append(mle(predict,y_test))
    return error


# In[57]:


K = range(2,50,2)
error = elbow_curve(K)


# In[58]:


plt.figure(figsize=(8,6),dpi=120)
plt.plot(K,error)
plt.xlabel('K Neighbours')
plt.ylabel('Error')


# In[59]:


for i in range(15,20):
    kn = KNeighborsRegressor(n_neighbors=i)
    kn.fit(x_train,y_train)
    predict = kn.predict(x_test)
    err = mle(predict,y_test)
    print(err," ",i)


# ### The least error comes out be on 18. Thus we can set our value of n_neighbours = 18
# 

# In[60]:


kn = KNeighborsRegressor(n_neighbors=18)
kn.fit(x_train,y_train)
predict_kn = kn.predict(x_test)
error_kn = mle(predict,y_test)
error_kn 


# ### Decision Tree Prediction

# Varying the maximum depth of the DecisionTree to find optimal Predictor

# In[61]:


train_score=[]
test_score=[]
for i in range(1,21):
    dt = DecisionTreeRegressor(max_depth=i,random_state=97)
    dt.fit(x_train,y_train)
    train_score.append(dt.score(x_train,y_train))
    test_score.append(dt.score(x_test,y_test))    


# In[62]:


data = pd.DataFrame({'Depth':range(1,21),'Train Score':train_score,'Test Score':test_score})
data


# In[63]:


plt.figure(figsize=(8,6),dpi=120)
plt.plot(data['Depth'],data['Train Score'],color='blue',label = 'Train Score',marker='o')
plt.plot(data['Depth'],data['Test Score'],color='red',label='Test Score',marker='o',linestyle='dashed')
plt.legend()


# Taking the maximum depth as 8 as it gives the optimal results

# ## Changing the leaf count keeping max depth 8

# In[64]:


train_score=[]
test_score=[]
for i in range(2,21):
    dt = DecisionTreeRegressor(max_depth=8,max_leaf_nodes=i,random_state=97)
    dt.fit(x_train,y_train)
    train_score.append(dt.score(x_train,y_train))
    test_score.append(dt.score(x_test,y_test))    


# In[66]:


data = pd.DataFrame({'Leaf':range(2,21),'Train Score':train_score,'Test Score':test_score})
data


# In[67]:


plt.figure(figsize=(8,6),dpi=120)
plt.plot(data['Leaf'],data['Train Score'],color='blue',label = 'Train Score',marker='o')
plt.plot(data['Leaf'],data['Test Score'],color='red',label='Test Score',marker='o',linestyle='dashed')
plt.legend()


# Taking number of leaf_node = 3

# In[68]:


#Making final prediction
dt = DecisionTreeRegressor(max_depth=8,max_leaf_nodes=3,random_state=97)
dt.fit(x_train,y_train)


# In[70]:


predict_dt = dt.predict(x_test)
score_dt = dt.score(x_test,y_test)
err_dt = mle(predict_dt,y_test)
err_dt


# ## Using ensemble method to take the mean of all predictions form all the models and thus get the best prediction with minimum error

# ### Taking Average of all Predictions

# In[72]:


final_predict=[]
from statistics import mean
for i in range(len(y_test)):
    final_predict.append(mean([predict_dt[i],predict_lr[i],predict_kn[i]]))
final_predict[:10]


# In[73]:


mle(final_predict,y_test)


# In[74]:


err_dt,error_kn,error_lr


# The final prediction seems to performing worse than KNeighbor.
# Thus using the Rank Averaging

# ### Rank Averaging

# In[82]:


data = pd.DataFrame({'Predictor':['Decision Tree','KNeighbor','LinearRegressor'],'Error':[err_dt,error_kn,error_lr]},index=[1,2,3])
data


# In[83]:


data=data.sort_values('Error',ascending=False)
data


# In[84]:


data['Rank'] = [i for i in range(1,4)]
data


# In[85]:


Sum = data['Rank'].sum()
data['Weight'] = data['Rank']/Sum
data


# In[105]:


wt_pred_lr = predict_lr*float(data.loc[[3],['Weight']].values)
wt_pred_kn = predict_kn*float(data.loc[[2],['Weight']].values)
wt_pred_dt = predict_dt*float(data.loc[[1],['Weight']].values)
final = wt_pred_dt+wt_pred_kn+wt_pred_lr
final[:10]


# In[106]:


mle(final,y_test)


# Still the predictor is weaker than KNeighbor 
# Thus we will do our final predictions with KNeighbor, ignoring ensemble approach
