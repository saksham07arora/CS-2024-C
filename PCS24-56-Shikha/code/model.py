#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"TRAIN.csv")


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.duplicated().sum()


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


df.describe()


# In[8]:


df_train = pd.read_csv(r"TRAIN.csv")
df_test  = pd.read_csv(r"Test1.csv")


# In[9]:


df = pd.concat([df_train,df_test],ignore_index=True)


# In[10]:


df.describe()


# In[22]:


df.size


# In[11]:


df= df[df['Speed'] < 109]


# In[12]:


df.describe()


# In[15]:


new_df= df[df['Vertical_Acceleration'] < 0.008]


# In[17]:


new_df= new_df[new_df['Lateral_Acceleration'] < 0.005]


# In[18]:


new_df= new_df[new_df['Longitudinal_Acceleration'] < 0.04]


# In[19]:


new_df.describe()


# In[23]:


new_df.size


# In[24]:


X=df.drop('Status',axis=1)
y=df['Status']


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[27]:


X_train


# In[28]:


from sklearn.preprocessing import StandardScaler


# In[29]:


scaler=StandardScaler()


# In[30]:


scaler.fit(df.drop('Status',axis=1))


# In[31]:


scaler_feature =scaler.transform(df.drop('Status',axis=1))


# In[32]:


df_feature=pd.DataFrame(scaler_feature,columns=df.columns[:-1])


# In[33]:


df_feature.head()


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(scaler_feature, df['Status'], test_size=0.33, random_state=42)


# In[36]:


from sklearn.ensemble import RandomForestClassifier


# In[37]:


hyperparameters = {
    'max_features': 'sqrt',
    'min_samples_leaf': 12,
    'min_samples_split': 27,
    'n_estimators': 3000
}


# In[38]:


rfc = RandomForestClassifier(**hyperparameters)


# In[40]:


rfc.fit(X_train, y_train)


# In[41]:


pred12=rfc.predict(X_test)


# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


accuracy = accuracy_score(y_test, pred12)
print("Test Set Accuracy with Best Hyperparameters:", accuracy)


# In[45]:


df2 = pd.read_csv(r"TRAIN.csv")


# In[46]:


df2.head(1)


# In[47]:


rfc.fit(X,y)


# In[48]:


df3= pd.read_csv(r"Test1.csv")


# In[49]:


df4=df3


# In[50]:


df3.size


# In[51]:


df4.size


# In[52]:


df3 = df3[['Speed', 'Vertical_Acceleration', 'Lateral_Acceleration', 'Longitudinal_Acceleration', 'Roll', 'Pitch', 'Yaw']]


# In[53]:


df3.size


# In[54]:


df3.columns


# In[55]:


test=df4['Status']


# In[56]:


df4.size


# In[57]:


df3.tail()


# In[58]:


train=df3


# In[61]:


train.size


# In[59]:


pred=rfc.predict(train)


# In[63]:


pred.size


# In[64]:


accuracy = accuracy_score(test, pred)
print("Test Set Accuracy with Best Hyperparameters:", accuracy)


# In[69]:


from sklearn.metrics import confusion_matrix, classification_report


# In[70]:


confusion_matrix(pred,test)


# In[71]:


print(classification_report(pred,test))


# In[67]:


import pickle


# In[68]:


file_path = 'model.pkl'

# Dump the object to a file
with open(file_path, 'wb') as file:
    pickle.dump(rfc, file)

