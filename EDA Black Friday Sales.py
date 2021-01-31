#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('train.csv')


# In[4]:


df.head(3)


# In[5]:


df.info()


# In[6]:


df.describe().transpose()


# In[7]:


df.isnull().sum()


# In[8]:


sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm');


# In[9]:


newdf = df.copy()


# # Exploratory Data Analysis

# ## Age Distribution

# In[10]:


sns.countplot(newdf['Age']);
plt.title('Age Distribution');


# Most Populated Age Group is 26-35 years

# ## Gender Distribution

# In[11]:


sns.countplot(newdf['Gender']);
plt.title('Gender Distribution');


# More Males than Females

# ## Occupation Distribution

# In[12]:


sns.countplot(newdf['Occupation']);
plt.title('Occupation Distribution');


# Occupation number 0 and 4 employ the most customers.
# Occupation number 8 and 9 employ the least customers.

# ## City Category Distribution

# In[13]:


city = newdf['City_Category'].value_counts()


# In[14]:


plt.pie(city.values, labels=city.index, startangle=-30,
       explode=(0,0.20,0), autopct='%1.1f%%');
plt.title('City_Category Distribution');


# Most Customers are from City B

# ## Marital Status Distribution

# In[15]:


sns.countplot(newdf['Marital_Status']);
plt.title('Marital Status Distribution');


# Majority of Customers are Unmarried

# ## Stay in City Distribution

# In[16]:


sns.countplot(newdf['Stay_In_Current_City_Years']);
plt.title('Stay_In_Current_City_Years Distribution');


# Most customers are living in the city for 1 years

# ## Purchase Distribution

# In[17]:


sns.displot(newdf['Purchase'], bins=20);
plt.title('Purchase amount Distribution');
plt.xlabel('Amount');
plt.ylabel('Number of People');


# There is a direct correlation with number of customers and amount spent

# # Bivariate Analysis

# ##  Age - Gender Analysis

# In[18]:


sns.countplot(newdf['Age'], hue=newdf['Gender']);
plt.title('Gender - Age Distribution');


# ## Age - Purchase Analysis

# In[19]:


sns.boxplot(newdf['Age'], newdf['Purchase']);
plt.title('Age - Purchase Analysis');


# In[20]:


age = ['0-17', '55+', '26-35', '46-50', '51-55', '36-45', '18-25']
purchase = []
for item_age in age:
    purchase.append(newdf[newdf['Age'] == item_age]['Purchase'].sum())


# In[21]:


plt.bar(age, purchase, align='center');
plt.xlabel('Age');
plt.ylabel('Money Spent');
plt.title('Age - Purchase Analysis');


# In[22]:


newdf['Purchase'].mean()


# In[23]:


newdf['Combined_G_M'] = newdf.apply(lambda x: '%s-%s' % (x['Gender'], x['Marital_Status']), axis=1)


# In[24]:


sns.heatmap(newdf.corr(), annot=True)

