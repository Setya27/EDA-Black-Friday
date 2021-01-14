#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

### Import Libraries and Dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('train.csv')
df.head(3)

### EDA

df.shape
df.info()
df.describe().transpose()
df.isnull().sum()
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='coolwarm');

### Plot Dataset

newdf = df.copy()

## Age Distribution
sns.countplot(newdf['Age']);
plt.title('Age Distribution');
# Most Populated Age Group is 26-35 years

## Gender Distribution
sns.countplot(newdf['Gender']);
plt.title('Gender Distribution');
# More Males than Females

## Occupation Distribution
sns.countplot(newdf['Occupation']);
plt.title('Occupation Distribution');
# Occupation number 0 and 4 employ the most customers.
# Occupation number 8 and 9 employ the least customers.

## City Category Distribution
city = newdf['City_Category'].value_counts()
plt.pie(city.values, labels=city.index, startangle=-30,
       explode=(0,0.20,0), autopct='%1.1f%%');
plt.title('City_Category Distribution');
# Most Customers are from City B

## Marital Status Distribution
sns.countplot(newdf['Marital_Status']);
plt.title('Marital Status Distribution');
# Majority of Customers are Unmarried

## Stay in City Distribution
sns.countplot(newdf['Stay_In_Current_City_Years']);
plt.title('Stay_In_Current_City_Years Distribution');
# Most customers are living in the city for 1 years

## Purchase Distribution
sns.displot(newdf['Purchase'], bins=20);
plt.title('Purchase amount Distribution');
plt.xlabel('Amount');
plt.ylabel('Number of People');
# There is a direct correlation with number of customers and amount spent

### Bivariate Analysis

##  Age - Gender Analysis
sns.countplot(newdf['Age'], hue=newdf['Gender']);
plt.title('Gender - Age Distribution');

## Age - Purchase Analysis
sns.boxplot(newdf['Age'], newdf['Purchase']);
plt.title('Age - Purchase Analysis');
age = ['0-17', '55+', '26-35', '46-50', '51-55', '36-45', '18-25']
purchase = []
for item_age in age:
    purchase.append(newdf[newdf['Age'] == item_age]['Purchase'].sum())
plt.bar(age, purchase, align='center');
plt.xlabel('Age');
plt.ylabel('Money Spent');
plt.title('Age - Purchase Analysis');


### Multivariate Analysis 
sns.heatmap(newdf.corr(), annot=True)

