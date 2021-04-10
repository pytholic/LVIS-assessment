#!/usr/bin/env python
# coding: utf-8

# # Dataset description

# This data set consists of three types of entities: (a) the specification of an auto in terms of various characteristics, (b) its assigned insurance risk rating, (c) its normalized losses in use as compared to other cars. The second rating corresponds to the degree to which the auto is more risky than its price indicates. Cars are initially assigned a risk factor symbol associated with its price. Then, if it is more risky (or less), this symbol is adjusted by moving it up (or down) the scale. Actuarians call this process "symboling". A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.
# 
# The third factor is the relative average loss payment per insured vehicle year. This value is normalized for all autos within a particular size classification (two-door small, station wagons, sports/speciality, etc...), and represents the average loss per car per year.
# 
# Note: Several of the attributes in the database could be used as a "class" attribute.

# In[1]:


## Import Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from xgboost import XGBRegressor
import sklearn.metrics as metrics
import math
from scipy.stats import norm, skew
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cols = ['symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']


# In[3]:


# Importing dataset from UCI Machine Learning Repository
df = pd.read_csv('./data/imports-85.data', names=cols)

# Seeing the first 5 rows
df.head()


# In[4]:


df.shape 


# In[5]:


id = np.array(range(0,205))
df = df.assign(ID=id)
df.head()


# # Data Visualization, Preprocessing and Cleaning

# # # Finding and removing missing values

# In[6]:


# Seeing dataset structure
df.info()


# In[7]:


# Seeing columns
df.describe()


# In[8]:


# Looking for total missing values in each column
df.isna().sum()


# In[9]:


# Replace missign values with numpy NaN values and drop the corresponding observations
df.replace('?',np.nan, inplace=True)


# In[10]:


df.head()


# In[11]:


df.dropna(inplace=True)
df.head()


# In[12]:


# Checking if we still have any missing values
plt.subplots(0,0, figsize = (18,5))
ax = (df.isnull().sum()).sort_values(ascending = False).plot.bar(color = 'blue')
plt.title('Missing values per column', fontsize = 20);


# # # Target Analysis

# In[13]:


# Our target column is the nosmalized losses column
print(df['normalized-losses'].describe())


# In[14]:


sns.distplot(df['normalized-losses'])


# In[15]:


print("Skewness: %f" % df['normalized-losses'].skew())
print("Kurtosis: %f" % df['normalized-losses'].kurt())


# In[16]:


df["normalized-losses"] = pd.to_numeric(df["normalized-losses"])


# In[17]:


# fixing the skew

df['normalized-losses'] = np.log1p(df['normalized-losses'])
sns.distplot(df['normalized-losses'], fit=norm);


# # # Exploratory Analysis

# In[18]:


# Plotting a histogram of our ftarget feature normalzied-losses
plt.figure(figsize=(20,6)) # creating the figure
plt.hist(df['normalized-losses'] # plotting the histogram
         ,bins=30 # defyning number of bars
         ,label='normalzied loss' # add legend
        ,color='blue') # defyning the color

plt.xlabel('price') # add xlabel
plt.ylabel('frequency') # add ylabel
plt.legend()
plt.title('normalized losses');


# In[19]:


# Saving numerical features
num_var = ['symboling','normalized-losses','wheel-base','length'
          ,'width','height','curb-weight','engine-size','bore'
           ,'stroke','compression-ratio','horsepower','peak-rpm'
           ,'city-mpg','highway-mpg']

# plotting a histogram for each feature
df[num_var].hist(bins=10
                   , figsize=(50,30)
                   , layout=(4,4));


# In[20]:


# Numerical variables correlation
corr = df.corr() # creting the correlation matrix

plt.figure(figsize=(12,12)) # creating the and difyning figure size
ax = sns.heatmap( # plotting correlation matrix
    corr,vmin=-1, vmax=1, center=0,
    annot=True,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels( # adding axes values
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# Symboling, and engine-size have highest correlation with normalized loss

# In[21]:


# # Plotting a scatter plot of relation between engine_size, symboling and normalzied-losses
g = sns.pairplot(data=df,
                  y_vars=['normalized-losses'],
                  x_vars=['symboling', 'engine-size'])
g.fig.set_size_inches(10,5)


# Let's make a bivariate a analysis of each categorical feature with our target to understand better this features¶

# In[22]:


# Plotting Distribution of symboling into normalized-losses
plt.figure(figsize=(8,6))
sns.boxplot(x='symboling',y='normalized-losses',data=df, 
                 palette="colorblind")
plt.title('Distribution of symboling into normalized-losses');


# As mentioned in the data set description, the cars with high symboling are more risky. These risky cars also show high normalized-loss values. The higher the risk, the higher the normalized-loss.

# In[23]:


# Plotting Distribution of symboling into noemalized-losses
plt.figure(figsize=(16,8))
sns.boxplot(x='engine-size',y='normalized-losses',data=df, 
                 palette="colorblind")
plt.title('Distribution of engine-size into normalized-losses');


# Here we can see that there is not a very clear pattern or relation between engine size and normalzied loss value. That is also the relation why correlation value was very low (0.19). Maybe some of the cars with large engine size indicate a little higher normalized-loss but still the trend is not very consistent. Therefore, it is not a very useful feature.

# #  Training

# # # Data preparation

# In[24]:


# Conveting categorical data to numerical

df = pd.get_dummies(df)
df.head()


# In[25]:


# splitting the data with our target into y1 and the rest of data into x1
x = df.drop(['ID', 'normalized-losses'], axis=1)
y = df['normalized-losses']


# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42) 
# 80% train and 20% test


# In[27]:


x_train.shape, y_train.shape


# In[28]:


x_test.shape, y_test.shape


# # # Model Selection

# In[29]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error


# In[30]:


import xgboost as XGB

#1st model
model = XGB.XGBRegressor()
model.fit(x_train, y_train, verbose=True) # or x_train['symboling'] because it depends mostly on symboling


# In[31]:


from sklearn.linear_model import LinearRegression

# 2nd model
model2 = LinearRegression()
model2.fit(x_train, y_train)


# In[32]:


y_pred = np.floor(np.expm1(model.predict(x_test)))
y_pred


# In[33]:


y_pred2 = np.floor(np.expm1(model2.predict(x_test)))
y_pred2


# In[34]:


y_test = np.floor(np.expm1(y_test))


# In[35]:


y_test.head()


# In[36]:


# model1 quality metrics in test dataset prediction
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[37]:


# model2 quality metrics in test dataset prediction
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))


# XGB Regressor seems to perform better

# In[ ]:


# sub = pd.DataFrame()
# sub['ID'] = test_id
# sub['normalized-error'] = y_pred
# sub.to_csv('mysubmission.csv',index=False)

