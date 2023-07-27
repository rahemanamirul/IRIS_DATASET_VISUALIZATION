#!/usr/bin/env python
# coding: utf-8

# # IRIS DATASET VISUALIZATION 

# In[1]:


import numpy as np 
import pandas as pd 


# Importing pandas and seaborn module

# In[2]:


import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# Importing IRIS Data set

# In[3]:


iris=pd.read_csv(r'C:\Users\hp\OneDrive\Documents\Desktop\Iris.csv')
iris


# In[4]:


iris.head()


# In[5]:


iris.tail()


# In[6]:


iris.drop("Id",axis=1,inplace=True)


# In[7]:


iris.head(1)


# # Checking there is some missing values

# In[8]:


iris.info()


# In[9]:


iris["Species"].value_counts()


# 2.Bar plot:Here the frequency of the observation is plotted.In this case we are plotting the frequency of the three species in the Iris Dataset

# In[11]:


import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="Species", data=iris)
plt.show()


# We can see that there are 50 samples each of all the Iris Species in the data set.

# **4. Joint plot: ** Jointplot is seaborn library specific and can be used to quickly visualize and analyze the relationship between two variables and describe their individual distributions on the same plot.

# In[12]:


iris.head()


# In[13]:


fig=sns.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=iris)


# In[14]:


sns.jointplot(x="SepalLengthCm",y="SepalWidthCm", data=iris, kind="reg")


# In[15]:


fig=sns.jointplot(x="SepalLengthCm",y="SepalWidthCm", data=iris, kind="hex")


# # FacetGrit Plot

# In[16]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.FacetGrid(iris,hue='Species',height=5)\
.map(plt.scatter,"SepalLengthCm","SepalWidthCm")\
.add_legend()


# 6. Boxplot or Whisker plot Box plot was was first introduced in year 1969 by Mathematician John Tukey.Box plot give a statical summary of the features being plotted.Top line represent the max value,top edge of box is third Quartile, middle edge represents the median,bottom edge represents the first quartile value.The bottom most line respresent the minimum value of the feature.The height of the box is called as Interquartile range.The black dots on the plot represent the outlier values in the data.

# In[17]:


iris.head()


# In[18]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxplot(x='Species',y='PetalLengthCm',data=iris,order=['Iris-virginica','Iris-versicolor','Iris-setosa'],linewidth=2.5,orient='v',dodge=False)


# In[19]:


iris.boxplot(by="Species", figsize=(12, 6))


# # strip plot

# In[20]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.stripplot(x="Species",y="SepalLengthCm",data=iris,jitter=True,edgecolor="gray",size=8,palette="winter",orient="v")


#  # 8 combining box and strip plots

# In[21]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxplot(x="Species",y="SepalLengthCm",data=iris)
fig=sns.stripplot(x="Species",y="SepalLengthCm",data=iris,jitter=True,edgecolor="gray")


# 9. Violin Plot It is used to visualize the distribution of data and its probability distribution.This chart is a combination of a Box Plot and a Density Plot that is rotated and placed on each side, to show the distribution shape of the data. The thick black bar in the centre represents the interquartile range, the thin black line extended from it represents the 95% confidence intervals, and the white dot is the median.Box Plots are limited in their display of the data, as their visual simplicity tends to hide significant details about how values in the data are distributed

# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create the strip plot
ax= sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax= sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")

# Access the dotplot (individual data points)
dotplot = ax.collections

# Customize the dotplot (points)
for dot in dotplot:
    dot.set_facecolor("yellow")
    dot.set_edgecolor("black")

plt.show()


# In[ ]:





# In[23]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.violinplot(x="Species",y="SepalLengthCm",data=iris)


# In[24]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='PetalWidthCm',data=iris)


# 10. Pair Plot: A “pairs plot” is also known as a scatterplot, in which one variable in the same data row is matched with another variable's value, like this: Pairs plots are just elaborations on this, showing all variables paired with all the other variables.

# In[25]:


sns.pairplot(data=iris,kind="scatter")


# In[26]:


sns.pairplot(iris,hue="Species");


# 11. Heat map Heat map is used to find out the correlation between different features in the dataset.High positive or negative value shows that the features have high correlation.This helps us to select the parmeters for machine learning.
# 

# In[27]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(iris.corr(),annot=True,cmap="cubehelix",linewidths=1,linecolor="k",square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# 12. Distribution plot: The distribution plot is suitable for comparing range and distribution for groups of numerical data. Data is plotted as value points along an axis. You can choose to display only the value points to see the distribution of values, a bounding box to see the range of values, or a combination of both as shown here.The distribution plot is not relevant for detailed analysis of the data as it deals with a summary of the data distribution.

# In[28]:


iris.hist(edgecolor="black", linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)


# 13. Swarm plot It looks a bit like a friendly swarm of bees buzzing about their hive. More importantly, each data point is clearly visible and no data are obscured by overplotting.A beeswarm plot improves upon the random jittering approach to move data points the minimum distance away from one another to avoid overlays. The result is a plot where you can see each distinct data point, like shown in below plot

# In[29]:


sns.set(style="darkgrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
fig = sns.swarmplot(x="Species",y="PetalLengthCm", data=iris)


# In[30]:


iris.head()


# In[31]:


sns.set(style="whitegrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
ax = sns.violinplot(x="Species", y="PetalLengthCm", data=iris, inner=None)
ax = sns.swarmplot(x="Species", y="PetalLengthCm", data=iris,color="white", edgecolor="black")


# # LM PLOT

# In[32]:


fig=sns.lmplot(x="PetalLengthCm", y="PetalWidthCm", data=iris)


# # FacetGrid

# In[33]:


sns.FacetGrid(iris, hue="Species", height=6) \
    .map(sns.kdeplot, "PetalLengthCm") \
    .add_legend()
plt.ioff()


# # 22: boxplot plot

# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x="Species", y="SepalLengthCm", data=iris)
plt.show()


# In[35]:


iris.head(1)


# In[36]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxenplot(x='Species',y='SepalLengthCm',data=iris)
plt.show()


# # Dashboard

# In[38]:


sns.set_style("darkgrid")
f,axes=plt.subplots(2,2,figsize=(15,15))

k1=sns.boxplot(x="Species", y="PetalLengthCm", data=iris,ax=axes[0,0])
k2=sns.violinplot(x="Species", y="PetalLengthCm", data=iris,ax=axes[0,1])
k3=sns.stripplot(x="Species", y="PetalLengthCm", data=iris,jitter=True,edgecolor="gray",size=8,palette="winter",orient="v",ax=axes[1,0])
axes[1,1].hist(iris.PetalLengthCm,bins=100)
plt.show()


# # 31. Stacked Histogram

# In[39]:


iris["Species"] = iris["Species"].astype("category")


# In[40]:


list1=list()
mylabels=list()
for gen in iris.Species.cat.categories:
    list1.append(iris[iris.Species==gen].SepalLengthCm)
    mylabels.append(gen)
    
h=plt.hist(list1,bins=30,stacked=True,rwidth=1,label=mylabels)
plt.legend()
plt.show()


# With Stacked Histogram we can see the distribution of Sepal Length of Different Species together.This shows us the range of Sepan Length for the three different Species of Iris Flower.

# # 32.Area Plot:
# 
# Area Plot gives us a visual representation of Various dimensions of Iris flower and their range in dataset.

# In[41]:


iris.plot.area(y=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'],alpha=0.4,figsize=(12, 6));
plt.show()


# # 33.Distplot: 
# 
# It helps us to look at the distribution of a single variable.Kde shows the density of the distribution

# In[44]:


sns.displot(iris["SepalLengthCm"],kde=True,bins=20);
plt.show()


# In[ ]:




