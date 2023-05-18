#!/usr/bin/env python
# coding: utf-8

# # Small Support Vector Machine tutorial: 
# Needed packages sklearn   
#                            SMA May 2023  
#                            More info (and it has really detailed material): https://scikit-learn.org/stable/modules/svm.html     
#                            https://scikit-learn.org/stable/modules/svm.html  
#                            (The first part of the tutorial is a reproduction from a tutorial in XX website)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#This tutorial is done first with pandas for handling the CSV but the same thing can be done with numpu


# In[2]:


bankdata = pd.read_csv("bill_authentication.csv")


# In[3]:


bankdata.shape


# In[4]:


bankdata.head()


# In[5]:


print(bankdata)


# SVM needs an array where the **X** is the samples and **Y** the features

# In[6]:


X = bankdata.drop('Class', axis=1)
y = bankdata['Class'] #Saves in the y the type "the label"
#X = bankdata['Variance'] #Here it will use the variance to build the model


# The following is a feature from sklearn that randomly can select a subset from the arrays for doing the training and the test 

# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle=True, random_state=1)


# Now we need to import the SVM package from sklearn

# In[8]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[9]:


#from sklearn.svm import SVC
#svclassifier = SVC(kernel='linear') #There are types of kernels:polynomial, "RBF"
#svclassifier.fit(x_train.reshape(-1,1),y_train)#Here we directly fit


# In[27]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='linear') #There are types of kernels:polynomial, "RBF"
svclassifier.fit(X_train,y_train)#Here we directly fit


# In[11]:


y_pred = svclassifier.predict(X_test) #Here we predict new values that are outside the trainign data


# We need a way to quantify how it perfomed and this is done already with sklearn

# In[12]:


#print(svclassifier.score(x_test.reshape(-1,1),y_pred))
print(svclassifier.score(X_test,y_pred))


# In[13]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # Feature Selection and SVM-ANOVA (Step 2 variation)

# (This is the first time I'm using it, so I need to understand first what does this thing do) https://scikit-learn.org/stable/auto_examples/svm/plot_svm_anova.html#sphx-glr-auto-examples-svm-plot-svm-anova-py

# For more about feature selection: https://amueller.github.io/aml/05-advanced-topics/12-feature-selection.html

# In[54]:


import numpy as np
from sklearn.datasets import load_iris #This is a standard database in ML for testing

X, y = load_iris(return_X_y=True)
#print(X) It has 4 entries that are relevant for the classification
# Add non-informative features
rng = np.random.RandomState(0)
X = np.hstack((X, 2 * rng.random((X.shape[0], 36)))) #Add 36 more which are only random


# In[55]:


print(X)


# In sklearn you can creat pipelines where you are going to join different models. This can be also used for the NN

# In[56]:


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# In[57]:


# Create a feature-selection transform, a scaler and an instance of SVM that we
# combine together to have a full-blown estimator

clf = Pipeline(
    [
        ("anova", SelectPercentile(f_classif)), #This makes a feature selection
        ("scaler", StandardScaler()), #Normalizes the data, this is important
        #("svc", SVC(gamma="auto")),
        #("svc", SVC(kernel='linear')),
        ("svc", LinearSVC(max_iter=100000,tol=1e-5)),
    ]
)


# In[58]:


import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, X, y)  #This performs training in subsets of our original data then we can see if it's overfitting
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title("Performance of the SVM-Anova varying the percentile of features selected")
plt.xticks(np.linspace(0, 100, 11, endpoint=True))
plt.xlabel("Percentile")
plt.ylabel("Accuracy Score")
plt.axis("tight")
plt.show()


# In[59]:


clf.set_params(anova__percentile=20)
this_scores = cross_val_score(clf, X, y)  #This performs training in subsets of our original data then we can see if it's overfitting
this_scores.mean()
this_scores.std()


# ### Softness $S=wF_n+b$

# In[60]:


clf.fit(X,y)


# In[61]:


print(clf[2])


# In[66]:


w=clf[2].coef_


# In[65]:


b=clf[2].intercept_


# In[74]:


s=clf.decision_function(X)


# In[53]:


#print(s)


# In[ ]:




