#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
from sklearn.svm import SVC
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[106]:


face=pd.read_csv('emotion.csv')
print(face)


# In[ ]:





# In[107]:


face.head()


# In[108]:


face.info()


# In[109]:


face.describe()


# In[110]:


face.isnull().sum()


# In[111]:


face.shape


# In[ ]:





# In[112]:


x=face['emotion'].values
y=face['age'].values
x
y


# In[113]:


plt.scatter(x,y,color='blue',label='scatterplot')
plt.title("facial emotion recognization")
plt.xlabel('emotion')
plt.ylabel('age')
plt.legend(loc=4)
plt.show()
print(x.shape)
print(y.shape)


# In[114]:


x=x.reshape(-1,1)
y=y.reshape(-1,1)
print(x.shape)
print(y.shape)


# In[115]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)


# In[116]:


from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)


# In[117]:


# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the classifier on the training data
svm_classifier.fit(x_train, y_train)

# Make predictions on the test data
y_pred = svm_classifier.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[118]:


prediction = classifier.predict(x_test) #And finally, we predict our data test.


# In[119]:


class_names=np.array(['0','1']) 


# In[120]:


#accuracy on training data
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)


# In[121]:


print('Accuracy on Training data:',training_data_accuracy)


# In[122]:


# accuracy on test data
x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)


# In[123]:


print('Accuracy score on Test Data:',test_data_accuracy)

