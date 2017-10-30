
# coding: utf-8

# # <font color='blue'>Author: Pradeep Sathyamurthy </font>
# # <font color='blue'>Date Started: Oct 15, 2017</font>
# # <font color='blue'>Last Modified Date: Oct 28, 2017</font>
# # <font color='blue'>Topic Focussed: SVM Kernals</font>
# # <font color='blue'>Dataset: Voice Dataset for Gender Regonition from <a href='https://www.kaggle.com/primaryobjects/voicegender'> Kaggle </a></font>

# # Introduction:
# ### <font color='green'> 1. SVM is a kernal trick which can be used for both supervised and unsupervised learning. </font>
# ### <font color='green'> 2. As part of this case study I am going to apply SVM for a supervised learning as I am aware of the class labels to be classified.</font>
# ### <font color='green'> 3. Thus in this notebook I will be using the voice dataset obtained from URL sighted below to classify if the parameters for a particular instances is a male or a female</font>
# 
# 
# # Objective of case study:
# ### <font color='green'> 1. My main objective is to apply SVM and its different kernals and observe how the margin defined helps in improving the classification accuracy</font>
# ### <font color='green'> 2. I will try to tune different parameters in Kernal and choose the best tuning parameter wrt SVM to  classify the dataset </font> 
# ### <font color='green'> 3. I will also apply different classification techniques and compare the results obtained from these with result obtained from SVM classifier </font> 
# 
# # Steps involved in this case study
# ### <font color='red'> 1. Data Manipulation </font>
# ### <font color='red'> 2. Setting a benchmark accuracy for classifiers using Raw Data & Naive Bayes</font>
# ### <font color='red'> 3. Exploratory Data Analysis </font>
# ### <font color='red'> 4. Data Munging and Partition </font>
# ### <font color='red'> 5. Validating the cleaned dataset with benchmark accuracy obtained</font>
# ### <font color='red'> 6. Core Model Building - Applying Different Kernals for SVM </font>
# ####      <font color='brown'>6.1. Linear Kernal SVM </font>
# ####      <font color='brown'>6.2. RBF Kernal SVM </font>
# ####      <font color='brown'>6.3. Polynomial Kernal SVM </font>
# ####      <font color='brown'>6.4. Sigmoidal Kernal SVM </font>
# ### <font color='red'> 7. Perfomance Evaluation on Different Kernals for SVM with 10-fold cross validation </font>
# ####      <font color='brown'>7.1. Evaluation on Linear Kernal SVM </font>
# ####      <font color='brown'>7.2. Evaluation on RBF Kernal SVM </font>
# ####      <font color='brown'>7.3. Evaluation on Polynomial Kernal SVM </font>
# ####      <font color='brown'>7.4. Evaluation on Sigmoidal Kernal SVM </font>
# ### <font color='red'> 8. Parameter tuning on Different Kernals for SVM with 10-fold cross validation </font>
# ####      <font color='brown'>8.1. Tuning on Linear Kernal SVM </font>
# ####      <font color='brown'>8.2. Tuning on RBF Kernal SVM </font>
# ####      <font color='brown'>8.3. Tuning on Polynomial Kernal SVM </font>
# ### <font color='red'> 9. Choosing best Kernals Parameters with grid search</font>
# ### <font color='red'> 10. Visualization of kernal Margin and boundries considereing on two columns meanfun & sp.ent </font>
# ### <font color='red'> 11. Building a Decision Tree </font>
# ### <font color='red'> 12. Building a KNN model </font>
# ### <font color='red'> 13. Comparing individual classifier results</font>
# ### <font color='red'> 14. Ensemble Learning</font>
# ### <font color='red'> 15. Reporting and Discussing final results</font>
# ### <font color='red'> 16. Final Model</font>

# # Dataset URL: 
# 
# http://www.primaryobjects.com/2016/06/22/identifying-the-gender-of-a-voice-using-machine-learning/

# # Importing Packages:

# In[195]:

import pandas as pd # for data handling
import numpy as np # for data manipulation 
import sklearn as sk
from matplotlib import pyplot as plt # for plotting
from sklearn.preprocessing import LabelEncoder # For encoding class variables
from sklearn.model_selection import train_test_split # for train and test split
from sklearn.svm import SVC # to built svm model
from sklearn import svm # inherits other SVM objects
from sklearn import metrics # to calculate classifiers accuracy
from sklearn.model_selection import cross_val_score # to perform cross validation
from sklearn.preprocessing import StandardScaler # to perform standardization
from sklearn.model_selection import GridSearchCV # to perform grid search for all classifiers
from sklearn import tree # to perform decision tree classification
from sklearn import neighbors # to perform knn
from sklearn import naive_bayes # to perform Naive Bayes
from sklearn.metrics import classification_report # produce classifier reports
from sklearn.ensemble import RandomForestClassifier # to perform ensemble bagging - random forest
from sklearn.ensemble import AdaBoostClassifier # to perform ensemble boosting
from sklearn.metrics import roc_curve, auc # to plot ROC Curve
get_ipython().magic('matplotlib inline')


# In[2]:

get_ipython().magic('pwd')


# In[3]:

get_ipython().magic('ls')


# # Step-1: Data Manipulation

# ### <font color='green'>Reading Data: </font>

# In[4]:

# Reding the data as pandas dataframe
data_raw = pd.read_csv('voice.csv',sep=',')
data_raw.shape


# In[5]:

# Verifying if all records are read 
data_raw.head(3)


# In[6]:

# having the headers handy
columns = data_raw.columns
print(columns)


# ### <font color='green'>Data Types of Features: </font>

# In[7]:

# Data type
df = pd.DataFrame(data_raw.dtypes,columns=['Data Type'])
df = df.reset_index()
df.columns = ['Attribute Name','Data Type']
df


# ### <font color='green'>Checking for Missing Values: </font>

# In[9]:

# Checking for any missing values in data and other junk values if any
if data_raw.isnull() is True:
    print('There are missing records')
else:
    print('No missing records')


# ### <font color='green'>Seperating Independent and Target Variables: </font>

# In[46]:

# let us seperate the independent and dependent variables seperately
data_x = data_raw[columns[0:20]].copy()
data_y = data_raw[columns[-1]].copy()
print('Independent var: \n',data_x.head(3),'\n')
print('Dependent var: \n',data_y.head(3))


# ### <font color='green'>Target Variable Encoding: </font>

# In[ ]:

# encoding the target variable from categorical values to binary form
encode_obj = LabelEncoder()
data_y = encode_obj.fit_transform(data_y)
print('sample values of target values:\n',data_y[0:3])


# ### <font color='red'> Inference: </font>
# #### 1. All independent variables are continuous in nature
# #### 2. While the target variables seems binary in nature of typr str
# #### 3. There are totally 3168 rows with 21 columns
# #### 4. There are no missing values in any of the record.

# # Step-2: Setting a benchmark accuracy for classifiers using Raw Data & Naive Bayes 

# In[47]:

# Let us do a 80-20 split
test_x_train,test_x_test,test_y_train,test_y_test = train_test_split(data_x,data_y,train_size=0.8,test_size=0.2,random_state=1)


# In[48]:

nbclf = naive_bayes.GaussianNB()
nbclf = nbclf.fit(test_x_train, test_y_train)
nbpreds_test = nbclf.predict(test_x_test)
print('Accuracy obtained from train-test split on training data is:',nbclf.score(test_x_train, test_y_train))
print('Accuracy obtained from train-test split on testing data is:',nbclf.score(test_x_test, test_y_test))


# In[49]:

test_eval_result = cross_val_score(nbclf, data_x, data_y, cv=10, scoring='accuracy')
print('Accuracy obtained from 10-fold cross validation on actual raw data is:',test_eval_result.mean())


# ### <font color='red'> Inference: </font>
# #### 1. Naive Bayes is a naive method which uses the probablistic theory to classify a target table
# #### 2. Since, it has a fast computation power in training a data and testing it, we can use it as a base method to validate our dataset
# #### 3. Accuracy obtained from this can be set as a bench mark for any classifier that we will start to work going forward
# #### 4. <font color='brown'> Using the raw data and classifying the dataset with Naive implementation with cross validation i obtained an accuracy of 0.85671 </font>
# #### 5. Thus, any data clean up we do further or any classifier model we build should not decrease the accuracy that we obtained here and it must always yeald a high or atleast an accuracy equal to 0.85671, else we will discard the data cleaning done or classifier built to classify the target variable.

# # Step-3: Exploratory Data Analysis (EDA)

# In[50]:

### plotting the independent variables
plt.subplot(221)
plt.hist(data_x['meanfreq'])
plt.subplot(222)
plt.hist(data_x['sd'])
plt.subplot(223)
plt.hist(data_x['median'])
plt.subplot(224)
plt.hist(data_x['Q25'])


# #### 1. Variables meanfreq, sd, median, Q25 are normally distributed

# In[51]:

plt.subplot(221)
plt.hist(data_x['Q75'])
plt.subplot(222)
plt.hist(data_x['IQR'])
plt.subplot(223)
plt.hist(data_x['skew'])
plt.subplot(224)
plt.hist(data_x['kurt'])


# In[52]:

print('Mean and Median value for Q75 is: ',[data_x.Q75.mean(), data_x.Q75.median()])
print('Mean and Median value for IQR is: ',[data_x.IQR.mean(), data_x.IQR.median()])


# #### 1. From above visualization and summary stats we can say Q75 is normally distributed
# #### 2. While IQR, skew and kurt are skewed to right

# In[53]:

plt.subplot(221)
plt.hist(data_x['sp.ent'])
plt.subplot(222)
plt.hist(data_x['sfm'])
plt.subplot(223)
plt.hist(data_x['mode'])
plt.subplot(224)
plt.hist(data_x['centroid'])


# In[54]:

print('Mean and Median value for Mode is: ',[data_x['mode'].mean(), data_x['mode'].median()])


# #### 1. sp.ent, s.fm, centroid are normally distributed
# #### 2. While mode is skewed

# In[55]:

plt.subplot(221)
plt.hist(data_x['meanfun'])
plt.subplot(222)
plt.hist(data_x['minfun'])
plt.subplot(223)
plt.hist(data_x['maxfun'])
plt.subplot(224)
plt.hist(data_x['meandom'])


# #### 1. Variables meanfun is normally distributed
# #### 2. While variables minfun, maxfun, meandom are skewed

# In[56]:

plt.subplot(221)
plt.hist(data_x['mindom'])
plt.subplot(222)
plt.hist(data_x['maxdom'])
plt.subplot(223)
plt.hist(data_x['dfrange'])
plt.subplot(224)
plt.hist(data_x['modindx'])


# #### 1. Variables modindx is normally distributed
# #### 2. While variables mindom, maxdom and dfrange are skewed

# In[57]:

# let us do a descriptive statistics
means = data_x.describe().loc['mean']
medians = data_x.describe().loc['50%']
pd.DataFrame([means,medians], index=['mean','median'])


# In[58]:

# Distribution of target variables
print(pd.Series(data_y).value_counts())
pd.Series(data_y).value_counts().plot(kind='bar', title='Bar graph of Number of male and female users')


# ### <font color='red'> Inference: </font>
# #### 1. Lets explain the skeweness in data from above visualization and summary stats
# #### 2. Irrespectve to viz of histogram, we can also infer those attributes with mean and median values almost equal have gaussian distribution.
# #### 3. Thus, variables meanfreq, sd, median, Q25, Q75, sp.ent, sfm, centroid, meanfun are Normally distributed
# #### 4. Variables skew, kurt, minfun, maxfun, meandom, mindom, maxdom, dfrange, midindex, IQR, mode are skewed
# #### <font color='brown'> 5. Exceptable range of voice freq for a human as per wiki is between 0.085 and 0.255KHz  and hence we will remove any values from the dataset below 0.085 and above 0.255 assuming it to be a outlier based on domain knowledge</font> 
# #### 6. Our target variables (1 = Male and 0 = Female) are symmetrical in nature with equal count of 1584 records for both Male and Female

# # Step-4: Data Munging and Partition

# ### <font color='green'>Data Cleaning: </font>
# 
# #### 1.Exceptable range of voice freq for a human as per wiki is between 0.085 and 0.255KHz and hence we will identify the variable which has this frequncy information and remove them assuming it to be a outlier based on domain knowledge
# #### 2.In our data set meanfun is the variable which have the value of Fundamental frequency
# #### 3. As per the sitation given in  <a href='https://en.wikipedia.org/wiki/Voice_frequency'>wiki </a> we can say that typical adult male will have a fundamental frequency from 85 to 180 Hz and typical adult female from 165 to 255 Hz
# #### 4. Thus, from given dataset, <font color='brown'> we will filter values based on meanfun whose values less than 0.085 and greater than 0.18 for male and values less than 0.165 and greater than 0.255 for female and consider them as outliers and remove them.</font>

# In[59]:

# Actual Raw Data size
data_raw.shape


# In[60]:

# Filtering ouliers from male category
male_funFreq_outlier_index = data_raw[((data_raw['meanfun'] < 0.085) | (data_raw['meanfun'] > 0.180)) & 
                                      (data_raw['label'] == 'male')].index
male_funFreq_outlier_index = list(male_funFreq_outlier_index)
data_raw[((data_raw['meanfun'] < 0.085) | (data_raw['meanfun'] > 0.180)) & (data_raw['label'] == 'male')].shape


# In[61]:

# Filtering ouliers from female category
female_funFreq_outlier_index = data_raw[((data_raw['meanfun'] < 0.165) | (data_raw['meanfun'] > 0.255)) & 
                                        (data_raw['label'] == 'female')].index
female_funFreq_outlier_index = list(female_funFreq_outlier_index)
data_raw[((data_raw['meanfun'] < 0.165) | (data_raw['meanfun'] > 0.255)) & (data_raw['label'] == 'female')].shape


# In[62]:

index_to_remove = male_funFreq_outlier_index + female_funFreq_outlier_index
len(index_to_remove)


# In[63]:

# Thus, we need to remove 710 rows from both data_x and data_y using the index obtained from above filters
# Preparing final dataset for model building
data_x = data_x.drop(index_to_remove,axis=0)
data_x.shape


# In[65]:

# Target dataset
data_y = pd.Series(data_y).drop(index_to_remove,axis=0)
data_y.shape


# In[69]:

# Distribution of target variables after cleanup
print(data_y.value_counts())
data_y.value_counts().plot(kind='bar', title='Target variable after cleanup (1/0=Male/Female)')


# ### <font color='green'>Normalization: </font>
# #### 1. In this dataset meanfreq, median, Q25, Q75, IQR are the only variables associated with unit kHz
# #### 2. let us normalize these variables to make them unit free
# #### 3. we will apply the z-score normalization for meanfreq, median, Q25, Q75
# #### 4. we will apply min-max normalization for IQR

# In[70]:

# Z-score Normalization
z_score_norm = lambda colname: (data_x[colname]- data_x[colname].mean())/(data_x[colname].std())
min_max_norm = lambda colname: (data_x[colname]- data_x[colname].min())/(data_x[colname].max()-data_x[colname].min())


# ### <font color='green'>Creating Partially Normalized Data </font>

# In[72]:

data_x1 = data_x.copy()
data_x1['z_meanfreq'] = z_score_norm('meanfreq')
data_x1['z_median'] = z_score_norm('median')
data_x1['z_Q25'] = z_score_norm('Q25')
data_x1['z_Q75'] = z_score_norm('Q75')
data_x1['Norm_IQR'] = min_max_norm('IQR')


# In[73]:

# Lets now drop the original column from data_x as we have these as backup in data_raw dataframe
data_x1 = data_x1.drop(['meanfreq','median','Q25','Q75','IQR'],axis=1)


# In[74]:

data_x1.head(3)


# In[75]:

# Plotting the normalized columns
# we could see that z-score norm variables have mean 0 and standard deviation 1
# And the min-max norm varibales value are confined between 0-1 and stays positive
plt.subplot(231)
plt.hist(data_x1['z_meanfreq'])
plt.subplot(232)
plt.hist(data_x1['z_median'])
plt.subplot(233)
plt.hist(data_x1['z_Q25'])
plt.subplot(234)
plt.hist(data_x1['z_Q75'])
plt.subplot(235)
plt.hist(data_x1['Norm_IQR'])


# ### <font color='green'>Handling Multicollinearity: </font>

# In[76]:

# let us see the correlation in data
corr_mat = data_x1.corr()
corr_mat


# In[77]:

for names in corr_mat.index:
    if len(corr_mat[(corr_mat.loc[names] > 0.9) & (corr_mat.loc[names].index != names)].index) > 0:
        print('column', names,' correlates strongly with: ',corr_mat[(corr_mat.loc[names] > 0.9) & 
                                                                     (corr_mat.loc[names].index != names)].index)


# In[78]:

corr_df = pd.DataFrame([{'Column Name':'skew', 'Correlated with':'kurt'},
                        {'Column Name':'kurt', 'Correlated with':'skew'},
                        {'Column Name':'centroid', 'Correlated with':['z_meanfreq', 'z_median', 'z_Q25']},
                        {'Column Name':'maxdom', 'Correlated with':['dfrange']},
                        {'Column Name':'dfrange', 'Correlated with':['maxdom']},
                        {'Column Name':'z_meanfreq', 'Correlated with':['centroid', 'z_median', 'z_Q25']},
                        {'Column Name':'z_median', 'Correlated with':['centroid', 'z_meanfreq']},
                        {'Column Name':'z_Q25', 'Correlated with':['centroid', 'z_meanfreq']},
                        ])
corr_df


# In[79]:

# Thus we see high correlation exist between above variables, 
# thus let us create a dataset by removing variables that create high Variance Inflation Factor
# Thus, removing kurt, Centroid, dfrange, z_meanfreq
data_x2 = data_x1.drop(['kurt', 'centroid', 'dfrange', 'z_meanfreq'],axis=1).copy()
data_x2.head(3)


# ### <font color='green'>Creating Completely Normalized Dataset - All columns are normalized </font>

# In[83]:

# let me not do any dimentionality reduction and do z-score normalization on all independent variables
xDataStdardized = StandardScaler()
xDataStdardized.fit(data_x)
data_x3 = xDataStdardized.transform(data_x).copy()


# In[90]:

columns[0:20]


# In[91]:

data_x3 = pd.DataFrame(data_x3, columns=columns[0:20])
data_x3.head(3)


# ### <font color='green'>Data Partition </font>

# In[92]:

# Let us do a 80-20 split on raw dataset
data_x_train,data_x_test,data_y_train,data_y_test = train_test_split(data_x,data_y,train_size=0.8,test_size=0.2,random_state=1)


# In[93]:

# let us do a 80-20 split on dimention reduced dataset too
data_x2_train,data_x2_test,data_y2_train,data_y2_test=train_test_split(data_x2,data_y,train_size=0.8,test_size=0.2,random_state=1)


# In[94]:

# let us do a 80-20 split on raw dataset which was only normalized
data_x3_train,data_x3_test,data_y3_train,data_y3_test=train_test_split(data_x3,data_y,train_size=0.8,test_size=0.2,random_state=1)


# In[95]:

# let us check the size
data_x_train.shape


# In[96]:

data_x_test.shape


# In[97]:

data_y_train.shape


# In[98]:

# let is cross check the size of dimention reduced data set too 
data_x2_train.shape


# In[99]:

data_x2_test.shape


# In[100]:

# let is cross check the size of normalized raw data set too 
data_x3_train.shape


# In[101]:

data_x3_test.shape


# ### <font color='red'> Inference: </font>
# #### 1. I treated the variables with units making them unit free by standardizing them
# #### 2. z-score normalization for meanfreq, median, Q25, Q75 was done
# #### 3. min-max normalization was done for IQR variable
# #### 4. correlation between independent variables was checked to handle the multicollinearity issues
# #### 5. correlation between two variables greater than 0.9 are considered to be heavily coreelated and with respective VIF factor
# #### 6. Variables  kurt, Centroid, dfrange, z_meanfreq was removed from dataset and this was maintained as a whole new dataset
# #### 7. Target variable was converted to numeric male as 1 and female as 0 using sklearn preprocessing pack  n labelencoder object
# #### 8. Data partition was done based on sklearns model_selection package using train_test_split object
# #### 9. Thus I have 4 dataset treated from raw data:
# ###### a.data_x_train
# ###### b.data_x_test
# ###### c.data_y_train
# ###### d.data_y_test
# #### 10. I have 4 dataset treated from raw data and dimentionality reduced:
# ###### a.data_x2_train
# ###### b.data_x2_test
# ###### c.data_y2_train
# ###### d.data_y2_test
# #### 11. I have 4 dataset treated from raw data with all independent variables normalized:
# ###### a.data_x3_train
# ###### b.data_x3_test
# ###### c.data_y3_train
# ###### d.data_y3_test

# # Step-5: Validating the cleaned dataset with benchmark accuracy obtained

# In[102]:

# defining the Naive Bayes object
nbclf = naive_bayes.GaussianNB()


# #### 1. NB Cross Validation on Treated raw dataset

# In[103]:

# lets do a 10 fold Cross validation to make sure the accuracy obtained above
nbclf = nbclf.fit(data_x_train, data_y_train)
nbpreds_test = nbclf.predict(data_x_test)
nb_eval_result1 = cross_val_score(nbclf, data_x, data_y, cv=10, scoring='accuracy')
print('Mean accuracy with 10 fold cross validation on Naive Bayes with treated data: ',nb_eval_result1.mean())


# #### 2. NB Cross Validation on Treated, partially normalized and dimension reduced dataset (This can at times help in building best SVM)

# In[104]:

# lets do a 10 fold Cross validation to make sure the accuracy obtained above
nbclf = nbclf.fit(data_x2_train, data_y2_train)
nbpreds_test = nbclf.predict(data_x2_test)
nb_eval_result2 = cross_val_score(nbclf, data_x2, data_y, cv=10, scoring='accuracy')
print('Mean accuracy with 10 fold cross validation on Naive Bayes with dimention reduced data: ',nb_eval_result2.mean())


# #### 3. NB Cross Validation on Treated and Completely Normalized dataset

# In[105]:

# lets do a 10 fold Cross validation to make sure the accuracy obtained above
nbclf = nbclf.fit(data_x3_train, data_y3_train)
nbpreds_test = nbclf.predict(data_x3_test)
nb_eval_result3 = cross_val_score(nbclf, data_x3, data_y, cv=10, scoring='accuracy')
print('Mean accuracy with 10 fold cross validation on Naive Bayes with Normalized data: ',nb_eval_result3.mean())


# In[207]:

validation_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':nb_eval_result1.mean()},
                                    {'Dataset':'Dimention Reduced', 'Accuracy':nb_eval_result2.mean()},
                                    {'Dataset':'Completely Normalized', 'Accuracy':nb_eval_result3.mean()}], 
                                 columns=['Dataset','Accuracy'])
validation_result


# ### <font color='red'> Inference: </font>
# #### 1. <font color='brown'> Naive bayes classifier after data tretment produce an avg accuracy of 0.95 being the data is normalized or not normalized </font>
# #### 2. <font color='brown'> we see a significant increase in accuracy from 0.85671 to 0.952 after we clean the data </font>
# #### 3. We see the data with dimention reduced and data which are completely normalized works better than raw treated dataset.
# #### 4. However, this can be considered as a base classifier at this point and above result makes sure that our data clean up holds good and we havent removed any influential datas from dataset.
# #### 5. This also set a new benchmark for any complex classifier that will be built further
# #### 6. Thus, accuracy of 0.95 can be set as a bench mark accuracy value for this dataset which is cleaned and processed.
# #### 7. Any model which produce accuracy less than 0.95 can be consodired as a non-efficient model for this dataset from now on

# # Step-6: Core Model Building - Applying Different Kernals for SVM 

# In[111]:

def funct_svm(kernal_type,xTrain,yTrain,xTest,yTest):
    svm_obj=SVC(kernel=kernal_type)
    svm_obj.fit(xTrain,yTrain)
    yPredicted=svm_obj.predict(xTest)
    print('Accuracy Score of',kernal_type,'Kernal SVM is:',metrics.accuracy_score(yTest,yPredicted))
    return metrics.accuracy_score(yTest,yPredicted)


# ### <font color='green'>6.1. Linear Kernal SVM </font>

# In[128]:

# Partially normlized dataset
get_ipython().magic('timeit 10')
PN_linear_result = funct_svm('linear',data_x_train,data_y_train,data_x_test,data_y_test)


# In[129]:

# Dimention reduced dataset
get_ipython().magic('timeit 10')
DR_linear_result = funct_svm('linear',data_x2_train,data_y2_train,data_x2_test,data_y2_test)


# In[130]:

# Completely normalized dataset
get_ipython().magic('timeit 10')
CN_linear_result = funct_svm('linear',data_x3_train,data_y3_train,data_x3_test,data_y3_test)


# In[131]:

linear_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_linear_result},
                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_linear_result},
                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_linear_result}], columns=['Dataset','Accuracy'])
linear_kernal_result


# ### <font color='red'> Inference: </font>
# #### 1. I subjected 3 different dataset as explained above to a linear SVM model and I can observe that dataset which is completely normalize is performing well.
# #### 2. As part of this kernal trick, we have our hyperplane to be linear in a 20-dimentional space
# #### 3. This model <font color='brown'> exhibit a classification accuracy of 0.993902 </font>
# #### 4. Since the data is 20-dimentional, we cannot visualize if the data pocesses a linear or curved relation in feature space, we can take a domain level expertise here. 
# #### 5. However, since we have none for individual analysis purpose we will try to build a model with other kernal tricks types too and see how the model behaves in classifying the gender.

# ### <font color='green'>6.2. RBF Kernal SVM </font>

# In[116]:

# Partially normlized dataset
get_ipython().magic('timeit 10')
PN_rbf_result = funct_svm('rbf',data_x_train,data_y_train,data_x_test,data_y_test)


# In[117]:

# Dimention reduced dataset
get_ipython().magic('timeit 10')
DR_rbf_result = funct_svm('rbf',data_x2_train,data_y2_train,data_x2_test,data_y2_test)


# In[118]:

# Completely normalized dataset
get_ipython().magic('timeit 10')
CN_rbf_result = funct_svm('rbf',data_x3_train,data_y3_train,data_x3_test,data_y3_test)


# In[119]:

gausian_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_rbf_result},
                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_rbf_result},
                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_rbf_result}], columns=['Dataset','Accuracy'])
gausian_kernal_result


# ### <font color='red'> Inference: </font>
# #### 1. RBF or Gaussian is the default kernal which SVM uses in sklearn
# #### 2. Performance of RBF kernal trick is also same as linear kernal SVM
# #### 3. <font color='brown'> I obtained a accuracy of 0.993902 for RBF Kernal using SVM for normalized dataset</font>
# #### 4. This, shows that our voice dataset are both linearly and gaussian seperable

# ### <font color='green'>6.3. Polynomial Kernal SVM </font>

# In[120]:

# Partially normlized dataset
get_ipython().magic('timeit 10')
PN_poly_result = funct_svm('poly',data_x_train,data_y_train,data_x_test,data_y_test)


# In[121]:

# Dimentione reduced dataset
get_ipython().magic('timeit 10')
DR_poly_result = funct_svm('poly',data_x2_train,data_y2_train,data_x2_test,data_y2_test)


# In[122]:

# Completely normalized dataset
get_ipython().magic('timeit 10')
CN_poly_result = funct_svm('poly',data_x3_train,data_y3_train,data_x3_test,data_y3_test)


# In[123]:

poly_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy': PN_poly_result},
                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_poly_result},
                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_poly_result}], columns=['Dataset','Accuracy'])
poly_kernal_result


# ### <font color='red'> Inference: </font>
# #### 1. To acheive much more high accuracy, i tried using polynomial kernal too 
# #### 2. <font color='brown'> I obtained an accuracy of 0.985 for polynomial kernal on normalized dataset </color>
# #### 3. This is comparitively much less than the linear and rbf kernals
# #### 4. However, we cannot conclude this result at this stage as, our training dataset is just one single sample on which we obtained this result.

# ### <font color='green'>6.4. Sigmoidal Kernal SVM </font>

# In[124]:

# Partially normlized dataset
get_ipython().magic('timeit 10')
PN_sigmoid_result = funct_svm('sigmoid',data_x_train,data_y_train,data_x_test,data_y_test)


# In[125]:

# Dimentione reduced dataset
get_ipython().magic('timeit 10')
DR_sigmoid_result = funct_svm('sigmoid',data_x2_train,data_y2_train,data_x2_test,data_y2_test)


# In[126]:

# Completely normalized dataset
get_ipython().magic('timeit 10')
CN_sigmoid_result = funct_svm('sigmoid',data_x3_train,data_y3_train,data_x3_test,data_y3_test)


# In[127]:

sigmoid_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_sigmoid_result},
                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_sigmoid_result},
                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_sigmoid_result}], columns=['Dataset','Accuracy'])
sigmoid_kernal_result


# ### <font color='red'> Inference: </font>
# #### 1. When a dataset is behaving well linearly, it is explicitly known that it doesn't work well in a sigmoidal space
# #### 2. Above result obtained is the evident for this
# #### 3. <font color='brown'> I obtained accuracy of just 0.831 with sigmoidal kernal </font>

# ### <font color='green'>4.5. Consolidated model accuracy </font>

# In[132]:

kernal_result = pd.DataFrame([{'Dataset':'Completely Normalized','Kernal':'Linear', 'Accuracy':CN_linear_result},
                            {'Dataset':'Completely Normalized','Kernal':'Gaussian', 'Accuracy':CN_rbf_result},
                            {'Dataset':'Completely Normalized','Kernal':'Polynomial', 'Accuracy':CN_poly_result}, 
                            {'Dataset':'Completely Normalized','Kernal':'Sigmoidal', 'Accuracy':CN_sigmoid_result}], 
                             columns=['Dataset','Kernal','Accuracy'])
kernal_result


# ### <font color='red'> Inference: </font>
# #### 1. From above table it is clear that a completely normalized dataset behaves well compare to un-normalized dataset
# #### 2. I obtain a maximum accuracy due to the data treatment done, that is treating the meanfun attribute based on biological fact
# #### 3. Maximum accuracy i could acheive is 0.9939 whcih is from Linear and Gaussian Kernal using SVM
# #### 4. While the polinomial and Sigmoidal kernal doesn't seems to classify the target variable accurately and giving a low accuracy of 0.95 and 0.83 for Polynomial and Sigmoidal keransl respectively.
# #### 5. <font color='brown'> However, I cannot blindly accept this accuracy result because this is derived from one sample of training set and validated with a sample test set. In order to evaluate this model to be more robust and to ensure data doesnt overfit, I wanted to subject these model and dataset to a 10-fold cross validation and observe its result as part of next session</font>

# # Step-7: Perfomance Evaluation on Different Kernals for SVM with 10-fold cross validation

# In[138]:

def funct_svm_cv(kernal_type,xData,yData,k,eval_param):
    svm_obj=SVC(kernel=kernal_type)
    eval_result = cross_val_score(svm_obj, xData, yData, cv=k, scoring=eval_param)
    print(eval_param,'of each fold is:',eval_result)
    print('Mean accuracy with 10 fold cross validation for',kernal_type,' kernal SVM is: ',eval_result.mean())
    return eval_result.mean()


# ### <font color='green'>7.1. Evaluation on Linear Kernal SVM  </font>

# In[139]:

# Partially normlized dataset
get_ipython().magic('timeit 10')
PN_CV_linear_result = funct_svm_cv('linear',data_x,data_y,10,'accuracy')


# In[140]:

# Dimentione reduced dataset
get_ipython().magic('timeit 10')
DR_CV_linear_result = funct_svm_cv('linear',data_x2,data_y,10,'accuracy')


# In[141]:

# Completely normalized dataset
get_ipython().magic('timeit 10')
CN_CV_linear_result = funct_svm_cv('linear',data_x3,data_y,10,'accuracy')


# In[142]:

cv_linear_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_CV_linear_result},
                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_CV_linear_result},
                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_CV_linear_result}], columns=['Dataset','Accuracy'])
cv_linear_kernal_result


# ### <font color='red'> Inference: </font>
# #### 1. I see even with 10 fold cross validation, our linear kernal SVM is providing a high accuracy of 0.9939
# #### 2. Thus, I can consider linear Kernal SVM as one of the serious model to subject for further tuning and see if it increases the accuracy
# #### 3. From abov table it is still evident that the completely normalized dataset behaves well comparitively

# ### <font color='green'>7.2. Evaluation on RBF Kernal SVM  </font>

# In[143]:

# Partially normlized dataset
get_ipython().magic('timeit 10')
PN_CV_rbf_result = funct_svm_cv('rbf',data_x,data_y,10,'accuracy')


# In[144]:

# Dimentione reduced dataset
get_ipython().magic('timeit 10')
DR_CV_rbf_result = funct_svm_cv('rbf',data_x2,data_y,10,'accuracy')


# In[145]:

# Completely normalized dataset
get_ipython().magic('timeit 10')
CN_CV_rbf_result = funct_svm_cv('rbf',data_x3,data_y,10,'accuracy')


# In[146]:

cv_rbf_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_CV_rbf_result},
                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_CV_rbf_result},
                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_CV_rbf_result}], columns=['Dataset','Accuracy'])
cv_rbf_kernal_result


# ### <font color='red'> Inference: </font>
# #### 1. From above table, I see a slight decrease in accuracy when I subject Gaussian kernal to 10-fold cross validation
# #### 2. <font color='brown'> With out 80-20 split test set we saw an accuracy of 0.9939 however, with 10-fold CV we obtain accuracy of 0.986</font>
# #### 3. Thus, so far we see linear kernal is behaving well consistently and there is a slight decrese with gaussian kernal

# ### <font color='green'>7.4. Evaluation on Sigmoidal Kernal SVM </font>

# In[147]:

# Partially normlized dataset
get_ipython().magic('timeit 10')
PN_CV_sigmoid_result = funct_svm_cv('sigmoid',data_x,data_y,10,'accuracy')


# In[148]:

# Dimentione reduced dataset
get_ipython().magic('timeit 10')
DR_CV_sigmoid_result = funct_svm_cv('sigmoid',data_x2,data_y,10,'accuracy')


# In[149]:

# Completely normalized dataset
get_ipython().magic('timeit 10')
CN_CV_sigmoid_result = funct_svm_cv('sigmoid',data_x3,data_y,10,'accuracy')


# In[150]:

cv_sigmoid_kernal_result = pd.DataFrame([{'Dataset':'Partially Normalized', 'Accuracy':PN_CV_sigmoid_result},
                                    {'Dataset':'Dimention Reduced', 'Accuracy':DR_CV_sigmoid_result},
                                    {'Dataset':'Completely Normalized', 'Accuracy':CN_CV_sigmoid_result}], columns=['Dataset','Accuracy'])
cv_sigmoid_kernal_result


# ## <font color='red'> Inference: </font>
# #### 1. Like Gaussian kernal, even polynomial and sigmoidal kernals yeald less accuracy with 10 fold CV
# #### 2. I did not include the results of polynomial kernal subjected to 10 fold CV because it was consuming more time to compute
# #### 3. <font color='brown'>However, results of sigmoidal kernal is shown above and we see accuracy is dropped from 0.81 to 0.79</font>

# ### <font color='green'>7.5. Consolidated SVM Kernal Model's Evaluation Result </font>

# In[152]:

cv_kernal_result = pd.DataFrame([{'Dataset':'Completely Normalized','Kernal':'Linear', 'Accuracy':CN_CV_linear_result},
                            {'Dataset':'Completely Normalized','Kernal':'Gaussian', 'Accuracy':CN_CV_rbf_result},
                            {'Dataset':'Completely Normalized','Kernal':'Polynomial', 'Accuracy':CN_poly_result}, 
                            {'Dataset':'Completely Normalized','Kernal':'Sigmoidal', 'Accuracy':CN_CV_sigmoid_result}], 
                             columns=['Dataset','Kernal','Accuracy'])
cv_kernal_result


# ### <font color='red'> Inference: </font>
# #### 1.<font color='brown'> From above table it is clearly evident that Linear SVM Kernal on a completely normalized datset behaves really well</font>
# #### 2. Even with 10-fold cross validation, I obtaned an accuracy of 0.9933927 which seems consistent when compare to other kernals.
# #### 3. After linear kernal it is the Gaussian and Polynomial kernal which gives high accuracy
# #### 4. <font color='brown'>So as part of next session, we will drop Sigmoidal kernal from our further analyis as it doens't even satisfy the bench mark accuracy. </font>
# #### 5. I will take up other 3 SVM models for performance tuning and see how the accuracychanges when we tradeoff between kernal parameters like penalty (C) and gamma in order to obtain a soft margin.

# # Step-8: Parameter tuning on Different Kernals for SVM with 5-fold cross validation - experimenting with margins

# #### From above experimentation we see dataset which was normalized yeald a good result
# #### <font color='brown'>Thus, for further experimentation we will use the dataset whose independent variables are normalized i.e.</font>
# #### <font color='brown'> data_x3 and data_y3 </font>

# In[153]:

# penality parameter C is 1.0 by default in sklearn
# I would like to experiment it with multiple margins in range of c from 1 to 10
def funct_tune_svm(kernal_type,margin_val,xData,yData,k,eval_param):
    if(kernal_type=='linear'):
        svm_obj=SVC(kernel=kernal_type,C=margin_val)
    elif(kernal_type=='rbf'):
        svm_obj=SVC(kernel=kernal_type,gamma=margin_val)
    elif(kernal_type=='poly'):
        svm_obj=SVC(kernel=kernal_type,degree=margin_val) 
    eval_result = cross_val_score(svm_obj, xData, yData, cv=k, scoring=eval_param)
    return eval_result.mean()


# ### <font color='green'>8.1. Tuning on Linear Kernal SVM  </font>

# In[154]:

# Completely normlized dataset
accu_list = list()
for c in np.arange(0.1,10,0.5):
    result = funct_tune_svm('linear',c,data_x3,data_y,5,'accuracy')
    accu_list.append(result)


# In[155]:

C_values=np.arange(0.1,10,0.5)
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,accu_list)
plt.xticks(np.arange(0.1,10,0.5))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')


# In[156]:

tuning_linear_svm = pd.DataFrame(columns=['Penality Parameter C', 'Accuracy'])
tuning_linear_svm['Penality Parameter C'] = np.arange(0.1,10,0.5)
tuning_linear_svm['Accuracy'] = accu_list
tuning_linear_svm


# ### <font color='red'> Inference: </font>
# #### 1. Ultimate aim in building a kernal is to find an optimum hyper plane in feature space which has maximum margin in classifying our target variable.
# #### 2. Kernal which I have built above so far in order to check the performance are those with hard margins, this is not good to be generalized as it may cause overfitting.
# #### 3. So, in this session, we will trade off between margin and Support vectors to choose an optimum boundry which will not overfit the model and at the same time deliver a high accuracy in classifying the target variable.
# #### 4. With linear kernal it is the penalty measure through which we can do some trade off
# #### 5. Above table shows the accuracy (model performance) for different values of C
# #### 6. <font color='brown'> Both from graph and above table we see 0.6 and 1.1 to be the optimum penalty measure or C value which we can treade off with in classifying the target variable.</font>
# #### 7. Even with such trade off , we obtain almost 0.9939 accuracy for linear kernal

# ### <font color='green'>8.2. Tuning on RBF Kernal SVM </font>

# In[157]:

# Completely normlized dataset
accu_list = list()
for c in np.arange(0.1,10,1):
    result = funct_tune_svm('rbf',c,data_x3,data_y,5,'accuracy')
    accu_list.append(result)


# In[158]:

C_values=list(range(0,10))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,accu_list)
plt.xticks(np.arange(0,10,1))
plt.xlabel('Value of Gamma for SVC')
plt.ylabel('Cross-Validated Accuracy')


# In[159]:

tuning_rbf_svm = pd.DataFrame(columns=['Parameter Gamma', 'Accuracy'])
tuning_rbf_svm['Parameter Gamma'] = np.arange(0.1,10,1)
tuning_rbf_svm['Accuracy'] = accu_list


# In[160]:

tuning_rbf_svm


# In[161]:

# Doing further tradeoff
accu_list = list()
for c in np.arange(0.001,0.01,0.001):
    result = funct_tune_svm('rbf',c,data_x3,data_y,5,'accuracy')
    accu_list.append(result)

C_values=list(np.arange(0.001,0.01,0.001))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,accu_list)
plt.xticks(np.arange(0.001,0.01,0.001))
plt.xlabel('Value of Gamma for SVC')
plt.ylabel('Cross-Validated Accuracy')


# In[162]:

tuning_rbf_svm = pd.DataFrame(columns=['Parameter Gamma', 'Accuracy'])
tuning_rbf_svm['Parameter Gamma'] = np.arange(0.001,0.01,0.001)
tuning_rbf_svm['Accuracy'] = accu_list
tuning_rbf_svm


# ### <font color='red'> Inference: </font>
# #### 1. In Gaussian kernal, tradeoff is done with penalty (C) along with gamma parameter
# #### 2. I first experimented with wider Gamma values ranging between 1 and 10 and obsevred Kernal started to behave bad with gamma greater than 1 
# #### 3.<font color='brown'> So, I tried to find the most optimum value with in 0 and 1 and as show in above table, i obtained a maximum accuracy of 0.991 when gammal was equal to 0.03 and 0.05 </font>
# #### 4. However when compare to Linear kernal, we see rbf produce an accuracy of 0.002 times less.
# #### 5. Thus, it is quite evident again that linear kernal acts well on this dataset in classification of target variable.

# ### <font color='green'>8.3. Tuning on Polynomial Kernal SVM  </font>

# In[165]:

# Completely normlized dataset
accu_list = list()
for c in np.arange(0.1,10,1):
    result = funct_tune_svm('poly',c,data_x3,data_y,5,'accuracy')
    accu_list.append(result)


# In[166]:

np.arange(0.1,10,1)


# In[168]:

C_values=list(np.arange(0.1,10,1))
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(C_values,accu_list)
plt.xticks(np.arange(0.1,10,1))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')


# In[170]:

tuning_poly_svm = pd.DataFrame(columns=['Parameter Degree', 'Accuracy'])
tuning_poly_svm['Parameter Degree'] = np.arange(0.1,10,1)
tuning_poly_svm['Accuracy'] = accu_list
tuning_poly_svm


# ### <font color='red'> Inference: </font>
# #### 1. Along with penalty and gamma parameter, with polynomial kernal we can trade off with degree 
# #### 2. I experimented with various degree as shown above and obtained degree = 1.1 produce a high accuracy
# #### 3. <font color='brown'> Accuracy obtained by polynomial is almost same as Linear which is 0.993 </font>
# #### 4. So, to produce a final inference in choosing the best kernal we will apply a grid search in our next session and see which model and which parameter produce a high accuracy.

# # Step-9: Choosing best Kernals Parameters with grid search

# In[171]:

# Now performing SVM by taking hyperparameter C=0.1 and kernel as linear
svc=SVC(kernel='linear',C=0.6)
scores = cross_val_score(svc, data_x3, data_y, cv=10, scoring='accuracy')
print(scores.mean())


# In[172]:

# With rbf gamma value = 0.01
svc= SVC(kernel='rbf',gamma=0.005)
svc.fit(data_x3_train,data_y3_train)
y_predict=svc.predict(data_x3_test)
metrics.accuracy_score(data_y3_test,y_predict)


# In[174]:

np.arange(0.001,0.0,0.001)


# ### <font color='green'>9.1. Choosing the best parameter </font>

# In[175]:

# performing grid search with different tuning parameters
svm_obj= SVC()
grid_parameters = {
 'C': [0.1,0.6,1.1,1.6] , 'kernel': ['linear'],
 'C': [0.1,0.6,1.1,1.6] , 'gamma': [0.002,0.003,0.004,0.005], 'kernel': ['rbf'],
 'degree': [1,2,3] ,'gamma':[0.002,0.003,0.004,0.005], 'C':[0.1,0.6,1.1,1.6] , 'kernel':['poly']
                   }
model_svm = GridSearchCV(svm_obj, grid_parameters,cv=10,scoring='accuracy')
model_svm.fit(data_x3_train, data_y3_train)
print(model_svm.best_score_)
print(model_svm.best_params_)
y_pred= model_svm.predict(data_x3_test)


# In[176]:

svm_performance = metrics.accuracy_score(y_pred,data_y3_test)
svm_performance


# In[177]:

gridSearch_kernal_result = pd.DataFrame([{'kernel': 'poly', 'gamma': 0.005, 'degree': 1, 'C': 1.6}],
                                       columns=['kernel','C','gamma','degree'])
gridSearch_kernal_result


# ### <font color='red'> Inference: </font>
# #### 1. I did a grid search, whcih is a structure way to obtain an optimized kernal and its parameter measures
# #### 2. <font color='brown'>  From above result, I see it is the polynomial kernal with penalty measure of C=1.6 and gamma = 0.005 and with degree=1 produce a high accuracy of 0.9939 in classifying the target variable.</font>
# #### 3. In this next session i have tried to visualize my margin and kernal behaviour by subjecting only 2 columns for analysis as it becomes a 2-dimentional space for visualization.

# # Step-10 Visualization of kernal Margin and boundries considereing only two columns meanfun & sp.ent to represent a 2D space

# ### <font color='green'>10.1. Choosing the best attribute to represent dataset in 2D space </font>

# In[178]:

# Scatter plot with strong correlation - not useful much to represnt the distribution wrt kernal boundries
plt.scatter(data_raw['meanfreq'],data_raw['centroid'])


# In[179]:

# Scatter plot with weak correlation - not useful much to represnt the distribution wrt kernal boundries
plt.scatter(data_raw['modindx'],data_raw['minfun'])


# In[181]:

# Scatter plot with moderate correlation - useful much to represnt the distribution wrt kernal boundries
plt.scatter(data_raw['dfrange'],data_raw['centroid'])


# In[180]:

# Scatter plot with moderate negative correlation - useful much to represnt the distribution wrt kernal boundries
plt.scatter(data_raw['meanfun'],data_raw['sp.ent'])


# ### <font color='red'> Inference: </font>
# #### 1. After doing necessary data cleanup and model building I was able to infer that a polinomial kernal SVM with parameters C=1.6, gamma=0.005 and degree=1 plots a perfect margin in a high dimentional space to classify gender label which is our target variable
# #### 2.However, vizualizing more than two dimention is complex to represnt
# #### 3. <font color='brown'> So, I would like to choose any 2 variables from dataset through which i can represnt my margin and kernal boundries in a 2-dimentional space </font>
# #### 4. <font color='brown'>  For this i used the correlation matrix and above scatter plot obtained above and choose two variable which is moderately correlated. As neither the strong nor the weak correlation variables might not be well represented in ourder to show the decision boundries.</font>
# #### 5. meanfun being the most important variable for the dataset, I decided to choose it and match it with another variable which has moderate correlation with it. with 0.52 as correlation value between i choose sp.ent and meanfun to be my choise of 2-dimentional feature space.

# ### <font color='green'>10.2. Visualizing the margin modeled </font>

# In[185]:

# import some data to play with
X = data_x3[['meanfun','sp.ent']].copy()
X = np.array(X)
y = np.array(data_y)

# fit the model, don't regularize for illustration purposes
clf = SVC(kernel='poly', degree=1.1, gamma = 0.05,C=1.6)
clf.fit(X, y)

# title for the plots
title = ('SVC with poly kernel(with degree=1.1 & gamma=0.05 & C=1.6)')

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none')
ax.set_xlabel('meanfun')
ax.set_ylabel('sp.ent')
ax.set_title(title)
plt.show()


# ### <font color='red'> Inference: </font>
# #### 1. meanfun being the most important variable for the dataset, I decided to choose it and match it with another variable which has moderate correlation with it. with 0.52 as correlation value between i choose sp.ent and meanfun to be my choise of 2-dimentional feature space.
# #### 2. I modeled polynomial kernal with penalty measure of C=1.6, gamma = 0.05 and degree=1 to obtain the above scatter plot.
# #### 3. When did, SVM projected my data in a 2 dimentional space and obtained an optimal margin that classifies my gender being male and female. 
# #### 4. <font color='brown'> From the above figure, we can infer: </font>
# ####           1. <font color='brown'> Orage points = Instance which are Male </font>
# ####           2. <font color='brown'> Blue Points = Instance which are Female </font>
# ####           3. <font color='brown'> Circled Points = Support Vectors used to obtain margin </font>
# ####           4. <font color='brown'> Straingh Line = Hard Margin </font>
# ####           5. <font color='brown'> Dotted Lines = Soft Margin (with trade off being C=1.6, gamma=0.05 and degree=1) </font>
# #### 5. With respective to only these two variables, meanfun and sp.ent, It is so evident that our model is not being overfit as it gives a clear distinction between two classes 'Male' and ' Female' with no complications in margins. Thus, accuracy of 0.99 can be considered to be valid enough at this point. However, this is just the visualization about margins, we will not visualize how the SVM boundy is placed in a for all our parameters in a 2D space.

# ### <font color='green'>10.3. Visualizing the Kernal boundaries </font>

# In[255]:

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# import some data to play with
X = data_x3[['meanfun','sp.ent']].copy()
X = np.array(X)
y = np.array(data_y)

C = 1.6  # SVM regularization parameter
models = (SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          SVC(kernel='rbf', gamma=0.005, C=C),
          SVC(kernel='poly', degree=1, gamma=0.005, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel (C=1.6)',
          'LinearSVC (linear kernel)',
          'RBF kernel(gamma=0.005)',
          'Polynomial (degree 1)')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('meanfun')
    ax.set_ylabel('sp.ent')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()


# ### <font color='red'> Inference: </font>
# #### 1.I still consider meanfun and sp.ent to be my favorite variables to visualize my kernal boundries in a 2D space.
# #### 2. I modeled polynomial kernal with same parameters penalty measure of C=1.6, gamma = 0.05 and degree=1 to obtain the above scatter plot.
# #### 3. When did, SVM projected my data in a 2 dimentional space and obtained above feature space with boundries that classifies  gender being male and female. 
# #### 4. <font color='brown'> From the above figure, we can infer: </font>
# ####           1. <font color='brown'> Linear kernal with c=1.6 have a strict boundry </font>
# ####           2. <font color='brown'> While in RBF kernal, the boundry is strict and also have some points misclassified </font>
# ####           3. <font color='brown'> Polynomial kernal have a lineant boundry which are discriminative </font>
# ####           4. <font color='brown'> From above figure, we dont see any complex boundries for polynomial and hence we need not worry about the model being over fitting </font>
# #### 5. With respective to only these two variables, meanfun and sp.ent, It is so evident that our model is not being overfit as it gives a clear distinction between two classes 'Male' and ' Female' with no complications in margins in a feature space. 
# #### 6. Thus, accuracy of 0.993 produced by Polynomial kernal can be considered to be valid enough, this means 7 out of 1000 times ther could be a misclassification. Let is see if we can minimize this error occurence by increasing the accuracy further using few ensemble learnings.

# # Step-11: Building a Decision Tree Classifier with grid search

# In[190]:

dt = tree.DecisionTreeClassifier()
parameters = {
    'criterion': ['entropy','gini'],
    'max_depth': np.linspace(1, 20, 10),
    #'min_samples_leaf': np.linspace(1, 30, 15),
    #'min_samples_split': np.linspace(2, 20, 10)
}
gs = GridSearchCV(dt, parameters, verbose=0, cv=5)
gs.fit(data_x3_train, data_y3_train)
gs.best_params_, gs.best_score_


# In[191]:

def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred = clf.predict(X)   
    if show_accuracy:
         print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred),"\n")
      
    if show_confussion_matrix:
        print("Confussion matrix")
        print(metrics.confusion_matrix(y, y_pred),"\n")


# In[192]:

dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)
dt.fit(data_x3_train, data_y3_train)
measure_performance(data_x3_test, data_y3_test, dt, show_confussion_matrix=False, show_classification_report=True)


# In[193]:

dt_performance = dt.score(data_x3_test, data_y3_test)
dt_performance


# In[261]:

# lets do a 10 fold Cross validation to make sure the accuracy obtained above
dt_eval_result = cross_val_score(dt, data_x3, data_y, cv=10, scoring='accuracy')
print('Mean accuracy with 10 fold cross validation for Decision tree is: ',dt_eval_result.mean())


# ### <font color='red'> Inference: </font>
# #### 1. <font color='brown'>I see accuracy yealded by decision tree is 0.9894 which is less when compare to SVM classifier which was 0.993</font>
# #### 2. We can say compare to decision tree SVM model seems more efficient
# #### 3. So, if scrutability is the requirement based on which a model needs to be built we can go ahead with decision tree model.

# # Step-12: Building a KNN with 5 nearest neighbors

# In[198]:

n_neighbors = 5
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(data_x3_train, data_y3_train)


# In[199]:

knnpreds_test = knnclf.predict(data_x3_test)


# In[202]:

print(knnclf.score(data_x3_test, data_y3_test))


# In[200]:

print(classification_report(data_y3_test, knnpreds_test))


# In[203]:

knn_performance = knnclf.score(data_x3_test, data_y3_test)


# In[264]:

# lets do a 10 fold Cross validation to make sure the accuracy obtained above
knn_eval_result = cross_val_score(knnclf, data_x3, data_y, cv=10, scoring='accuracy')
print('Mean accuracy with 10 fold cross validation for KNN is: ',knn_eval_result.mean())


# ### <font color='red'> Inference: </font>
# #### 1. KNN yealds an accuracy of 0.977 which is comparitive less to SVM
# #### 2. However, its accuracy touches the benchmark of 0.95 which we decided based on Naive Bayes, we can have this model for any ensemble building, etc., and it not advisable to just discard it.
# #### 2. Though KNN perform better than Naive Bayes, its accuracy is less compare to SVM

# # 13. Comparing individual classifier results

# In[204]:

final_resutls = pd.DataFrame(columns=['Classifier Name', 'Performance in terms of Accuracy'])


# In[267]:

final_resutls['Classifier Name'] = ['SVM','Decision Tree','KNN','Naive Bayes']
final_resutls['Performance in terms of Accuracy'] = [svm_performance, dt_eval_result.mean(), 
                                                     knn_eval_result.mean(),nb_eval_result.mean()]


# In[268]:

final_resutls


# In[269]:

final_resutls.plot.line(x=final_resutls['Classifier Name'])


# ### <font color='red'> Inference: </font>
# #### 1. From above table and graph it seems very clear that, SVM with polynomial kernal behaves best.
# #### 2. Accuracy produces by Polynomial kernal equal to 0.993 is the highest of all cross validation results obtained from other classifiers.
# #### 3. Thus, with individual classifiers we can infer that as a individual classifier, SVM with Polynomial Kernal does a best classification wrt his voice dataset in classifying an instance as Male or Female
# #### 4. This SVM polynomial kernal tend to miss classify only 7 out of 1000 times when subjected to such dataset which is pretty good.
# #### 5. However, we will yet try to improve the accuracy further using some ensemble techniques.

# # 14. Ensemble Learning

# ### <font color='green'>14.1. Bagging with Random Forest </font>

# In[237]:

# Applying Random forest to improve the decision tree model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy',max_depth=7)
rf_model = rf.fit(data_x3_train, data_y3_train)


# In[241]:

rfpreds_test = rf_model.predict(data_x3_test)
rf_performance = rf_model.score(data_x3_test, data_y3_test)


# In[242]:

print(rf_performance)


# In[270]:

# lets do a 10 fold Cross validation to make sure the accuracy obtained above
rf_eval_result = cross_val_score(rf_model, data_x3, data_y, cv=10, scoring='accuracy')
print('Mean accuracy with 10 fold cross validation for KNN is: ',rf_eval_result.mean())


# ### <font color='green'>14.2. Boosting with Random Forest </font>

# In[233]:

# adaboost
adaBoost = AdaBoostClassifier()
adaboost_model = adaBoost.fit(data_x3_train, data_y3_train)


# In[243]:

adboostpreds_test = adaboost_model.predict(data_x3_test)
adaboost_performance = adaboost_model.score(data_x3_test, data_y3_test)


# In[244]:

print(adaboost_performance)


# In[271]:

# lets do a 10 fold Cross validation to make sure the accuracy obtained above
adaboost_eval_result = cross_val_score(adaboost_model, data_x3, data_y, cv=10, scoring='accuracy')
print('Mean accuracy with 10 fold cross validation for KNN is: ',adaboost_eval_result.mean())


# # 15. Reporting and Discussing the final results

# In[273]:

final_report = pd.DataFrame(columns=['Classifier Name', 'Performance in terms of Accuracy'])
final_report['Classifier Name'] = ['SVM','AdaBoost','Random Forest','Decision Tree','KNN','Naive Bayes']
final_report['Performance in terms of Accuracy'] = [svm_performance, adaboost_eval_result.mean(),rf_eval_result.mean(),
                                                    dt_eval_result.mean(),
                                                    knn_eval_result.mean(),nb_eval_result.mean()]
final_report


# In[274]:

final_report.plot.line(x=final_report['Classifier Name'])


# # 16. Final Model

# In[204]:

# Building the ROC Curve for the final SVM Kernal model
final_model = SVC(kernel='poly', C=1.6, gamma=0.005, degree=1)
print('Final Model Detail:\n',final_model)
final_model_score = svm_classifier.fit(data_x3_train, data_y3_train).decision_function(data_x3_test)
# CV Accuracy 
final_eval_result = cross_val_score(final_model, data_x3, data_y, cv=10, scoring='accuracy')
print('\nAccuracy obtained from final model with 10 fold CV:\n',final_eval_result.mean())
# ROC measure
fpr, tpr, _ = roc_curve(data_y3_test,final_model_score)
roc_auc= auc(fpr, tpr)
print('\nROC Computed Area Under Curve:\n',roc_auc)


# In[205]:

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for Best SVM model')
plt.legend(loc="lower right")
plt.show()


# ### <font color='red'> Inference: </font>
# #### 1. From above table and graph it seems very clear that, SVM with polynomial kernal behaves best.
# #### 2. Accuracy produces by Polynomial kernal equal to 0.993 is the highest of all cross validation results obtained from other classifiers.
# #### 3. Thus, with individual classifiers we can infer that as a individual classifier, SVM with Polynomial Kernal does a best classification wrt his voice dataset in classifying an instance as Male or Female
# #### 4. This SVM polynomial kernal tend to miss classify only 7 out of 1000 times when subjected to such dataset which is pretty good.
# #### 5. However, we will yet try to improve the accuracy further using some ensemble techniques.

# ##                    -------------------------------- End of the Book -------------------------------------------
