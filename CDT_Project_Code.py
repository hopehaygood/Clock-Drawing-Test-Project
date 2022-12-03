#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Running the libraries
from pdf2image import convert_from_path # for converting PDF to JPG
from matplotlib.image import imread
import numpy as np
import pandas as pd
import scipy as sc
import os
import matplotlib.pyplot as plt
from pylab import *
from skimage.transform import resize  # for resizing the images 
import statistics as stats

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, f1_score, precision_score
from keras.models import load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.models import Sequential
from pdf2image.exceptions import (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError)

# Packages for ANOVA Table
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
from scipy.stats import f_oneway
from scipy.stats import levene

#Chi-Square
from bioinfokit.analys import stat, get_data

# Fisher-Exact Test
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()

# KNN
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

# CNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

# K-Means
from sklearn.cluster import KMeans


# We will import the dataset containg the FEV1, GOLD Stage, COPD Score, & Adjusted Clock Drawing Test Score for 1,716 unique Subject IDs.

# In[16]:


Dataset_1 = pd.read_csv('C:/Users/12563/Documents/Res. Methods in Math & Stats/Data_Final.csv')
Dataset_1


# In[17]:


# Rename column in a dataframe
Dataset_1.rename(columns = {'sid':'Clocks' }, inplace = True)


# We will perform the K-Nearest Neighbors Algorithm to solve the classification problem of weather or not a particular person (Subject ID) has COPD or not.

# In[4]:


df_knn = pd.DataFrame(Dataset_1, columns = [ 'fev1pp_final', 'goldstage_final_V2', 'COPD_P3','Score'])
df_knn


# In[6]:


X_1 = df_knn.drop("COPD_P3", axis = 1)
X_1 = X_1.values
y_1 = df_knn['COPD_P3']
y_1 = y_1.values


# In[7]:


# Split the data into train and test sets
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size= 0.2, train_size=0.8, random_state=33)


# In[8]:


# KNN is a supervised machine learning method
# Trying to predict weather or not someone will have COPD based
# on their CDT Score (the predictor)
knn_model_1 = KNeighborsRegressor(n_neighbors=2)
knn_model_1.fit(X_train_1, y_train_1)


# In[9]:


# Calculate RMSE for Training Data 
train_preds_1 = knn_model_1.predict(X_train_1)
mse_1 = mean_squared_error(y_train_1, train_preds_1)
rmse_1 = sqrt(mse_1)
rmse_1


# In[10]:


# Calculate RMSE for Testing Data
test_preds_1_t = knn_model_1.predict(X_test_1)
mse_1_t = mean_squared_error(y_test_1, test_preds_1_t)
rmse_1_test = sqrt(mse_1_t)
rmse_1_test


# In[11]:


# Calculate Accurary
y_pred_1 = knn_model_1.predict(X_test_1)

# Evaluate the training & testing Accuracy
#
print('Training accuracy score: %.3f' % knn_model_1.score(X_train_1, y_train_1))
print('Test accuracy score: %.3f' % knn_model_1.score(X_test_1, y_test_1))


# Let us first import the Excel file containg 19,485 unqiue Subject IDs (SIDs) along with the assocaited Adjusted Clock Drawing Test Score (either 0 (High-Risk for Dementia) or 2 (not at High-Risk for Dementia)).

# In[12]:


# Reading clock scores from CSV score file

df = pd.read_csv('C:/Users/12563/Desktop/CDT/Adjudicated Mini-Cog Clock Scores.csv', na_values =  'NaN')
scores1 = df.shape
df = df[df['Adjudicated Clock Score'].notnull()]
scores2 = df.shape
df['Adjudicated Clock Score'] = df['Adjudicated Clock Score'].apply(lambda x: 2 if x == '2 Points' else 0)
scores3 = df.shape


# Now, we will create the Clock_List data frame that contains the SIDs of the Clock Drawing Test Images with the corresponding Adjusted Clock Score from the Excel file.

# In[13]:


# Reading the image files from the source folder
path = r'C:/Users/12563/Documents/Test Clocks 2/'
files = os.listdir(path)


# The code below extracts the patient_id from the image names and match it with the CSV file, and find the score and 
# add it to the file name (format : score_patientID) 
scores = []
name_sel = []

nummatch = 0
numdiscard = 0

for i in files:
    
    # Depending on the image file names different split patterns are needed.
    f_name_prim = os.path.basename(i).split('.')[0].split(' ')[0].split('_')[0]
#     f_name_prim = os.path.basename(i).split('.')[0].split(' ')[0].split('_')[0]
#     f_name_prim = os.path.basename(i).split(' ')[1].split('.')[0]
#     f_name_prim = os.path.basename(i).split(' ')[1].split('-')[1].split('.')[0]
#     f_name_prim = os.path.basename(i).split(' ')[1].split('#')[1].split('.')[0]
#     f_name_prim = os.path.basename(i).split(' ')[1].split('.')[0]
#     f_name_prim = os.path.basename(i).split(' ')[0].split('_')[1]
#     f_name_prim = os.path.basename(i).split(' ')[1].split('.')[0]
#     f_name_prim = os.path.basename(i).split('_')[1].split('_')[0]
    
    for k in range(df.shape[0]):
        if f_name_prim  == df.iloc[k,0]:
            nummatch += 1
            scores.append(df.iloc[k,1])
            name_sel.append(f_name_prim )
            
#             os.rename(os.path.join(path, i), os.path.join(path, ''.join(["S" + str(df.iloc[k,1]) + '_' + f_name_prim + '.PDF'])))
        else:
            numdiscard += 1


Clock_list = pd.DataFrame([name_sel, scores], index = ['Clocks', 'Scores'])
Clock_list = Clock_list.T

# saving the results in the target folder
Clock_list.to_csv(path + 'Clock_list.csv') 
Clock_list


# Furthermore, we will create a new data frame that will merge the Clock_List Data Frame with the excel file that contains the assoicated FEV1 Score, GOLD Stage, & COPD Score for 370 of the the SIDs in which we have the Clock Drawing Test image & the associated score. 

# In[18]:


Dataset_2 = Clock_list.merge(Dataset_1,on = 'Clocks')
Dataset_2


# We will now perform Statistical Analysis on Dataset 2, this Data Frame created by Merging Dataset 1 & Clock_List.

# We begin the statistical Analysis by performing the Chi-Squared Test for Independence on the Adjusted Clock Drawing Test Score by the Self-Reported COPD Score

# In[20]:


# Chi-Square Test

df_use_2_copd = pd.DataFrame(Dataset_2, columns = ['Score', 'COPD_P3']) 
df_use_2_copd


# In[22]:


# create contingency table
data_crosstab_2_copd = pd.crosstab(df_use_2_copd['Score'],
                            df_use_2_copd['COPD_P3'],
                           )

data_crosstab_2_copd


# In[23]:


# Chi-Squared Test

res_2 = stat()
res_2.chisq(df=data_crosstab_2_copd)
# output
print(res_2.summary)


# We will now observe the Expected Frequency Counts for the Chi-Squared Test for Independence executed on the Adjusted Clock Drawing Test Score by the Self-Reported COPD Score variable.

# In[24]:


print(res_2.expected_df)

# Need at least 5, preferably 15
# Assumptions met.


# Chi-Squared Test for Independence on the Adjusted Clock Drawing Test Score by the Final GOLD Baseline Variable

# In[28]:


# Chi-Square Test

df_use_2_gold = pd.DataFrame(Dataset_2, columns = ['Score', 'goldstage_final_V2']) 
df_use_2_gold


# In[30]:


# create contingency table
data_crosstab_2_gold = pd.crosstab(df_use_2_gold['Score'],
                            df_use_2_gold['goldstage_final_V2'],
                           )

data_crosstab_2_gold


# In[31]:


# Chi-Squared Test

res_2_gold = stat()
res_2_gold.chisq(df=data_crosstab_2_gold)
# output
print(res_2_gold.summary)


# Finally, we will perform the K-Nearest Neighbors Algorithm to solve the classification problem of a patients Clock Drawing Test Score.

# In[38]:


df_knn_2 = pd.DataFrame(Dataset_2, columns = ['fev1pp_final', 'goldstage_final_V2', 'COPD_P3', 'Score'])
df_knn_2


# In[45]:


X_2 = df_knn_2.drop("Score", axis = 1)
X_2 = X_2.values
y_2 = df_knn_2['Score']
y_2 = y_2.values


# In[46]:


# Split the data into train and test sets
X_train_2_score, X_test_2_score, y_train_2_score, y_test_2_score = train_test_split(X_2, y_2, test_size= 0.2, train_size=0.8, random_state=33)


# In[47]:


# KNN is a supervised machine learning method
# Trying to predict weather or not someone will have COPD based
# on their CDT Score (the predictor)
knn_model_2_score = KNeighborsRegressor(n_neighbors=2)
knn_model_2_score.fit(X_train_2_score, y_train_2_score)


# In[48]:


# Calculate RMSE for Training Data 
train_preds_2_score = knn_model_2_score.predict(X_train_2_score)
mse_2_score = mean_squared_error(y_train_2_score, train_preds_2_score)
rmse_2_score = sqrt(mse_2_score)
rmse_2_score


# In[49]:


# Calculate RMSE for Testing Data
test_preds_2_test_s = knn_model_2_score.predict(X_test_2_score)
mse_2_test_s = mean_squared_error(y_test_2_score, test_preds_2_test_s)
rmse_2_test_s = sqrt(mse_2_test_s)
rmse_2_test_s


# In[50]:


# Calculate Accurary
y_pred_2_s = knn_model_2_score.predict(X_test_2_score)

# Evaluate the training & testing Accuracy
#
print('Training accuracy score: %.3f' % knn_model_2_score.score(X_train_2_score, y_train_2_score))
print('Test accuracy score: %.3f' % knn_model_2_score.score(X_test_2_score, y_test_2_score))


# Analysis of the Factor Variables on the Adjusted Clock Drawing Test Score

# Now, we will pixelate the 565 Clock Drawing Test images

# In[57]:


# pixalating PDF clock images and convering them to data matrix

# Reading in Images with an Adjusted Clock Drawing Test Score 
# of 0

path = r'C:/Users/12563/Documents/Test Clocks 2/'
#path = r'C:/Users/12563/Documents/Clocks_0/'
files = os.listdir(path)
dmat1 = np.zeros((len(files), 288, 288))

for index, file in enumerate(files):
    f_name_prim = os.path.basename(file)
    print(f_name_prim)
    print(path)
    print(path+f_name_prim)
    
    _, ext = os.path.splitext(file)
    if ext.lower() == '.pdf':
        # pdf processing
        #print(file)
        images = convert_from_path(path + f_name_prim, poppler_path = r"C:\Users\12563\Desktop\poppler-0.68.0_x86 (1)\poppler-0.68.0\bin")
        image = np.array(images[0])
        y,x,_ = image.shape
        
        # cropping the clocks from the images
        #print(image.shape)
         # saving the clocks in the target folder
        #fig, ax = plt.subplots()    
        #plt.title('File name: {}, Index: {}'.format(file, index))
        #ax.imshow(image, cmap='gray')
        #Cropping
        # I decided on the cropping bounds by looking at printed images
        image = image[350:1250,350:1350] 
        #print(image.shape)
         # saving the clocks in the target folder
        fig, ax = plt.subplots()    
        plt.title('File name: {}, Index: {}'.format(file, index))
        ax.imshow(image, cmap='gray')
        print()
        
        # Grayscaling the images
        image = np.mean(image, axis = 2)

        # resizing the cropped clocks
        image_resized = resize(image, (288, 288))

        dmat1[index] = image_resized

        # saving the clocks into the target folder
        fig, ax = plt.subplots() 
        plt.title('File name: {}, Index: {}'.format(file, index))
        ax.imshow(image_resized, cmap='gray')
        #fig.savefig(path + f_name_prim + '.jpeg') 

        # saving the datamatrix in the target folder
        #np.save(path + 'dmat2', dmat2) 

    elif ext.lower() == '.jpeg':
        # jpeg processing
        
  
        image = imread(path + f_name_prim)

        # cropping the clocks from the images
        image = image[550:2400, 550:2400] 

        # Grayscaling the images
        image = np.mean(image, axis = 2)

        # resizing the cropped clocks
        image_resized = resize(image, (288, 288))

        dmat1[index] = image_resized

        # saving the clocks in the target folder
        fig, ax = plt.subplots()    
        plt.title('File name: {}, Index: {}'.format(file, index))
        ax.imshow(image, cmap='gray')
        
        #fig.savefig(path + f_name_prim)  
    #else:
        #print(file, ext)
        
        #if file == "JPEG":
            #continue
        #print(file)
        #images = convert_from_path(path + f_name_prim, poppler_path = r"C:\Users\12563\Desktop\poppler-0.68.0_x86 (1)\poppler-0.68.0\bin")
        #image = np.array(images[0])
        #y,x,_ = image.shape


# Now, we will rotate the images in need of rotation as well as delete images that need to be deleted.

# In[62]:


# This code removes the noise and rotates the images

# Images that need rotation (Indexes via printing in the cell above)
S_rot = [31, 216, 445, 554]

# Noise images that will be removed, Clock not shown on images
S_noise = [4, 12, 14, 15, 27,33, 35, 41, 44, 47, 51, 52, 93, 103, 125, 137, 157, 160, 162, 176, 185, 188, 233, 240, 249,
           251, 259, 270, 271, 274, 278, 281, 285, 287, 288, 293, 296, 304, 322, 325, 326, 331, 340, 360, 362, 418, 442,
          449, 450, 451, 467, 468, 526, 559, 561]
        
#S_rot = []
#S_noise= [4, 12, 13, 34, 46, 65]
# function for rotating the JPG images
           
def img_rotate (df, rotate) :
    for i in rotate:
        df[i] = np.rot90(df[i],3)
    return(df)

dmat1 = img_rotate (dmat1, S_rot)


# removing noise images
dmat1_nonoise = np.delete(dmat1, S_noise , axis = 0)


# In[63]:


# Loading X(clock matrix) and Y (labels) data (already suffeled and noises removed)
X = np.load('/Users/12563/Desktop/Batch 1 (2)/Xall_nonoi_shfl_use_0.npy')
y = np.load('/Users/12563/Desktop/Batch 1 (2)/Yall_nonoi_shfl_use_0.npy')


# In[64]:


# Creating array of the files to create
files = np.array(files)

files_nonoise = np.delete(files, S_noise, axis = 0)


# Now, we are going to create a data frame that consists of the Subject ID's in which we have succefully pixeled their Clock Drawing Test Image. 

# In[65]:


# Generating the List of scores associated with the files_nonoise
score_list = np.zeros(files_nonoise.shape[0]) 

blank = []

# for loop to match score of df_join to Pixelated clock images
for indices in range(files_nonoise.shape[0]):
    ext, _ = os.path.splitext(files_nonoise[indices])
    #print(ext)
    if ext in Clock_list['Clocks'].tolist():
        score_update = Clock_list[Clock_list['Clocks'] == ext]['Scores'].iloc[0]
        print(score_update)
        score_list[indices] = score_update
    else:
        blank.append(indices)
        
# Cleaned Image Data        
dmat1_delete = np.delete(dmat1_nonoise, blank, axis = 0)

# Cleaned Score Data
Scores_delete = np.delete(score_list, blank, axis = 0)


# In[87]:


# Deleting the SIDs of the files that were delted
files_delete = np.delete(files_nonoise, blank, axis = 0)

# Convert the NumPy Array to Pandas data Frame
files_delete = pd.DataFrame(files_delete)

# Rename column in a dataframe
files_delete.columns = ['Clocks']


# In[88]:


# We need to get rid of the .PDF, .pdf, & .jpeg
# Strip .pdf
files_delete['Clocks'] = files_delete['Clocks'].str.rstrip('.pdf')
# Strip .PDF
files_delete['Clocks'] = files_delete['Clocks'].str.rstrip('.PDF')
# Strip .jpeg
files_delete['Clocks'] = files_delete['Clocks'].str.rstrip('.jpeg')

files_delete


# In[86]:


files_delete


# In[92]:


# Rename column in a dataframe
df_factors.rename(columns = {'sid':'Clocks' }, inplace = True)


# In[93]:


df = files_delete.merge(df_factors,on = 'Clocks')
df


# In[94]:


dataset_factors_use = df.merge(Clock_list,on = 'Clocks')
dataset_factors_use


# In[97]:


df_box = df.drop('Clocks', axis = 1)
df_box


# In[98]:


# Creating plot
plt.boxplot(df_box, showmeans = True)
plt.xticks([1, 2, 3, 4], ['F1compP2', 'F2compP2', 'F3compP2', 'F4compP2'])


# In[99]:


# T-test Factor 1
sc.stats.ttest_ind(dataset_factors_use['Scores'], dataset_factors_use['F1compP2'])


# In[100]:


# T-test Factor 2
sc.stats.ttest_ind(dataset_factors_use['Scores'], dataset_factors_use['F2compP2'])


# In[101]:


# T-test Factor 3
sc.stats.ttest_ind(dataset_factors_use['Scores'], dataset_factors_use['F3compP2'])


# In[102]:


# T-test Factor 4
sc.stats.ttest_ind(dataset_factors_use['Scores'], dataset_factors_use['F4compP2'])


# In[103]:


df_fac_adj = pd.read_csv('C:/Users/12563/Documents/Res. Methods in Math & Stats/Minicog_selected_variables.csv')
df_fac_adj


# In[106]:


df_fac_adj = df_fac_adj.rename(columns = {'sid' : 'Clocks'})


# In[109]:


df_fac_adj_ana = dataset_factors_use.merge(df_fac_adj,on = 'Clocks')
df_fac_adj_ana


# In[112]:


B1 = pd.read_csv('C:/Users/12563/Documents/Res. Methods in Math & Stats/Book1.csv')
B1 = B1.rename(columns = {'sid' : 'Clocks'})


# In[113]:


df_b = B1.merge(df_fac_adj_ana,on = 'Clocks')
df_b


# In[115]:


# F1 ANOVA
# w/o GOLD
F1_anova_model = ols('F1compP2 ~ C(Scores) + age_final + C(gender_final) + C(race_final) + C(education_finalV1) + C(income_final) + C(kidney_disease_final)', data=df_fac_adj_ana).fit()
anova_table_F1 = sm.stats.anova_lm(F1_anova_model, typ=1)
anova_table_F1


# In[118]:


# F2 ANOVA
F2_anova_model = ols('F2compP2 ~ C(Scores) + age_final + C(gender_final) + C(race_final) + C(education_finalV1) + C(income_final) + C(kidney_disease_final)', data=df_fac_adj_ana).fit()
anova_table_F2 = sm.stats.anova_lm(F2_anova_model, typ=1)
anova_table_F2


# In[120]:


# F3 ANOVA
# Gas-Trapping
F3_anova_model = ols('F3compP2 ~ C(Scores) + age_final + C(gender_final) + C(race_final) + C(education_finalV1) + C(income_final) + C(kidney_disease_final)', data=df_fac_adj_ana).fit()
anova_table_F3 = sm.stats.anova_lm(F3_anova_model, typ=1)
anova_table_F3


# In[122]:


# F4 ANOVA
# TLC FRC
F4_anova_model = ols('F4compP2 ~ C(Scores) + age_final + C(gender_final) + C(race_final) + C(education_finalV1) + C(income_final) + C(kidney_disease_final)', data=df_fac_adj_ana).fit()
anova_table_F4 = sm.stats.anova_lm(F4_anova_model, typ=1)
anova_table_F4


# In[ ]:





# In[ ]:


# pixalating PDF clock images and convering them to data matrix

# Reading in Images with an Adjusted Clock Drawing Test Score 
# of 0

path = r'C:/Users/12563/Documents/Test Clocks 2/'
#path = r'C:/Users/12563/Documents/Clocks_0/'
files = os.listdir(path)
dmat1 = np.zeros((len(files), 288, 288))

for index, file in enumerate(files):
    f_name_prim = os.path.basename(file)
    print(f_name_prim)
    print(path)
    print(path+f_name_prim)
    
    _, ext = os.path.splitext(file)
    if ext.lower() == '.pdf':
        # pdf processing
        #print(file)
        images = convert_from_path(path + f_name_prim, poppler_path = r"C:\Users\12563\Desktop\poppler-0.68.0_x86 (1)\poppler-0.68.0\bin")
        image = np.array(images[0])
        y,x,_ = image.shape
        
        # cropping the clocks from the images
        #print(image.shape)
         # saving the clocks in the target folder
        #fig, ax = plt.subplots()    
        #plt.title('File name: {}, Index: {}'.format(file, index))
        #ax.imshow(image, cmap='gray')
        #Cropping
        # I decided on the cropping bounds by looking at printed images
        image = image[350:1250,350:1350] 
        #print(image.shape)
         # saving the clocks in the target folder
        fig, ax = plt.subplots()    
        plt.title('File name: {}, Index: {}'.format(file, index))
        ax.imshow(image, cmap='gray')
        print()
        
        # Grayscaling the images
        image = np.mean(image, axis = 2)

        # resizing the cropped clocks
        image_resized = resize(image, (288, 288))

        dmat1[index] = image_resized

        # saving the clocks into the target folder
        fig, ax = plt.subplots() 
        plt.title('File name: {}, Index: {}'.format(file, index))
        ax.imshow(image_resized, cmap='gray')
        #fig.savefig(path + f_name_prim + '.jpeg') 

        # saving the datamatrix in the target folder
        #np.save(path + 'dmat2', dmat2) 

    elif ext.lower() == '.jpeg':
        # jpeg processing
        
  
        image = imread(path + f_name_prim)

        # cropping the clocks from the images
        image = image[550:2400, 550:2400] 

        # Grayscaling the images
        image = np.mean(image, axis = 2)

        # resizing the cropped clocks
        image_resized = resize(image, (288, 288))

        dmat1[index] = image_resized

        # saving the clocks in the target folder
        fig, ax = plt.subplots()    
        plt.title('File name: {}, Index: {}'.format(file, index))
        ax.imshow(image, cmap='gray')
        
        #fig.savefig(path + f_name_prim)  
    #else:
        #print(file, ext)
        
        #if file == "JPEG":
            #continue
        #print(file)
        #images = convert_from_path(path + f_name_prim, poppler_path = r"C:\Users\12563\Desktop\poppler-0.68.0_x86 (1)\poppler-0.68.0\bin")
        #image = np.array(images[0])
        #y,x,_ = image.shape


# We will now import the factor data, the 4 different subtypes of COPD

# In[53]:


df_factors = pd.read_csv('C:/Users/12563/Documents/Res. Methods in Math & Stats/Data/P2_factor_scores_20220610.csv')

df_factors.rename(columns = {'sid':'Clocks' }, inplace = True)
df_factors


# In[54]:


df_use = df_factors.merge(Clock_list, on = 'Clocks')
df_use


# In[ ]:





# In[55]:


dataset_factors_use = df.merge(Clock_list,on = 'Clocks')
dataset_factors_use


# In[ ]:





# In[56]:


Clock_list

