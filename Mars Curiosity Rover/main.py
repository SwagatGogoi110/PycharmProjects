
# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


# In[2]:


import os
import os.path
from pathlib import Path
import glob


# In[3]:


from PIL import Image
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
import skimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import convolve
from skimage import data, io, filters


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


# In[5]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
#OPTIMIZER
from keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD


# In[6]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D,ZeroPadding2D, Convolution2D, ZeroPadding2D,AveragePooling2D,Input, GlobalMaxPooling2D, Conv2DTranspose, Reshape
from keras import models
from keras import layers
from keras import Input
from keras.models import Model 
import tensorflow as tf
from keras.applications import VGG16,VGG19,inception_v3
from keras import backend as K
from keras.utils import plot_model
from keras.models import load_model
from keras.regularizers import l1,l2,L1L2
from keras import regularizers

from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)


# In[7]:


Mars_Train_csv = pd.read_csv(r"E:\PycharmProjects\MArs\Mars Surface and Curiosity Image\Train_CSV.csv")
Mars_Test_csv = pd.read_csv(r"E:\PycharmProjects\MArs\Mars Surface and Curiosity Image\Test_CSV.csv")
Mars_Validation_csv = pd.read_csv(r"E:\PycharmProjects\MArs\Mars Surface and Curiosity Image\Validation_CSV.csv")


# In[8]:


print(Mars_Train_csv.head(-1))


# In[9]:


print(Mars_Test_csv.head(-1))


# In[10]:


print(Mars_Validation_csv.head(-1))


# In[11]:


def new_path_function(jpg_path,jpg_labels,new_jpg_list,new_label_list,splitting_string = "calibrated"):
    
    for image_path, path_label in zip(jpg_path,jpg_labels):
        ID_pathing,exporting_type = image_path.split(splitting_string)
        New_File_Path_Name = "E:\PycharmProjects\MArs\Mars Surface and Curiosity Image\images" + str(exporting_type)
        new_jpg_list.append(New_File_Path_Name)
        new_label_list.append(path_label)


# In[12]:


New_JPG_Path_Train = []
New_Labels_Train = []
splitting_string = "calibrated"

new_path_function(Mars_Train_csv.JPG,Mars_Train_csv.LABELS,New_JPG_Path_Train,New_Labels_Train,splitting_string)


# In[13]:


New_JPG_Path_Test = []
New_Labels_Test = []
splitting_string = "calibrated"

new_path_function(Mars_Test_csv.JPG,Mars_Test_csv.LABELS,New_JPG_Path_Test,New_Labels_Test,splitting_string)


# In[14]:


New_JPG_Path_Validation = []
New_Labels_Validation = []
splitting_string = "calibrated"

new_path_function(Mars_Validation_csv.JPG,Mars_Validation_csv.LABELS,New_JPG_Path_Validation,New_Labels_Validation,splitting_string)


# In[15]:


Train_JPG_Series = pd.Series(New_JPG_Path_Train,name="JPG").astype(str)
Train_Labels_Series = pd.Series(New_Labels_Train,name="CATEGORY")


# In[16]:


Test_JPG_Series = pd.Series(New_JPG_Path_Test,name="JPG").astype(str)
Test_Labels_Series = pd.Series(New_Labels_Test,name="CATEGORY")


# In[17]:


Validation_JPG_Series = pd.Series(New_JPG_Path_Validation,name="JPG").astype(str)
Validation_Labels_Series = pd.Series(New_Labels_Validation,name="CATEGORY")


# In[18]:


Main_Train_Data = pd.concat([Train_JPG_Series,Train_Labels_Series],axis=1)


# In[19]:


print(Main_Train_Data.head(-1))


# In[20]:


Main_Test_Data = pd.concat([Test_JPG_Series,Test_Labels_Series],axis=1)


# In[21]:


print(Main_Test_Data.head(-1))


# In[22]:


Main_Validation_Data = pd.concat([Validation_JPG_Series,Validation_Labels_Series],axis=1)
print(Main_Validation_Data.head(-1))


# In[23]:


frame_list = [Main_Train_Data,Main_Test_Data,Main_Train_Data]


# In[24]:


Main_Mars_Data = pd.concat(frame_list)
print(Main_Mars_Data.head(-1))


# In[25]:


Main_Mars_Data = Main_Mars_Data.sample(frac=1).reset_index(drop=True)
print(Main_Mars_Data.head(-1))


# In[26]:


def threshold_function(image_path):
    
    Picking_IMG = image_path
    Picking_IMG = cv2.cvtColor(cv2.imread(Picking_IMG),cv2.COLOR_BGR2RGB)
    _,Threshold_IMG = cv2.threshold(Picking_IMG,200,255,cv2.THRESH_BINARY_INV)
    
    plt.xlabel(Threshold_IMG.shape)
    plt.ylabel(Threshold_IMG.size)
    plt.imshow(Threshold_IMG)


# In[27]:


def simple_vision(image_path):
    
    Picking_IMG = image_path
    Picking_IMG = cv2.cvtColor(cv2.imread(Picking_IMG),cv2.COLOR_BGR2RGB)
    
    plt.xlabel(Picking_IMG.shape)
    plt.ylabel(Picking_IMG.size)
    plt.imshow(Picking_IMG)


# In[28]:


def just_vision(image_path):
    
    plt.xlabel(image_path.shape)
    plt.ylabel(image_path.size)
    plt.imshow(image_path)


# In[29]:


def just_threshold(image_path):
    
    _,threshold_IMG = cv2.threshold(image_path,220,255,cv2.THRESH_BINARY_INV)
    
    plt.xlabel(threshold_IMG.shape)
    plt.ylabel(threshold_IMG.size)
    plt.imshow(threshold_IMG)


# In[30]:


def just_canny(image_path):
    
    Canny_Image = cv2.Canny(image_path,10,100)
    
    plt.xlabel(Canny_Image.shape)
    plt.ylabel(Canny_Image.size)
    plt.imshow(Canny_Image)
def just_drawing_contour(image_path):
    
    Canny_Image = cv2.Canny(image_path,10,100)
    contour,_ = cv2.findContours(Canny_Image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    Drawing_Contour = cv2.drawContours(image_path,contour,-1,(255,0,0),2)
    
    plt.xlabel(Drawing_Contour.shape)
    plt.ylabel(Drawing_Contour.size)
    plt.imshow(Drawing_Contour)


# In[31]:


simple_vision(Main_Mars_Data["JPG"][3])


# In[32]:


simple_vision(Main_Mars_Data["JPG"][30])


# In[33]:


simple_vision(Main_Mars_Data["JPG"][4560])


# In[34]:


figure,axis = plt.subplots(5,5,figsize=(10,10))

for indexing,operation in enumerate(axis.flat):
    
    Picking_Image = Main_Mars_Data["JPG"][indexing]
    Reading_Image = cv2.cvtColor(cv2.imread(Picking_Image),cv2.COLOR_BGR2RGB)
    
    operation.set_xlabel(Reading_Image.shape)
    operation.set_ylabel(Reading_Image.size)
    operation.set_title(Main_Mars_Data["CATEGORY"][indexing])
    operation.imshow(Reading_Image)
    
plt.tight_layout()
plt.show()


# In[35]:


threshold_function(Main_Mars_Data["JPG"][3])


# In[36]:


threshold_function(Main_Mars_Data["JPG"][44])


# In[37]:


threshold_function(Main_Mars_Data["JPG"][400])


# In[38]:


threshold_function(Main_Mars_Data["JPG"][827])


# In[39]:


Another_Mars_Path = Path(r"../Mars Surface and Curiosity Image/additional_images")


# In[40]:


Another_JPG_List = list(Another_Mars_Path.glob(r"*.jpg"))
Another_JPG_Series = pd.Series(Another_JPG_List,name="JPG").astype(str)
print(Another_JPG_Series.head(-1))


# In[41]:


Another_JPG_Series = Another_JPG_Series[0:2000]
print(Another_JPG_Series.head(-1))


# In[42]:


Transformed_X = []

for X_image in Another_JPG_Series:
    
    One_Image = cv2.cvtColor(cv2.imread(X_image),cv2.COLOR_BGR2RGB)
    One_Image = cv2.resize(One_Image,(180,180))
    One_Image = One_Image / 255.0
    Transformed_X.append(One_Image)


# In[43]:


X_AE_Train = np.array(Transformed_X)
print(X_AE_Train.shape)


# In[44]:


Reading_IMG = cv2.cvtColor(cv2.imread(Another_JPG_Series[0]),cv2.COLOR_BGR2RGB)

just_canny(Reading_IMG)


# In[45]:


Reading_IMG = cv2.cvtColor(cv2.imread(Another_JPG_Series[200]),cv2.COLOR_BGR2RGB)

just_canny(Reading_IMG)


# In[46]:


Reading_IMG = cv2.cvtColor(cv2.imread(Another_JPG_Series[600]),cv2.COLOR_BGR2RGB)

just_canny(Reading_IMG)


# In[47]:


Reading_IMG = cv2.cvtColor(cv2.imread(Another_JPG_Series[600]),cv2.COLOR_BGR2RGB)

just_drawing_contour(Reading_IMG)


# In[48]:


Reading_IMG = cv2.cvtColor(cv2.imread(Another_JPG_Series[1600]),cv2.COLOR_BGR2RGB)

just_drawing_contour(Reading_IMG)


# In[49]:


X_Train,X_Test = train_test_split(Main_Mars_Data,train_size=0.9,random_state=42,shuffle=True)
print(X_Train.shape)
print(X_Test.shape)


# In[50]:


Validation_Set = X_Train[6000:7917]
X_Train = X_Train[0:2000]
Validation_Set = Validation_Set.reset_index()
print(X_Train.shape)
print(X_Test.shape)
print(Validation_Set.shape)


# In[51]:


print(X_Train.CATEGORY.value_counts())


# In[52]:


Transformed_Y_Train = []
Transformed_Y_Train_Labels = []

for Y_image, Y_labels in zip(X_Train.JPG,X_Train.CATEGORY):
    
    Y_image = cv2.cvtColor(cv2.imread(Y_image),cv2.COLOR_BGR2RGB)
    Y_image = cv2.resize(Y_image,(180,180))
    Y_image = Y_image / 255.0
    Transformed_Y_Train.append(Y_image)
    Transformed_Y_Train_Labels.append(Y_labels)


# In[53]:


Y_S_Train_Img = np.array(Transformed_Y_Train)
Y_S_Train_Labels = to_categorical(Transformed_Y_Train_Labels)


# In[54]:


print(Y_S_Train_Img.shape)
print(Y_S_Train_Labels.shape)
print(X_AE_Train.shape)


# In[55]:


Early_Stopper = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3,mode="min")
Checkpoint_Model = tf.keras.callbacks.ModelCheckpoint(monitor="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      filepath="./modelcheck")
Reduce_Model = tf.keras.callbacks.ReduceLROnPlateau(monitor="accuracy",
                                                   factor=0.1,
                                                   patience=7)


# In[56]:


Input_Layer = tf.keras.Input(shape=(180,180,3))
#
x = Conv2D(32,(3,3),activation="relu",padding="same")(Input_Layer)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64,(3,3),activation="relu",padding="same")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(128,(2,2),activation="relu",padding="same")(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(256,(2,2),activation="relu",padding="same")(x)
x = GlobalMaxPooling2D()(x)
x = Dense(128,activation="relu")(x)
x = Dropout(0.5)(x)
class_prediction_layer = Dense(25,activation="softmax",name="CLASS_PREDICTION")(x)
#
encoder = Dense(128,activation="relu")(Input_Layer)
encoder = Dense(64,activation="relu")(encoder)
encoder = Dense(32,activation="relu")(encoder)
#
decoder = Dense(64,input_shape=[32],activation="relu")(encoder)
decoder = Dense(128,activation="relu")(decoder)
ae_output = Dense(3,activation="sigmoid",name="AE_OUTPUT")(decoder)
Configure_Model = Model(Input_Layer,[class_prediction_layer,ae_output])
print(Configure_Model.summary())


# In[57]:


plot_model(Configure_Model, to_file='Conf.png', show_shapes=True, show_layer_names=True)


# In[58]:


Configure_Model.compile(optimizer="adam",loss={"CLASS_PREDICTION":"categorical_crossentropy",
                                              "AE_OUTPUT":"binary_crossentropy"},metrics=["accuracy"])


# In[59]:


import time
start = time.time()
Model_Configure_Total = Configure_Model.fit(Y_S_Train_Img,
                    [Y_S_Train_Labels,X_AE_Train],
                    epochs=1,
                    batch_size=2,
                    callbacks=[Early_Stopper,Checkpoint_Model,Reduce_Model])
end = time.time()
total = end - start
print(total)


# In[60]:


plt.style.use("dark_background")


# In[61]:


Grap_Data = pd.DataFrame(Model_Configure_Total.history)
Grap_Data.plot()


# In[62]:


plt.plot(Model_Configure_Total.history["CLASS_PREDICTION_loss"])
plt.plot(Model_Configure_Total.history["AE_OUTPUT_loss"])
plt.ylabel("ACCURACY")
plt.legend()
plt.show()


# In[63]:


plt.plot(Model_Configure_Total.history["CLASS_PREDICTION_accuracy"])
plt.plot(Model_Configure_Total.history["AE_OUTPUT_accuracy"])
plt.ylabel("ACCURACY")
plt.legend()
plt.show()


# In[64]:


Test_Prediction = Configure_Model(Y_S_Train_Img[0:10])
print(type(Test_Prediction))


# In[65]:


print(Test_Prediction[0])


# In[66]:


print(Test_Prediction[0].argmax(axis=-1)) # this is first class(label) prediction


# In[ ]:


fig, axes = plt.subplots(nrows=2,
                         ncols=5,
                         figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(Transformed_Y_Train[i])
    ax.set_title(f"PREDICTION:{Test_Prediction[0].argmax(axis=-1)[i]}")
plt.tight_layout()
plt.show()


# In[ ]:


Transformed_Y_Test = []
Transformed_Y_Test_Labels = []

for Y_image, Y_labels in zip(X_Test.JPG,X_Test.CATEGORY):
    
    Y_image = cv2.cvtColor(cv2.imread(Y_image),cv2.COLOR_BGR2RGB)
    Y_image = cv2.resize(Y_image,(180,180))
    Y_image = Y_image / 255.0
    Transformed_Y_Test.append(Y_image)
    Transformed_Y_Test_Labels.append(Y_labels)
Y_S_Test_Img = np.array(Transformed_Y_Test)
Y_S_Test_Labels = to_categorical(Transformed_Y_Test_Labels)
Other_Test_Prediction = Configure_Model.predict(Y_S_Test_Img[0:10])


# In[ ]:


fig, axes = plt.subplots(nrows=2,
                         ncols=5,
                         figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(Transformed_Y_Test[i])
    ax.set_title(f"PREDICTION:{Other_Test_Prediction[0].argmax(axis=-1)[i]}")
plt.tight_layout()
plt.show()


# In[ ]:


print("NORMAL")
plt.imshow(X_AE_Train[0])
plt.show()
print("Auto Encoder")
plt.imshow(Test_Prediction[1][1])


# In[ ]:


print("NORMAL")
plt.imshow(Y_S_Test_Img[0])
plt.show()
print("Auto Encoder")
plt.imshow(Other_Test_Prediction[1][1])

