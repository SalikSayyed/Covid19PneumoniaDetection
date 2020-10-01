 # Covid-19 and Pnemonia Prediction with Convolutional Neural Network
 ## Project by Salik Sayyed
 ### Here we will train models to predict Covid-19 and Pnemonia and along with training answering one question of X-ray footprint relation between Covid-19 and Pnemonia.


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns



train_path='/content/xray_dataset_covid19/train/'
test_path='/content/xray_dataset_covid19/test/'

train_batches =ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['COVID','NORMAL'],batch_size=10)
test_batches=ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224),classes=['COVID','NORMAL'],batch_size=10)

images,labels=next(train_batches)


```

    Found 74 images belonging to 2 classes.
    Found 20 images belonging to 2 classes.
    

### Importing VGG16 model for further changes


```python
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c
```


```python
image_size= 224
def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model
```


```python
unetmodel=UNet()
```


```python
unetmodel.summary()
```

    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 224, 224, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 224, 224, 16) 448         input_1[0][0]                    
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 224, 224, 16) 2320        conv2d[0][0]                     
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, 112, 112, 16) 0           conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 112, 112, 32) 4640        max_pooling2d[0][0]              
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 112, 112, 32) 9248        conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 56, 56, 32)   0           conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 56, 56, 64)   18496       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 56, 56, 64)   36928       conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 28, 28, 64)   0           conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 28, 28, 128)  73856       max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 28, 28, 128)  147584      conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 14, 14, 128)  0           conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 14, 14, 256)  295168      max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 14, 14, 256)  590080      conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    up_sampling2d (UpSampling2D)    (None, 28, 28, 256)  0           conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 28, 28, 384)  0           up_sampling2d[0][0]              
                                                                     conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 28, 28, 128)  442496      concatenate[0][0]                
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 28, 28, 128)  147584      conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_1 (UpSampling2D)  (None, 56, 56, 128)  0           conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 56, 56, 192)  0           up_sampling2d_1[0][0]            
                                                                     conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 56, 56, 64)   110656      concatenate_1[0][0]              
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 56, 56, 64)   36928       conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_2 (UpSampling2D)  (None, 112, 112, 64) 0           conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 112, 112, 96) 0           up_sampling2d_2[0][0]            
                                                                     conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 112, 112, 32) 27680       concatenate_2[0][0]              
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 112, 112, 32) 9248        conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_3 (UpSampling2D)  (None, 224, 224, 32) 0           conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 224, 224, 48) 0           up_sampling2d_3[0][0]            
                                                                     conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 224, 224, 16) 6928        concatenate_3[0][0]              
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 224, 224, 16) 2320        conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 224, 224, 1)  17          conv2d_17[0][0]                  
    ==================================================================================================
    Total params: 1,962,625
    Trainable params: 1,962,625
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
from tensorflow.keras import layers
from tensorflow.keras import Model
last_layer = unetmodel.layers[-1]
x = tf.keras.layers.Flatten()(last_layer.output)
x = tf.keras.layers.Dense(500,activation="relu")(x)
x = tf.keras.layers.Dense(1,activation="sigmoid")(x)
newmodelunet=Model(unetmodel.input,x)
```

**DATASET 1st**


```python
train_path='/content/chest_xray/train/'
test_path='/content/chest_xray/test/'
val_path='/content/chest_xray/val/'

train_batches =ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),class_mode='binary',batch_size=100)
test_batches=ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224),class_mode='binary',batch_size=50)
val_bathes=ImageDataGenerator().flow_from_directory(val_path,target_size=(224,224),class_mode='binary',batch_size=4)
print(val_bathes)
values_batch=val_bathes
images,labels=next(train_batches)
```

    Found 5216 images belonging to 2 classes.
    Found 624 images belonging to 2 classes.
    Found 16 images belonging to 2 classes.
    <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x7fc0a13ffbe0>
    


```python
fig = plt.gcf()
ncols=4
nrows=4
train_dir=os.path.join('/content/chest_xray/train/')
train_covid_dir = os.path.join(train_dir, 'PNEUMONIA')
train_normal_dir = os.path.join(train_dir, 'NORMAL')
train_covid_fnames = os.listdir( train_covid_dir )
train_normal_fnames = os.listdir( train_normal_dir )

fig.set_size_inches(ncols*4, nrows*4)
pic_index=0
pic_index+=8
%matplotlib inline

import matplotlib.image as mpimg

next_covid_pix = [os.path.join(train_covid_dir, fname) 
                for fname in train_covid_fnames[ pic_index-8:pic_index] 
               ]

next_normal_pix = [os.path.join(train_normal_dir, fname) 
                for fname in train_normal_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_covid_pix+next_normal_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

```


![png](output_10_0.png)



```python
newmodelunet.summary()
device_name=tf.test.gpu_device_name()
```

    Model: "functional_3"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 224, 224, 3) 0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 224, 224, 16) 448         input_1[0][0]                    
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 224, 224, 16) 2320        conv2d[0][0]                     
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, 112, 112, 16) 0           conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 112, 112, 32) 4640        max_pooling2d[0][0]              
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 112, 112, 32) 9248        conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, 56, 56, 32)   0           conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    conv2d_4 (Conv2D)               (None, 56, 56, 64)   18496       max_pooling2d_1[0][0]            
    __________________________________________________________________________________________________
    conv2d_5 (Conv2D)               (None, 56, 56, 64)   36928       conv2d_4[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, 28, 28, 64)   0           conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    conv2d_6 (Conv2D)               (None, 28, 28, 128)  73856       max_pooling2d_2[0][0]            
    __________________________________________________________________________________________________
    conv2d_7 (Conv2D)               (None, 28, 28, 128)  147584      conv2d_6[0][0]                   
    __________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)  (None, 14, 14, 128)  0           conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    conv2d_8 (Conv2D)               (None, 14, 14, 256)  295168      max_pooling2d_3[0][0]            
    __________________________________________________________________________________________________
    conv2d_9 (Conv2D)               (None, 14, 14, 256)  590080      conv2d_8[0][0]                   
    __________________________________________________________________________________________________
    up_sampling2d (UpSampling2D)    (None, 28, 28, 256)  0           conv2d_9[0][0]                   
    __________________________________________________________________________________________________
    concatenate (Concatenate)       (None, 28, 28, 384)  0           up_sampling2d[0][0]              
                                                                     conv2d_7[0][0]                   
    __________________________________________________________________________________________________
    conv2d_10 (Conv2D)              (None, 28, 28, 128)  442496      concatenate[0][0]                
    __________________________________________________________________________________________________
    conv2d_11 (Conv2D)              (None, 28, 28, 128)  147584      conv2d_10[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_1 (UpSampling2D)  (None, 56, 56, 128)  0           conv2d_11[0][0]                  
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 56, 56, 192)  0           up_sampling2d_1[0][0]            
                                                                     conv2d_5[0][0]                   
    __________________________________________________________________________________________________
    conv2d_12 (Conv2D)              (None, 56, 56, 64)   110656      concatenate_1[0][0]              
    __________________________________________________________________________________________________
    conv2d_13 (Conv2D)              (None, 56, 56, 64)   36928       conv2d_12[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_2 (UpSampling2D)  (None, 112, 112, 64) 0           conv2d_13[0][0]                  
    __________________________________________________________________________________________________
    concatenate_2 (Concatenate)     (None, 112, 112, 96) 0           up_sampling2d_2[0][0]            
                                                                     conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    conv2d_14 (Conv2D)              (None, 112, 112, 32) 27680       concatenate_2[0][0]              
    __________________________________________________________________________________________________
    conv2d_15 (Conv2D)              (None, 112, 112, 32) 9248        conv2d_14[0][0]                  
    __________________________________________________________________________________________________
    up_sampling2d_3 (UpSampling2D)  (None, 224, 224, 32) 0           conv2d_15[0][0]                  
    __________________________________________________________________________________________________
    concatenate_3 (Concatenate)     (None, 224, 224, 48) 0           up_sampling2d_3[0][0]            
                                                                     conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    conv2d_16 (Conv2D)              (None, 224, 224, 16) 6928        concatenate_3[0][0]              
    __________________________________________________________________________________________________
    conv2d_17 (Conv2D)              (None, 224, 224, 16) 2320        conv2d_16[0][0]                  
    __________________________________________________________________________________________________
    conv2d_18 (Conv2D)              (None, 224, 224, 1)  17          conv2d_17[0][0]                  
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 50176)        0           conv2d_18[0][0]                  
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 500)          25088500    flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 1)            501         dense[0][0]                      
    ==================================================================================================
    Total params: 27,051,626
    Trainable params: 27,051,626
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
densenett = keras.applications.DenseNet169(input_shape=(224,224,3),include_top=False,classes=2,weights=None)
```


```python
densenett.trainable=True
```


```python
last_layer = densenett.layers[-1]
x = tf.keras.layers.Flatten()(last_layer.output)
x = tf.keras.layers.Dense(500,activation="relu")(x)
x = tf.keras.layers.Dense(1,activation="sigmoid")(x)
newmodel=Model(densenett.input,x)
```


```python
newmodelunet.compile(Adam(lr=.0001),loss='binary_crossentropy',metrics=['accuracy'])
```


```python
class Callc(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('val_accuracy')>0.85 and logs.get('accuracy')>0.99):
            print("Model Trained till desired limit!")
            self.model.stop_training = True
callbacks = Callc()
```


```python
history = newmodelunet.fit_generator(train_batches,validation_data=values_batch,validation_steps=4,steps_per_epoch=53,epochs=20,callbacks= [callbacks])
```

    WARNING:tensorflow:From <ipython-input-24-39fa5288df35>:1: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use Model.fit, which supports generators.
    Epoch 1/20
     2/53 [>.............................] - ETA: 34s - loss: 5.9251 - accuracy: 0.5150WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.4529s vs `on_train_batch_end` time: 0.8868s). Check your callbacks.
    53/53 [==============================] - 79s 1s/step - loss: 1.4363 - accuracy: 0.7228 - val_loss: 0.6644 - val_accuracy: 0.6875
    Epoch 2/20
    53/53 [==============================] - 78s 1s/step - loss: 0.3174 - accuracy: 0.8608 - val_loss: 0.4915 - val_accuracy: 0.8125
    Epoch 3/20
    53/53 [==============================] - 78s 1s/step - loss: 0.1689 - accuracy: 0.9329 - val_loss: 0.2893 - val_accuracy: 0.8750
    Epoch 4/20
    53/53 [==============================] - 79s 1s/step - loss: 0.0789 - accuracy: 0.9689 - val_loss: 0.2181 - val_accuracy: 0.9375
    Epoch 5/20
    53/53 [==============================] - 78s 1s/step - loss: 0.0538 - accuracy: 0.9799 - val_loss: 0.3599 - val_accuracy: 0.8125
    Epoch 6/20
    53/53 [==============================] - 77s 1s/step - loss: 0.0270 - accuracy: 0.9923 - val_loss: 0.3400 - val_accuracy: 0.8125
    Epoch 7/20
    53/53 [==============================] - ETA: 0s - loss: 0.0213 - accuracy: 0.9946Model Trained till desired limit!
    53/53 [==============================] - 77s 1s/step - loss: 0.0213 - accuracy: 0.9946 - val_loss: 0.2416 - val_accuracy: 0.8750
    


```python
import matplotlib.pyplot as plt
x_data=list(range(len(history.history['accuracy'])))
plt.plot(x_data,history.history['accuracy'],label=" training accuracy")
plt.plot(x_data,history.history['val_accuracy'],label="validation accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
```


![png](output_18_0.png)



```python
import numpy as np
import random
from   tensorflow.keras.preprocessing.image import img_to_array, load_img


successive_outputs = [layer.output for layer in newmodel2.layers[1:]]


visualization_model = tf.keras.models.Model(inputs = newmodel2.input, outputs = successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
covid_img_files = [os.path.join(train_covid_dir, f) for f in train_covid_fnames]
normal_img_files = [os.path.join(train_normal_dir, f) for f in train_normal_fnames]

img_path = random.choice(covid_img_files + normal_img_files)
img = load_img(img_path, target_size=(224, 224))  # this is a PIL image

x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255.0

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in newmodel2.layers]

# -----------------------------------------------------------------------
# Now let's display our representations
# -----------------------------------------------------------------------
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    
    n_features = feature_map.shape[-1] 
    size       = feature_map.shape[ 1]  
    
   
    display_grid = np.zeros((size, size * n_features))
    
    
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x 

    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:48: RuntimeWarning: invalid value encountered in true_divide
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:48: RuntimeWarning: divide by zero encountered in true_divide
    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:55: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
    


![png](output_19_1.png)



![png](output_19_2.png)



![png](output_19_3.png)



![png](output_19_4.png)



![png](output_19_5.png)



![png](output_19_6.png)



![png](output_19_7.png)



![png](output_19_8.png)



![png](output_19_9.png)



![png](output_19_10.png)



![png](output_19_11.png)



![png](output_19_12.png)



![png](output_19_13.png)



![png](output_19_14.png)



![png](output_19_15.png)



![png](output_19_16.png)



![png](output_19_17.png)



![png](output_19_18.png)



![png](output_19_19.png)



![png](output_19_20.png)



![png](output_19_21.png)



![png](output_19_22.png)



![png](output_19_23.png)



![png](output_19_24.png)



![png](output_19_25.png)



![png](output_19_26.png)



![png](output_19_27.png)



![png](output_19_28.png)



![png](output_19_29.png)



![png](output_19_30.png)



![png](output_19_31.png)



![png](output_19_32.png)



![png](output_19_33.png)



![png](output_19_34.png)



![png](output_19_35.png)



![png](output_19_36.png)



![png](output_19_37.png)



![png](output_19_38.png)



![png](output_19_39.png)



![png](output_19_40.png)



![png](output_19_41.png)



![png](output_19_42.png)



![png](output_19_43.png)



![png](output_19_44.png)



![png](output_19_45.png)



![png](output_19_46.png)



![png](output_19_47.png)



![png](output_19_48.png)



![png](output_19_49.png)



![png](output_19_50.png)



![png](output_19_51.png)



![png](output_19_52.png)



![png](output_19_53.png)



![png](output_19_54.png)



![png](output_19_55.png)



![png](output_19_56.png)



![png](output_19_57.png)



![png](output_19_58.png)



![png](output_19_59.png)



![png](output_19_60.png)



![png](output_19_61.png)



![png](output_19_62.png)



![png](output_19_63.png)



![png](output_19_64.png)



![png](output_19_65.png)



![png](output_19_66.png)



![png](output_19_67.png)



![png](output_19_68.png)



![png](output_19_69.png)



![png](output_19_70.png)



![png](output_19_71.png)



![png](output_19_72.png)



![png](output_19_73.png)



![png](output_19_74.png)



![png](output_19_75.png)



![png](output_19_76.png)



![png](output_19_77.png)



![png](output_19_78.png)



![png](output_19_79.png)



![png](output_19_80.png)



![png](output_19_81.png)



![png](output_19_82.png)



![png](output_19_83.png)



![png](output_19_84.png)



![png](output_19_85.png)



![png](output_19_86.png)



![png](output_19_87.png)



![png](output_19_88.png)



![png](output_19_89.png)



![png](output_19_90.png)



![png](output_19_91.png)



![png](output_19_92.png)



![png](output_19_93.png)



![png](output_19_94.png)



![png](output_19_95.png)



![png](output_19_96.png)



![png](output_19_97.png)



![png](output_19_98.png)



![png](output_19_99.png)



![png](output_19_100.png)



![png](output_19_101.png)



![png](output_19_102.png)



![png](output_19_103.png)



![png](output_19_104.png)



![png](output_19_105.png)



![png](output_19_106.png)



![png](output_19_107.png)



![png](output_19_108.png)



![png](output_19_109.png)



![png](output_19_110.png)



![png](output_19_111.png)



![png](output_19_112.png)



![png](output_19_113.png)



![png](output_19_114.png)



![png](output_19_115.png)



![png](output_19_116.png)



![png](output_19_117.png)



![png](output_19_118.png)



![png](output_19_119.png)



![png](output_19_120.png)



![png](output_19_121.png)



![png](output_19_122.png)



![png](output_19_123.png)



![png](output_19_124.png)



![png](output_19_125.png)



![png](output_19_126.png)



![png](output_19_127.png)



![png](output_19_128.png)



![png](output_19_129.png)



![png](output_19_130.png)



![png](output_19_131.png)



![png](output_19_132.png)



![png](output_19_133.png)



![png](output_19_134.png)



![png](output_19_135.png)



![png](output_19_136.png)



![png](output_19_137.png)



![png](output_19_138.png)



![png](output_19_139.png)



![png](output_19_140.png)



![png](output_19_141.png)



![png](output_19_142.png)



![png](output_19_143.png)



![png](output_19_144.png)



![png](output_19_145.png)



![png](output_19_146.png)



![png](output_19_147.png)



![png](output_19_148.png)



![png](output_19_149.png)



![png](output_19_150.png)



![png](output_19_151.png)



![png](output_19_152.png)



![png](output_19_153.png)



![png](output_19_154.png)



![png](output_19_155.png)



![png](output_19_156.png)



![png](output_19_157.png)



![png](output_19_158.png)



![png](output_19_159.png)



![png](output_19_160.png)



![png](output_19_161.png)



![png](output_19_162.png)



![png](output_19_163.png)



![png](output_19_164.png)



![png](output_19_165.png)



![png](output_19_166.png)



![png](output_19_167.png)



![png](output_19_168.png)



![png](output_19_169.png)



![png](output_19_170.png)



![png](output_19_171.png)



![png](output_19_172.png)



![png](output_19_173.png)



![png](output_19_174.png)



![png](output_19_175.png)



![png](output_19_176.png)



![png](output_19_177.png)



![png](output_19_178.png)



![png](output_19_179.png)



![png](output_19_180.png)



![png](output_19_181.png)



![png](output_19_182.png)



![png](output_19_183.png)



![png](output_19_184.png)



![png](output_19_185.png)



![png](output_19_186.png)



![png](output_19_187.png)



![png](output_19_188.png)



![png](output_19_189.png)



![png](output_19_190.png)



![png](output_19_191.png)



![png](output_19_192.png)



![png](output_19_193.png)



![png](output_19_194.png)



![png](output_19_195.png)



![png](output_19_196.png)



![png](output_19_197.png)



![png](output_19_198.png)



![png](output_19_199.png)



![png](output_19_200.png)



![png](output_19_201.png)



![png](output_19_202.png)



![png](output_19_203.png)



![png](output_19_204.png)



![png](output_19_205.png)



![png](output_19_206.png)



![png](output_19_207.png)



![png](output_19_208.png)



![png](output_19_209.png)



![png](output_19_210.png)



![png](output_19_211.png)



![png](output_19_212.png)



![png](output_19_213.png)



![png](output_19_214.png)



![png](output_19_215.png)



![png](output_19_216.png)



![png](output_19_217.png)



![png](output_19_218.png)



![png](output_19_219.png)



![png](output_19_220.png)



![png](output_19_221.png)



![png](output_19_222.png)



![png](output_19_223.png)



![png](output_19_224.png)



![png](output_19_225.png)



![png](output_19_226.png)



![png](output_19_227.png)



![png](output_19_228.png)



![png](output_19_229.png)



![png](output_19_230.png)



![png](output_19_231.png)



![png](output_19_232.png)



![png](output_19_233.png)



![png](output_19_234.png)



![png](output_19_235.png)



![png](output_19_236.png)



![png](output_19_237.png)



![png](output_19_238.png)



![png](output_19_239.png)



![png](output_19_240.png)



![png](output_19_241.png)



![png](output_19_242.png)



![png](output_19_243.png)



![png](output_19_244.png)



![png](output_19_245.png)



![png](output_19_246.png)



![png](output_19_247.png)



![png](output_19_248.png)



![png](output_19_249.png)



![png](output_19_250.png)



![png](output_19_251.png)



![png](output_19_252.png)



![png](output_19_253.png)



![png](output_19_254.png)



![png](output_19_255.png)



![png](output_19_256.png)



![png](output_19_257.png)



![png](output_19_258.png)



![png](output_19_259.png)



![png](output_19_260.png)



![png](output_19_261.png)



![png](output_19_262.png)



![png](output_19_263.png)



![png](output_19_264.png)



![png](output_19_265.png)



![png](output_19_266.png)



![png](output_19_267.png)



![png](output_19_268.png)



![png](output_19_269.png)



![png](output_19_270.png)



![png](output_19_271.png)



![png](output_19_272.png)



![png](output_19_273.png)



![png](output_19_274.png)



![png](output_19_275.png)



![png](output_19_276.png)



![png](output_19_277.png)



![png](output_19_278.png)



![png](output_19_279.png)



![png](output_19_280.png)



![png](output_19_281.png)



![png](output_19_282.png)



![png](output_19_283.png)



![png](output_19_284.png)



![png](output_19_285.png)



![png](output_19_286.png)



![png](output_19_287.png)



![png](output_19_288.png)



![png](output_19_289.png)



![png](output_19_290.png)



![png](output_19_291.png)



![png](output_19_292.png)



![png](output_19_293.png)



![png](output_19_294.png)



![png](output_19_295.png)



![png](output_19_296.png)



![png](output_19_297.png)



![png](output_19_298.png)



![png](output_19_299.png)



![png](output_19_300.png)



![png](output_19_301.png)



![png](output_19_302.png)



![png](output_19_303.png)



![png](output_19_304.png)



![png](output_19_305.png)



![png](output_19_306.png)



![png](output_19_307.png)



![png](output_19_308.png)



![png](output_19_309.png)



![png](output_19_310.png)



![png](output_19_311.png)



![png](output_19_312.png)



![png](output_19_313.png)



![png](output_19_314.png)



![png](output_19_315.png)



![png](output_19_316.png)



![png](output_19_317.png)



![png](output_19_318.png)



![png](output_19_319.png)



![png](output_19_320.png)



![png](output_19_321.png)



![png](output_19_322.png)



![png](output_19_323.png)



![png](output_19_324.png)



![png](output_19_325.png)



![png](output_19_326.png)



![png](output_19_327.png)



![png](output_19_328.png)



![png](output_19_329.png)



![png](output_19_330.png)



![png](output_19_331.png)



![png](output_19_332.png)



![png](output_19_333.png)



![png](output_19_334.png)



![png](output_19_335.png)



![png](output_19_336.png)



![png](output_19_337.png)



![png](output_19_338.png)



![png](output_19_339.png)



![png](output_19_340.png)



![png](output_19_341.png)



![png](output_19_342.png)



![png](output_19_343.png)



![png](output_19_344.png)



![png](output_19_345.png)



![png](output_19_346.png)



![png](output_19_347.png)



![png](output_19_348.png)



![png](output_19_349.png)



![png](output_19_350.png)



![png](output_19_351.png)



![png](output_19_352.png)



![png](output_19_353.png)



![png](output_19_354.png)



![png](output_19_355.png)



![png](output_19_356.png)



![png](output_19_357.png)



![png](output_19_358.png)



![png](output_19_359.png)



![png](output_19_360.png)



![png](output_19_361.png)



![png](output_19_362.png)



![png](output_19_363.png)



![png](output_19_364.png)



![png](output_19_365.png)



![png](output_19_366.png)



![png](output_19_367.png)



![png](output_19_368.png)



![png](output_19_369.png)



![png](output_19_370.png)



![png](output_19_371.png)



![png](output_19_372.png)



![png](output_19_373.png)



![png](output_19_374.png)



![png](output_19_375.png)



![png](output_19_376.png)



![png](output_19_377.png)



![png](output_19_378.png)



![png](output_19_379.png)



![png](output_19_380.png)



![png](output_19_381.png)



![png](output_19_382.png)



![png](output_19_383.png)



![png](output_19_384.png)



![png](output_19_385.png)



![png](output_19_386.png)



![png](output_19_387.png)



![png](output_19_388.png)



![png](output_19_389.png)



![png](output_19_390.png)



![png](output_19_391.png)



![png](output_19_392.png)



![png](output_19_393.png)



![png](output_19_394.png)



![png](output_19_395.png)



![png](output_19_396.png)



![png](output_19_397.png)



![png](output_19_398.png)



![png](output_19_399.png)



![png](output_19_400.png)



![png](output_19_401.png)



![png](output_19_402.png)



![png](output_19_403.png)



![png](output_19_404.png)



![png](output_19_405.png)



![png](output_19_406.png)



![png](output_19_407.png)



![png](output_19_408.png)



![png](output_19_409.png)



![png](output_19_410.png)



![png](output_19_411.png)



![png](output_19_412.png)



![png](output_19_413.png)



![png](output_19_414.png)



![png](output_19_415.png)



![png](output_19_416.png)



![png](output_19_417.png)



![png](output_19_418.png)



![png](output_19_419.png)



![png](output_19_420.png)



![png](output_19_421.png)



![png](output_19_422.png)



![png](output_19_423.png)



![png](output_19_424.png)



![png](output_19_425.png)



![png](output_19_426.png)



![png](output_19_427.png)



![png](output_19_428.png)



![png](output_19_429.png)



![png](output_19_430.png)



![png](output_19_431.png)



![png](output_19_432.png)



![png](output_19_433.png)



![png](output_19_434.png)



![png](output_19_435.png)



![png](output_19_436.png)



![png](output_19_437.png)



![png](output_19_438.png)



![png](output_19_439.png)



![png](output_19_440.png)



![png](output_19_441.png)



![png](output_19_442.png)



![png](output_19_443.png)



![png](output_19_444.png)



![png](output_19_445.png)



![png](output_19_446.png)



![png](output_19_447.png)



![png](output_19_448.png)



![png](output_19_449.png)



![png](output_19_450.png)



![png](output_19_451.png)



![png](output_19_452.png)



![png](output_19_453.png)



![png](output_19_454.png)



![png](output_19_455.png)



![png](output_19_456.png)



![png](output_19_457.png)



![png](output_19_458.png)



![png](output_19_459.png)



![png](output_19_460.png)



![png](output_19_461.png)



![png](output_19_462.png)



![png](output_19_463.png)



![png](output_19_464.png)



![png](output_19_465.png)



![png](output_19_466.png)



![png](output_19_467.png)



![png](output_19_468.png)



![png](output_19_469.png)



![png](output_19_470.png)



![png](output_19_471.png)



![png](output_19_472.png)



![png](output_19_473.png)



![png](output_19_474.png)



![png](output_19_475.png)



![png](output_19_476.png)



![png](output_19_477.png)



![png](output_19_478.png)



![png](output_19_479.png)



![png](output_19_480.png)



![png](output_19_481.png)



![png](output_19_482.png)



![png](output_19_483.png)



![png](output_19_484.png)



![png](output_19_485.png)



![png](output_19_486.png)



![png](output_19_487.png)



![png](output_19_488.png)



![png](output_19_489.png)



![png](output_19_490.png)



![png](output_19_491.png)



![png](output_19_492.png)



![png](output_19_493.png)



![png](output_19_494.png)



![png](output_19_495.png)



![png](output_19_496.png)



![png](output_19_497.png)



![png](output_19_498.png)



![png](output_19_499.png)



![png](output_19_500.png)



![png](output_19_501.png)



![png](output_19_502.png)



![png](output_19_503.png)



![png](output_19_504.png)



![png](output_19_505.png)



![png](output_19_506.png)



![png](output_19_507.png)



![png](output_19_508.png)



![png](output_19_509.png)



![png](output_19_510.png)



![png](output_19_511.png)



![png](output_19_512.png)



![png](output_19_513.png)



![png](output_19_514.png)



![png](output_19_515.png)



![png](output_19_516.png)



![png](output_19_517.png)



![png](output_19_518.png)



![png](output_19_519.png)



![png](output_19_520.png)



![png](output_19_521.png)



![png](output_19_522.png)



![png](output_19_523.png)



![png](output_19_524.png)



![png](output_19_525.png)



![png](output_19_526.png)



![png](output_19_527.png)



![png](output_19_528.png)



![png](output_19_529.png)



![png](output_19_530.png)



![png](output_19_531.png)



![png](output_19_532.png)



![png](output_19_533.png)



![png](output_19_534.png)



![png](output_19_535.png)



![png](output_19_536.png)



![png](output_19_537.png)



![png](output_19_538.png)



![png](output_19_539.png)



![png](output_19_540.png)



![png](output_19_541.png)



![png](output_19_542.png)



![png](output_19_543.png)



![png](output_19_544.png)



![png](output_19_545.png)



![png](output_19_546.png)



![png](output_19_547.png)



![png](output_19_548.png)



![png](output_19_549.png)



![png](output_19_550.png)



![png](output_19_551.png)



![png](output_19_552.png)



![png](output_19_553.png)



![png](output_19_554.png)



![png](output_19_555.png)



![png](output_19_556.png)



![png](output_19_557.png)



![png](output_19_558.png)



![png](output_19_559.png)



![png](output_19_560.png)



![png](output_19_561.png)



![png](output_19_562.png)



![png](output_19_563.png)



![png](output_19_564.png)



![png](output_19_565.png)



![png](output_19_566.png)



![png](output_19_567.png)



![png](output_19_568.png)



![png](output_19_569.png)



![png](output_19_570.png)



![png](output_19_571.png)



![png](output_19_572.png)



![png](output_19_573.png)



![png](output_19_574.png)



![png](output_19_575.png)



![png](output_19_576.png)



![png](output_19_577.png)



![png](output_19_578.png)



![png](output_19_579.png)



![png](output_19_580.png)



![png](output_19_581.png)



![png](output_19_582.png)



![png](output_19_583.png)



![png](output_19_584.png)



![png](output_19_585.png)



![png](output_19_586.png)



![png](output_19_587.png)



![png](output_19_588.png)



![png](output_19_589.png)



![png](output_19_590.png)



![png](output_19_591.png)



![png](output_19_592.png)



![png](output_19_593.png)



![png](output_19_594.png)


**saving in file later to be downloaded**


```python
newmodel.save('covid_trained_model_final.h5')
```


```python
train_path2='/content/xray_dataset_covid19/train/'
test_path2='/content/xray_dataset_covid19/test/'


train_batches2 =ImageDataGenerator().flow_from_directory(train_path2,target_size=(224,224),class_mode='binary',batch_size=10)
test_batches2=ImageDataGenerator().flow_from_directory(test_path2,target_size=(224,224),class_mode='binary',batch_size=10)
```

    Found 148 images belonging to 2 classes.
    Found 40 images belonging to 2 classes.
    


```python
newmodel2=newmodel
```


```python
newmodel2.compile(Adam(lr=.00001),loss='binary_crossentropy',metrics=['accuracy'])
```


```python
class Callcn(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy')>0.99):
            print("Model Trained till desired limit!")
            self.model.stop_training = True
callbacks1 = Callcn()
```


```python
history2=newmodel2.fit_generator(train_batches2,steps_per_epoch=15,epochs=3,callbacks=[callbacks1])
```

    Epoch 1/3
    15/15 [==============================] - 10s 641ms/step - loss: 0.3058 - accuracy: 0.8784
    Epoch 2/3
    15/15 [==============================] - 6s 428ms/step - loss: 0.0736 - accuracy: 0.9595
    Epoch 3/3
    15/15 [==============================] - 7s 442ms/step - loss: 0.0393 - accuracy: 0.9865
    


```python
import matplotlib.pyplot as plt
plt.plot(list(range(len(history2.history['accuracy']))),history2.history['accuracy'],label="training accuracy")
plt.plot(list(range(len(history2.history['loss']))),history2.history['loss'],label="loss")
plt.legend()
plt.show()
```


![png](output_27_0.png)



```python
newmodel2.save('new.h5')
```


```python
file=os.stat('new.h5')
print(file)
```

    os.stat_result(st_mode=33188, st_ino=3822155, st_dev=50, st_nlink=1, st_uid=0, st_gid=0, st_size=642522536, st_atime=1601547280, st_mtime=1601547284, st_ctime=1601547284)
    


```python
import os
os.chdir(r'/kaggle/working')
from IPython.display import FileLink
FileLink(r'new.h5')
```




<a href='new.h5' target='_blank'>new.h5</a><br>




```python
from keras.preprocessing import image

import numpy as np
import os
allnormalcases=[]
covidinnormalcases=[]
allcovidcases=[]
covidincovidcases=[]

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224,224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
                               # imshow expects values in the range [0, 1]

    
    return img_tensor

pnemoniadetected = [] 
allcoviddetected = [] 
allpnemoniadetected = [] 
coviddetected = []
if __name__ == "__main__":

   
    dir_path = '/content/xray_dataset_covid19/test/NORMAL/'   
    dir_path2='/content/xray_dataset_covid19/test/PNEUMONIA/'# dog
    imgs= os.listdir(dir_path)
    
    img_path2 = '/kaggle/input/random/images.jpg'
    imgs2=os.listdir(dir_path2)
    print("IN ---------->"+dir_path)
   
    pn_n=0
    nn_n=0
    for i in imgs:
        new_image = load_image(dir_path+i)
        pred=newmodel2.predict(new_image)
        
        if(pred[0][0]>pred[0][1]):
            print("COVID UPTO :",pred[0][0])
            covidinnormalcases.append(pred[0],[0])
            pn_n+=1
        else:
            print("NORMAL UPTO : ",pred[0][1])
            allnormalcases.append(pred[0][1])
            nn_n+=1
    print("Accuracy in prediction of normal cases from all normal cases:",nn_n/(pn_n+nn_n)*100)
    nn_p=0
    pn_p=0
    print("IN ---------->"+dir_path2)
    for i in imgs2:
        new_image = load_image(dir_path2+i)
        pred=newmodel2.predict(new_image)
        
        if(pred[0][0]>pred[0][1]):
            print("COVID UPTO :",pred[0][0])
            covidincovidcases.append(pred[0][0])
            pn_p+=1
        else:
            print("NORMAL UPTO : ",pred[0][1])
            allcovidcases.append(pred[0][1])
            nn_p+=1
    print("Accuracy in Prediction from All covid cases :",pn_p/(pn_p+nn_p)*100)
    print("Overall accuracy of covid out of predicted covid:-->>",(pn_p)/(pn_p+pn_n)*100)
```


```python
print(allnormalcases)
plt.plot(allnormalcases)
plt.ylabel("accuracy of normal in normal")
plt.xlabel("nos")
plt.show()
```

    [0.99974686, 0.85637265, 0.9356872, 0.98938423, 0.6270642, 0.958508, 0.99407417, 0.71173096, 0.9979596, 0.9404524, 0.99007684, 0.97994506, 0.99306184, 0.99777347, 0.8471994, 0.9960483, 0.9791173, 0.88805073, 0.8226056, 0.7821271]
    


![png](output_32_1.png)


### As you can see all the normal cases are predicted to be normal, but the accuracy does decrease below 60% which shows Covid prediction can be false positive


```python
print(allcovidcases)
plt.plot(allcovidcases)
plt.ylabel("accuracy of normal in covid")
plt.xlabel("nos")
plt.show()
```


![png](output_34_0.png)


### There are no prediction of Normal patient as Covid 


```python
plt.plot(covidincovidcases)
plt.ylabel("accuracy of covid in covid cases")
plt.xlabel("nos")
plt.show()
```


![png](output_36_0.png)


### Now when it comes to predicting Covid Cases from all Covid Cases model seems to work pretty well as all accuracies are above 0.993
### No case of Covid is left behind


```python
plt.plot(covidinnormalcases)
plt.plot(emp)
plt.ylabel("accuracy of covid in normal")
plt.xlabel("nos")
plt.show()
```


![png](output_38_0.png)


### Theres no prediction of Covid in Normal however in previous graph of accuracies for normal patients as normal
### we say accuracy decreased so False positive is a possibility


```python
test_loss, test_acc = newmodel2.evaluate(test_batches2)
print(test_loss,test_acc)
```

    4/4 [==============================] - 3s 690ms/step
    0.15482132136821747 0.9750000238418579
    


```python
test_loss, test_acc = newmodel.evaluate(test_batches)
print(test_loss,test_acc)
```

    13/13 [==============================] - 16s 1s/step
    0.33656978607177734 0.8782051205635071
    


```python
yamlmodel=newmodel.to_yaml()
type(yamlmodel)
print(yamlmodel)
```

    backend: tensorflow
    class_name: Sequential
    config:
      layers:
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: &id001 !!python/tuple
          - 1
          - 1
          dtype: float32
          filters: 64
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: &id002 !!python/tuple
          - 3
          - 3
          name: block1_conv1
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 64
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block1_conv2
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: MaxPooling2D
        config:
          data_format: channels_last
          dtype: float32
          name: block1_pool
          padding: valid
          pool_size: &id003 !!python/tuple
          - 2
          - 2
          strides: *id003
          trainable: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 128
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block2_conv1
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 128
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block2_conv2
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: MaxPooling2D
        config:
          data_format: channels_last
          dtype: float32
          name: block2_pool
          padding: valid
          pool_size: *id003
          strides: *id003
          trainable: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 256
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block3_conv1
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 256
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block3_conv2
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: Dropout
        config:
          dtype: float32
          name: dropout_17
          noise_shape: null
          rate: 0.5
          seed: null
          trainable: true
      - class_name: MaxPooling2D
        config:
          data_format: channels_last
          dtype: float32
          name: block3_pool
          padding: valid
          pool_size: *id003
          strides: *id003
          trainable: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 512
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block4_conv1
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 512
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block4_conv2
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: Dropout
        config:
          dtype: float32
          name: dropout_18
          noise_shape: null
          rate: 0.5
          seed: null
          trainable: true
      - class_name: MaxPooling2D
        config:
          data_format: channels_last
          dtype: float32
          name: block4_pool
          padding: valid
          pool_size: *id003
          strides: *id003
          trainable: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 512
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block5_conv1
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 512
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block5_conv2
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: Conv2D
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          data_format: channels_last
          dilation_rate: *id001
          dtype: float32
          filters: 512
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          kernel_size: *id002
          name: block5_conv3
          padding: same
          strides: *id001
          trainable: true
          use_bias: true
      - class_name: MaxPooling2D
        config:
          data_format: channels_last
          dtype: float32
          name: block5_pool
          padding: valid
          pool_size: *id003
          strides: *id003
          trainable: true
      - class_name: Flatten
        config:
          data_format: channels_last
          dtype: float32
          name: flatten
          trainable: true
      - class_name: Dense
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          dtype: float32
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          name: fc1
          trainable: true
          units: 4096
          use_bias: true
      - class_name: Dense
        config:
          activation: relu
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          dtype: float32
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          name: fc2
          trainable: true
          units: 4096
          use_bias: true
      - class_name: Dense
        config:
          activation: softmax
          activity_regularizer: null
          bias_constraint: null
          bias_initializer:
            class_name: Zeros
            config: {}
          bias_regularizer: null
          dtype: float32
          kernel_constraint: null
          kernel_initializer:
            class_name: VarianceScaling
            config:
              distribution: uniform
              mode: fan_avg
              scale: 1.0
              seed: null
          kernel_regularizer: null
          name: dense_3
          trainable: true
          units: 2
          use_bias: true
      name: sequential_11
    keras_version: 2.3.1
    
    

### Let's move on to answer one important question -
### Does Covid-19 X-ray shows pnemonia OR Pneomina X-rays are more likely to be Covid-19 cases ?


```python
plt.plot(allpnemoniadetected,label='pnemonia')
plt.plot(coviddetectedfromp,label='covid')
plt.xlabel('all pnemonia cases')
plt.ylabel('acc')
plt.legend()
plt.show()
```


![png](output_44_0.png)


### Collectively graph aboe shows that when we selected all the pnemonia cases likelihood of Covid-19 is more but model is not biased towards it


```python
plt.plot(pnemoniadetected,label='pnemonia')
plt.plot(allcoviddetected,label='covid')
plt.xlabel('all corona cases')
plt.ylabel('acc')
plt.legend()
plt.show()
```


![png](output_46_0.png)


### However when all corona postive cases selected there is relatively high possiblity of having Pnemonia 

#### Short conclusion of Deep learning detection of model : Models are not biased towards picking up only one of the pnemonia or covid-19
#### However, one thing concluded that as per model Covid-19 case is more likely to have pnemonia as well if model training is correct
