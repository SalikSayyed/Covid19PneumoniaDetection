# Covid19PneumoniaDetection
Covid-19 and Pneumonia detection through X-ray images and trying to answer weather a Covid-19 patient has Pnemonia

For Notebook see : ![Covid-19 MD](./covid_19_x_ray_with_pneumonia%20(1).md)

For Application Video : https://drive.google.com/file/d/11rNuIFvWNUW89yPdfU6cjC9JS15GMiU0/view?usp=sharing

![png](capture 3.jpg)
![pngg](Capture.png)
![pnggg](capture2.png)

## Notebook
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


