''' How to use these scripts: Please run these scripts in spyder, as they were made in spyder and I can't guarantee they will work from the command line, jupyter notebook etc. See the git repository readme for instructions on how to install the computing environment from the .yml file
    
    If you haven't done already, please sort your images. See the header of the previous script on how to do that
    This script trains a keras machine learning model so that it can categorise the different digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 and g. 
    Keras is only one type of machine learning model designed for categorising images. There are many other different types of machine learning models for different purposes and different algorithms which achieve similar things.
    This script saves the trained model object as a .h5 file in \outputs\trained_model\
    Please run this script, then move on to 3_interpreting_video.py
'''

##############################################################################################################################
################################################## Making a ML model with my own data from the video ##################################################
########################################################################################################################################
#Trains a simple ML model for image recognition

import os; working_in= os.path.dirname(os.path.realpath(__file__))
package_directory=working_in[0:-12]
outputs_directory=package_directory+r'\outputs\extracted_training_data\\'
model_directory=package_directory+r'\outputs\trained_model\\'


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 11
epochs = 24

# input image dimensions
img_rows, img_cols = 28, 28

# View training data
from PIL import Image
import numpy as np
import os

n=0
identified_numbers=[]

def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)


    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


subfolders, files = run_fast_scandir(outputs_directory, [".png"])
n=len(files)


allArrays = np.empty((n,28, 28))
i=-1
for name in files:
    if name[len(name)-4:len(name)]=='.png':
        i=i+1
        print(name)
        imge = Image.open(name)
        image_sequence = np.asarray(imge)
        allArrays[i,:,:] =  image_sequence
        in_folder_called = str.split(name,'\\')[-2] # identify digit from folder name
        
        # g needs to become a number. make it 10
        if in_folder_called=='g':
            in_folder_called='10'
            
        identified_numbers.append(in_folder_called)




pct20 = int(n*0.99)
x_train = allArrays[0:pct20,:,:]
x_test = allArrays[pct20:n,:,:]
y_train=identified_numbers[0:pct20]
y_test=identified_numbers[pct20:n]








# 5. Preprocess input data
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
 
# 6. Preprocess class labels
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
 
# 7. Define model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

 
# 8. Compile model

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

 
# 9. Fit model on training data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

 
# 10. Evaluate model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 11. Save model
model.save(model_directory+'\digits_trained_model_JR.h5')  # creates a HDF5 file
