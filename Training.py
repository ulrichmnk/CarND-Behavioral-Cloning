
from keras.models import Sequential
from keras.layers import ELU, Cropping2D
from keras.layers.core import Dense, Dropout, Flatten, Lambda, Activation
from keras.layers.convolutional import Conv2D
import cv2
import numpy as np
import csv
from keras.layers import Cropping2D
from sklearn.model_selection import train_test_split
import sklearn
import os
from math import ceil
from sklearn.utils import shuffle

ch, row, col = 3, 160, 320
# camera format

Model = Sequential()
Model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(row, col, ch)))
# Model.add(Lambda(lambda x: cv2.resize(x,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)))
Model.add(Lambda(lambda x: (x/255.) -0.5))
Model.add(Conv2D(24, (5, 5),strides= (2,2), padding='valid'))
Model.add(Activation('relu'))
Model.add(Conv2D(36, (5, 5), strides= (2,2), padding='valid'))
Model.add(Activation('relu'))
Model.add(Conv2D(48, (5, 5), strides= (2,2), padding='valid'))
Model.add(Activation('relu'))
Model.add(Conv2D(64, (3, 3), strides= (1,1), padding='valid'))
Model.add(Activation('relu'))
Model.add(Conv2D(64, (3, 3), strides= (1,1), padding='valid'))
Model.add(Activation('relu'))
Model.add(Flatten())
Model.add(Dropout(.5))
Model.add(Dense(100))
Model.add(Activation('relu'))
Model.add(Dense(50))
Model.add(Activation('relu'))
Model.add(Dense(10))
Model.add(Activation('relu'))
Model.add(Dense(1))

Model.compile(optimizer="adam", loss="mse")


Log_Lines = []
with open('../../../opt/MyData/driving_log.csv') as Log_File:
# with open('../../../opt/carnd_p3/driving_log.csv') as Log_File:
    Log = csv.reader(Log_File)
    for Line in Log:
        Log_Lines.append(Line)


Train_Samples, Valid_Samples = train_test_split(Log_Lines, test_size=0.2)

Meas_Correction = 0.15
def generator(Log_Lines, Batch_Size=32):
    Num_Log_Lines = len(Log_Lines)
    while 1: # Loop forever so the generator never terminates
        shuffle(Log_Lines)
        for Offset in range(0, Num_Log_Lines, Batch_Size):
            Log_Lines_Batch = Log_Lines[Offset:Offset+Batch_Size]

            Img = []
            Meas = []
            for Line in Log_Lines_Batch:
                Center_Img_Name = Line[0].split('/')[-1]
                Left_Img_Name = Line[1].split('/')[-1]
                Right_Img_Name = Line[2].split('/')[-1]
                Img_Path = '../../../opt/MyData/IMG/'
#                 Img_Path = '../../../opt/carnd_p3/IMG/'

                Center_Img_Path = Img_Path + Center_Img_Name
                Left_Img_Path = Img_Path + Left_Img_Name
                Right_Img_Path = Img_Path + Right_Img_Name

                Center_Img = cv2.imread(Center_Img_Path)
#                 Center_Img =cv2.resize(Center_Img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
                Left_Img = cv2.imread(Left_Img_Path)
#                 Left_Img =cv2.resize(Left_Img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
                Right_Img = cv2.imread(Right_Img_Path)
#                 Right_Img =cv2.resize(Right_Img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
                Img.append(Center_Img)
                Img.append(Left_Img)
                Img.append(Right_Img)
                Img.append(np.fliplr(Center_Img))
                Img.append(np.fliplr(Left_Img))
                Img.append(np.fliplr(Right_Img))

                Center_Meas = float(Line[3])
                Left_Meas = Center_Meas + Meas_Correction
                Right_Meas = Center_Meas - Meas_Correction
                Meas.append(Center_Meas)
                Meas.append(Left_Meas)
                Meas.append(Right_Meas)
                Meas.append(-Center_Meas)
                Meas.append(-Left_Meas)
                Meas.append(-Right_Meas)
                
            X_train = np.array(Img)
            y_train = np.array(Meas)
            yield shuffle(X_train, y_train)

# Set our batch size
Batch_Size=16

# compile and train the model using the generator function
Train_Generator = generator(Train_Samples, Batch_Size=Batch_Size)
Valid_Generator = generator(Valid_Samples, Batch_Size=Batch_Size)


Model.fit_generator(Train_Generator, steps_per_epoch=ceil(len(Train_Samples)/Batch_Size),
            validation_data=Valid_Generator,
            validation_steps=ceil(len(Valid_Samples)/Batch_Size),
            epochs=3)
Model.save('model_Enhanced.h5')
