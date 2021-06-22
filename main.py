"""
Developed by PIM-Tech
2151835 - İbrahim Aydın
2243392 - Pınar Dilbaz
2243384 - Muhammed Didin
"""
#importing libraries
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

#declaring lists as global variables
img_list = list()
train_list = list()
valid_list = list()
test_list = list()
train_result = list()
valid_result = list()
test_result = list()
#function that reads, crops and formats images. Splits test-valid-train lists
def croping_images():
    global train_list
    global train_result
    global test_list
    global test_result
    global valid_list
    global valid_result
    check = ['.bmp']
    param = list()
    
    #read parameters.txt 
    with open("Database/parameters.txt") as parameters:
        #loop line by line.
        for line in parameters:
            #if the line contains image, read...
            if any(x in line for x in check):
                #split by ',' and append the param list.
                param.append ([x.strip() for x in line.split(',')])
    """
    param = parameters list
    param[x][0] = image name
    param[x][1] = x coordinate
    param[x][2] = y coordinate
    param[x][3] = radius
    """
    #result counters
    j=0
    a=1
    b=1
    c=1
    #read the iamges.
    for i in param:
        #read images as grayscale.
        img = cv2.imread("Database/"+i[0],cv2.IMREAD_GRAYSCALE)

        #take image info from param about eyelid.
        x = int(i[1])
        y = int(i[2])
        r = int(i[3])
        j = j % 4
        
        if(j==0 or j==1):
            #crop, resize and format image.
            resized =cv2.resize(img[x-r:x+r,y-r:y+r], (128,128))
            resized = resized.reshape(128,128,1)
            #scale the image in the range 0,1
            resized = resized / 255.0
            #append the first 2 image in train_list
            train_list.append(resized)
            #append the result in train_result.
            train_result.append([a])
            if(j==1):
                a+=1
        elif(j==2):
            #crop, resize and format image.
            resized = cv2.resize(img[x-r:x+r,y-r:y+r], (128,128))
            resized = resized.reshape(128,128,1)
            #scale the image in the range 0,1
            resized = resized / 255.0
            #append the 3rd image in valid_list.
            valid_list.append(resized)
            #append the result in valid_result.
            valid_result.append([b])
            b+=1
        else:
            #crop, resize and format image
            resized = cv2.resize(img[x-r:x+r,y-r:y+r], (128,128))
            resized = resized.reshape(128,128,1)
            #scale the image in the range 0,1
            resized = resized / 255.0
            #append the last image in test_list
            test_list.append(resized)
            #append the result in test_result.
            test_result.append([c])
            c+=1
        j+=1

    #organize lists for model.
    ohe = OneHotEncoder()
    train_result = ohe.fit_transform(train_result).toarray()
    train_result = np.array(train_result,dtype=np.float)
    train_list = np.array(train_list,dtype=np.float)

    test_result = ohe.fit_transform(test_result).toarray()
    test_list = np.array(test_list,dtype=np.float)
    test_result = np.array(test_result,dtype=np.float)

    valid_result = ohe.fit_transform(valid_result).toarray()
    valid_list = np.array(valid_list,dtype=np.float)
    valid_result = np.array(valid_result,dtype=np.float)

    
def recognition():
    global train_list
    global train_result
    global test_list
    global test_result
    global valid_list
    global valid_result

    #declare parameters.
    output_len = 400
    batch_size = 128
    epoch = 250

    #create model as Sequential.
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3,3),activation='linear',padding='same',input_shape=(128,128,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3,3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                     
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))           
    model.add(Dropout(0.3))
    model.add(Dense(output_len, activation='softmax'))

    #compile model.
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(),metrics=['accuracy'])

    model_train = model.fit(train_list, train_result, batch_size=batch_size,epochs=epoch,verbose=1,validation_data=(valid_list, valid_result))
    test_eval = model.evaluate(test_list, test_result, verbose=0)
    model.summary()
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1]*100)

if __name__ == "__main__":
    croping_images()
    recognition()