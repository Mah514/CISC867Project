import os
import cv2
from PIL import Image
import numpy as np
import keras
import tensorflow as tf

import csv

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", 
    "Mass", "Nodule", "Pneumonia", "Pneumothorax", 
    "Consolidation", "Edema", "Emphysema", "Fibrosis", 
    "Pleural_Thickening", "Hernia"
]

def get_binary_encoding(file_name):
    
    all_encodings = []
    
    with open(file_name, 'r') as f:
        for line in f.readlines():
            
            line_list = line.strip().split()
            
            del(line_list[0])
            
            encoding = [int(x) for x in line_list ]
                    
            all_encodings.append(encoding)
            
    return np.asarray(all_encodings)

def read_images(pathImageDirectory, pathDatasetFile):

    img_list = []

    with open(pathDatasetFile, "r") as fileDescriptor:
        line = True

        while line:
            line = fileDescriptor.readline()

            if line:
                lineItems = line.split()
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                if (len(img_list) % 100 == 0):
                    print("appending image " + str(len(img_list)))
                img = cv2.imread(imagePath)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (100, 100))
                img_list.append(img)
    
    return np.asarray(img_list)


def main():
    
    tf.config.list_physical_devices('GPU')
    
    train_folder = 'CXR8/training_images'
    val_folder = 'CXR8/validation_images'

    # Read the train and test image lists
    train_images = read_images(pathImageDirectory='CXR8/images/all_images', pathDatasetFile="Xray14_train_official.txt")
    val_images = read_images(pathImageDirectory='CXR8/images/all_images', pathDatasetFile="Xray14_val_official.txt")
    #val_images = read_images(val_images_list, val_folder)
    #test_images = read_file("Xray14_test_official.txt")
    
    train_label = get_binary_encoding("CXR8/Xray14_train_official.txt")
    val_label = get_binary_encoding("CXR8/Xray14_val_official.txt")

    print(train_images.shape)
    print(train_label.shape)

    print(type(train_images))
    print(type(train_label))

    print(type(train_images[0]))
    print(type(train_label[0]))
    
    
    t_input = keras.Input(shape=(100, 100, 3))
    
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        res_model = keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=t_input)
    
        model =keras.models.Sequential()
        model.add(res_model)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(len(DISEASES), activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(learning_rate=2e-5), metrics=['accuracy'])
    
    history = model.fit(train_images, train_label, batch_size=32, epochs=10, verbose=1, validation_data=(val_images, val_label))
    
    model.summary()
    
main()