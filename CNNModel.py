"""Summary
"""
import numpy as np
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


class CNNModel(object):

    """Summary

    CNNModel is the class that defines the architecture and has functions for training, testing
    loading weights and predicting results based on the model.
    
    Attributes:
        color_checkpointer (ModelCheckpoint): 
        color_hist (History): Keras Model training history for accuracy and loss
        color_model (Model): 
        color_score (Keras Metric): Accuracy of color model
        color_test_datagen (ImageDataGenerator): 
        color_test_iter (ImageDataGenerator Iterator): 
        color_test_step_size (int): 
        color_train_datagen (ImageDataGenerator):
        color_train_iter (ImageDataGenerator Iterator): 
        color_train_step_size (int):
        color_valid_datagen (ImageDataGenerator): 
        color_valid_iter (ImageDataGenerator Iterator): 
        color_valid_step_size (int): 
        piece_checkpointer (ModelCheckpoint): 
        piece_hist (History): Keras Model training history for accuracy and loss
        piece_model (Model): 
        piece_score (Keras Metric): Accuracy of piece model
        piece_test_datagen (ImageDataGenerator): 
        piece_test_iter (ImageDataGenerator Iterator): 
        piece_test_step_size (int): 
        piece_train_datagen (ImageDataGenerator): 
        piece_train_iter (ImageDataGenerator Iterator): 
        piece_train_step_size (int): 
        piece_valid_datagen (ImageDataGenerator): 
        piece_valid_iter (ImageDataGenerator Iterator): 
        piece_valid_step_size (int): 
    """
    
    def __init__(self, model_version=1):
        """Summary
        
        Initialize CNN model object with specificed architecture

        Args:
            model_version (int, optional): 0 for baseline, 1 for final/advancedd
        """
        # Iniitalize Models
        if model_version == 1:
            self.piece_model = self.advanced_cnn_init(0)
            self.color_model = self.advanced_cnn_init(1)
        else:
            self.piece_model = self.baseline_cnn_init(0)
            self.color_model = self.baseline_cnn_init(1)

    def baseline_cnn_init(self,model_type):
        """Summary
        
        Create baseline CNN model for either a piece or color path

        Args:
            model_type (int): 0 for piece model, 1 for color model
        
        Returns:
            model: Keras model
        """
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu',
                         input_shape=(135, 135, 1)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=2,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=2,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(GlobalAveragePooling2D())
        if model_type == 1:
            model.add(Dense(3, activation='softmax'))
        else:
            model.add(Dense(7, activation='softmax'))
        return model

    def advanced_cnn_init(self,model_type):
        """Summary
        
        Create advanced CNN model for either piece or color path

        Args:
            model_type (int): 0 for piece model, 1 for color model
        
        Returns:
            model: Keras model with architecture for final/advanced model
        """
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu',
                         input_shape=(135, 135, 1)))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=3,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=2,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=2,
                         padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=2,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=2,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=2,
                         padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(2048, activation='relu'))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.4))
        if model_type == 1:
            model.add(Dense(3, activation='softmax'))
        else:
            model.add(Dense(7, activation='softmax'))

        return model

    def compile_model(self):
        """Summary

        Compiles the keras model

        """
        self.piece_model.compile(
            loss='categorical_crossentropy',optimizer=RMSprop(lr=0.00001), metrics=['accuracy'])
        self.color_model.compile(
            loss='categorical_crossentropy', optimizer=RMSprop(lr=0.00001), metrics=['accuracy'])

    def train_model(self,augment=1):
        """Summary
        
        Flows training and validation data into object and performs training step

        Args:
            augment (int, optional): Sets data generator augmentation parameters
        """
        # Initialize Datagenerators
        if augment:
            self.piece_train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)

            self.color_train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True)
        else:
            self.piece_train_datagen = ImageDataGenerator(rescale=1./255)
            self.color_train_datagen = ImageDataGenerator(rescale=1./255)

            self.piece_valid_datagen = ImageDataGenerator(rescale=1./255)
        self.color_valid_datagen = ImageDataGenerator(rescale=1./255)

        # Flow Data from directories
        self.piece_train_iter = self.data_gen_flow(
            self.piece_train_datagen, 'data/piece_data/train')
        self.piece_valid_iter = self.data_gen_flow(
            self.piece_valid_datagen, 'data/piece_data/valid')
        # Flow Data from directories
        self.color_train_iter = self.data_gen_flow(
            self.color_train_datagen, 'data/color_data/train')
        self.color_valid_iter = self.data_gen_flow(
            self.color_valid_datagen, 'data/color_data/valid')

        # Calculate step sizes
        self.piece_train_step_size = self.calc_step_size(self.piece_train_iter)
        self.piece_valid_step_size = self.calc_step_size(self.piece_valid_iter)
        self.color_train_step_size = self.calc_step_size(self.color_train_iter)
        self.color_valid_step_size = self.calc_step_size(self.color_valid_iter)

        # Create checkpointers and fit models
        self.piece_checkpointer = ModelCheckpoint(
            filepath='data/models/piece_model.weights.best.hdf5', save_best_only=True,verbose=1)
        self.color_checkpointer = ModelCheckpoint(
            filepath='data/models/color_model.weights.best.hdf5', save_best_only=True,verbose=1)

        self.piece_hist = self.piece_model.fit_generator(
            generator=self.piece_train_iter,
            steps_per_epoch=self.piece_train_step_size,
            validation_data=self.piece_valid_iter,
            validation_steps=self.piece_valid_step_size,
            epochs=100,
            callbacks=[self.piece_checkpointer],
            verbose=2
        )

        self.color_hist = self.color_model.fit_generator(
            generator=self.color_train_iter,
            steps_per_epoch=self.color_train_step_size,
            validation_data=self.color_valid_iter,
            validation_steps=self.color_valid_step_size,
            epochs=100,
            callbacks=[self.color_checkpointer],
            verbose=2
        )

    def data_gen_flow(self,generator, path):
        """Summary
        
        Helper function to flow data from directors into generator objects

        Args:
            generator (KerasImageDataGenerator): generator to flow images into 
            path (string): path to data images
        
        Returns:
            KerasImageDataGenerator: generator with flowed in data
        """
        return generator.flow_from_directory(
            directory=path,
            target_size=(135, 135),
            color_mode='grayscale',
            class_mode='categorical',
            seed=42)

    def calc_step_size(self,gen_iter):
        """Summary
        
        Calculates step size for training

        Args:
            gen_iter (ImageDataGenerator Iterator): generator iterator to calculate step size
        
        Returns:
            int: step size
        """
        return gen_iter.n/gen_iter.batch_size

    def load_model_best_weights(self):
        """Summary

		Load weights stroed in files

        """
        self.piece_model.load_weights('data/models/piece_model.weights.best.hdf5')
        self.color_model.load_weights('data/models/color_model.weights.best.hdf5')

    def test_models(self):
        """Summary
	
		Test model based on loaded weights

        """
        # Create Test generators, flow in data and define step size
        self.piece_test_datagen = ImageDataGenerator(rescale=1./255)
        self.color_test_datagen = ImageDataGenerator(rescale=1./255)
        self.piece_test_iter = self.data_gen_flow(
            self.piece_test_datagen, 'data/piece_data/test')
        self.color_test_iter = self.data_gen_flow(
            self.color_test_datagen, 'data/color_data/test')
        self.piece_test_step_size = self.calc_step_size(self.piece_test_iter)
        self.color_test_step_size = self.calc_step_size(self.color_test_iter)

        self.piece_score = self.piece_model.evaluate_generator(
            generator=self.piece_test_iter,
            steps=self.piece_test_step_size)

        self.color_score = self.color_model.evaluate_generator(
            generator=self.color_test_iter,
            steps=self.color_test_step_size)

        print('\n', 'Piece Test accuracy:', self.piece_score[1])
        print('\n', 'Color Test accuracy:', self.color_score[1])

    def predict_image(self,img):
    	"""Summary
    	
    	Predict piece type based on input 135x135 image

    	Args:
    	    img (array): Array containing cv2 grayscale image data for 135x135 image
    	
    	Returns:
    	    [char, char]: piece prediction char, color prediction char
    	"""
    	#Rescale image
    	img = np.array(img)/255.0
    	img = np.reshape(img, (1,135,135,1))

    	piece_pred = np.argmax(self.piece_model.predict(img))
    	color_pred = np.argmax(self.color_model.predict(img))

    	piece_dict = {0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r', 6: 's'}
    	color_dict = {0: 'b', 2: 'w', 1: 'e'}

    	return piece_dict[piece_pred], color_dict[color_pred]