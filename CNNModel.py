import numpy as np
import pandas as pd

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix


class CNNModel(object):

    def __init__(self, model_version=1):

        # Iniitalize Models
        if model_version == 1:
            self.piece_model = self.advanced_cnn_init(0)
            self.color_model = self.advanced_cnn_init(1)
        else:
            self.piece_model = self.baseline_cnn_init(0)
            self.color_model = self.baseline_cnn_init(1)

    def baseline_cnn_init(self,model_type):
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
        self.piece_model.compile(
            loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.color_model.compile(
            loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    def train_model(self,augment=1):
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
            filepath='data/piece_model.weights.best.hdf5', save_best_only=True,verbose=1)
        self.color_checkpointer = ModelCheckpoint(
            filepath='data/color_model.weights.best.hdf5', save_best_only=True,verbose=1)

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
        return generator.flow_from_directory(
            directory=path,
            target_size=(135, 135),
            color_mode='grayscale',
            class_mode='categorical',
            seed=42)

    def calc_step_size(self,gen_iter):
        return gen_iter.n/gen_iter.batch_size

    def load_model_best_weights(self):
        self.piece_model.load_weights('data/models/piece_model.weights.best.hdf5')
        self.color_model.load_weights('data/models/color_model.weights.best.hdf5')

    def test_models(self):
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
