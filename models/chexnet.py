from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
from keras import optimizers
import tensorflow as tf


class CheXNetModel:
    def __init__(self, train_data, val_data, num_epochs, verbose):
        # Initial variables
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = num_epochs
        self.verbose = verbose

        # CheXNet model instance variables
        self.densenet_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        self.last_output = self.densenet_model.output
        self.last_output = GlobalAveragePooling2D()(self.last_output)
        self.pretrained_model = Model(self.densenet_model.input, self.last_output)
        self.model = Model()

    def generate_layers(self):
        # layer 1
        x = Dense(units=512, activation='relu')(self.last_output)
        x = Dropout(0.2)(x)

        # layer 2
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(0.2)(x)

        # output layer
        x = Dense(units=1, activation='sigmoid')(x)

        # final model
        self.model = Model(self.pretrained_model.input, x)

    def predict_all(self):
        return self.model.predict(
            self.val_data,
            steps=np.ceil(self.val_data.samples / self.val_data.batch_size),
            verbose=2
        )

    def load_weights(self):
        self.generate_layers()
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(learning_rate=1e-4),
            metrics=['accuracy']
        )
        file_path = 'chexnet.h5'
        self.model.load_weights(filepath=file_path)
        self.predict()

    def predict(self):
        image_path = "./train/NORMAL/IM-0115-0001.jpeg"
        new_img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img = tf.keras.utils.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img / 255

        # print("Following is our prediction:")
        prediction = self.model.predict(img)
        return prediction, new_img

    def forward(self):
        self.generate_layers()
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(learning_rate=1e-4),
            metrics=['accuracy']
        )
        # Freezing already pretrained model of DenseNet121
        for layer in self.pretrained_model.layers:
            layer.trainable = False

        history = self.model.fit(
            self.train_data,
            steps_per_epoch=self.train_data.samples // self.train_data.batch_size,
            validation_data=self.val_data,
            validation_steps=self.val_data.samples // self.val_data.batch_size,
            epochs=self.epochs,
            verbose=self.verbose
        )
        file_path = 'chexnet.h5'
        self.model.save(file_path)
        return history
