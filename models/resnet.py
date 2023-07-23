import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, Model
import numpy as np


class ResNet50Model:
    def __init__(self, train_data, val_data, num_epochs, verbose):
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = num_epochs
        self.verbose = verbose

        self.resnet_model = ResNet50(input_shape=(200, 200, 3), include_top=False, weights="imagenet")
        self.last_output = self.resnet_model.layers[-1].output
        self.pretrained_model = Model(self.resnet_model.input, self.last_output)
        self.model = Model()

    def generate_layers(self):
        # Add GlobalAveragePooling2D layer
        x = GlobalAveragePooling2D()(self.last_output)

        # first layer
        x = Dense(units=512, activation="relu")(x)
        x = Dropout(0.2)(x)

        # layer 2
        x = Dense(units=128, activation="relu")(x)
        x = Dropout(0.2)(x)

        # output layer
        x = Dense(units=1, activation="sigmoid")(x)

        # final model
        self.model = Model(self.pretrained_model.input, x)

    def predict_all(self):
        return self.model.predict(self.val_data,
                                  steps=np.ceil(self.val_data.samples / self.val_data.batch_size),
                                  verbose=2)

    def load_weights(self):
        self.generate_layers()
        self.model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(learning_rate=1e-4),
            metrics=['accuracy']
        )
        file_path = 'resnet.h5'
        self.model.load_weights(filepath=file_path)

    def predict(self):
        img_path = "./train/NORMAL/IM-0115-0001.jpeg"
        new_img = tf.keras.utils.load_img(img_path, target_size=(200, 200))
        img = tf.keras.utils.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img / 255
        prediction = self.model.predict(img)
        return prediction, new_img

    def forward(self):
        self.generate_layers()
        self.model.compile(loss="binary_crossentropy",
                           optimizer='sgd',
                           metrics=["accuracy"]
                           )
        # Freeze layers
        for layer in self.pretrained_model.layers:
            layer.trainable = False

        # model fitting
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=self.epochs,
            verbose=self.verbose
        )
        filepath = 'resnet.h5'
        self.model.save(filepath)
        return history
