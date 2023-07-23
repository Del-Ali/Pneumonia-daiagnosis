from keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import Dense, Flatten
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
import numpy as np
from keras import optimizers


class Inception:
    def __init__(self, train_data, val_data, num_epochs, verbose):
        # Initial variables
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = num_epochs
        self.verbose = verbose

        # Inception model instance variables
        self.inception_model = InceptionV3(input_shape=(75, 75, 3), include_top=False, weights="imagenet")
        self.last_output = self.inception_model.layers[-1].output
        self.last_output = Flatten()(self.last_output)
        self.pretrained_model = Model(self.inception_model.input, self.last_output)
        self.model = Model()

    def generate_layers(self):
        # layer 1
        x = Dense(units=512, activation="relu")(self.last_output)
        x = Dropout(0.2)(x)

        # layer 2
        x = Dense(units=128, activation="relu")(x)
        x = Dropout(0.2)(x)

        # output layer
        x = Dense(units=1, activation="sigmoid")(x)

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
            loss="binary_crossentropy",
            optimizer=optimizers.RMSprop(learning_rate=1e-4),
            metrics=["accuracy"]
        )
        file_path = "inception.h5"
        self.model.load_weights(filepath=file_path)

    def forward(self):
        self.generate_layers()
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.RMSprop(learning_rate=1e-4),
            metrics=["accuracy"]
        )
        # Freezing already pretrained model of inceptionV3
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
        filepath = "inception.h5"
        self.model.save(filepath)
        return history
