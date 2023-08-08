# necessary libraries
import pathlib
import gradio as gr
import splitfolders
import tensorflow as tf
import numpy as np

from models.img_clasifier import ImagesClassifier
from models.inceptionv3 import Inception
from models.img_clasifier import ImagesClassifier
from models.chexnet import CheXNetModel
from models.densenet201 import DenseNet201Model
# from models.resnet import ResNet50Model
from gen.data import Generate
from util.utils import Utils
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

base_path = "./flowers"
base_path = pathlib.Path(base_path)
# splitfolders.ratio(base_path, output='img_classify', seed=123, ratio=(0.7, 0.15, 0.15),
#                   group_prefix=None)


# Generating the dataset
data = Generate(base_path=base_path, target_size=(75, 75), batch_size=32)


li, train_data, val_data = data.generate_data()

# Instance of the inception model
classifier = ImagesClassifier(train_data=train_data, val_data=val_data, num_epochs=15, verbose=1)
inception = Inception(train_data=train_data, val_data=val_data, num_epochs=100, verbose=1)
densenet121 = CheXNetModel(train_data=train_data, val_data=val_data, num_epochs=50, verbose=1)
densenet201 = DenseNet201Model(train_data=train_data, val_data=val_data, num_epochs=50, verbose=1)

is_train = False


def average_prediction(prediction_list):
    num_models = len(prediction_list)
    averaged_prediction = sum(prediction_list) / num_models
    print(averaged_prediction)
    return averaged_prediction


if is_train:
    # Training the model
    history = classifier.forward()
    print(history.history.keys())

    # Statistics about the model, (plots, etc)
    utilities = Utils(hist=history, model=classifier, validation_data=val_data, li=li)
    utilities.forward()
else:
    # Load model weight
    densenet201.load_weights()
    inception.load_weights()
    densenet121.load_weights()
    classifier.load_weights()
    image = gr.inputs.Image(shape=(75, 75))

    label = gr.outputs.Label(num_top_classes=2, label='Densenet 201')
    label2 = gr.outputs.Label(num_top_classes=2, label='Inception v3')
    label3 = gr.outputs.Label(num_top_classes=2, label='Densenet 121')
    label4 = gr.outputs.Label(num_top_classes=2, label='Ensemble layer')
    # label5 = gr.outputs.Label(num_top_classes=2, label='img_classify')


    def predict_image(input_img):
        class_names = ['Normal', 'Pneumonia']
        input_img = input_img.reshape(75, 75, -1)
        input_img = tf.keras.utils.img_to_array(input_img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = input_img / 255

        # input_img = np.repeat(input_img, 3, axis=-1)
        # input_img = np.resize(input_img, (75, 75, 3))
        # input_img = input_img / 255.0
        # input_img = np.expand_dims(input_img, axis=0)

        prediction1 = densenet201.model.predict(input_img)
        prediction2 = inception.model.predict(input_img)
        prediction3 = densenet121.model.predict(input_img)
        prediction4 = classifier.model.predict(input_img)

        m1 = prediction1.flatten()
        m2 = prediction2.flatten()
        m3 = prediction3.flatten()
        m4 = prediction4.flatten()

        # Image classifier
        if m4 < 0.5:
            class_names = ['Non-X-ray', 'X-ray']
            d = 1 - prediction4[0]
            prediction4 = np.insert(prediction4, 0, d)

            return {class_names[i]: float(prediction4[i]) for i in range(2)},

        else:
            d = 1 - prediction4[0]
            prediction4 = np.insert(prediction4, 0, d)
            # Densenet201
            if m1 < 0.5:
                d = 1 - prediction1[0]
                prediction1 = np.insert(prediction1, 0, d)
            else:
                d = 1 - prediction1[0]
                prediction1 = np.insert(prediction1, 0, d)

            # InceptionV3
            if m2 < 0.5:
                d = 1 - prediction2[0]
                prediction2 = np.insert(prediction2, 0, d)
            else:
                d = 1 - prediction2[0]
                prediction2 = np.insert(prediction2, 0, d)

            # Densenet121
            if m3 < 0.5:
                d = 1 - prediction3[0]
                prediction3 = np.insert(prediction3, 0, d)
            else:
                d = 1 - prediction3[0]
                prediction3 = np.insert(prediction3, 0, d)

            ensemble = average_prediction([prediction1, prediction2, prediction3])

            return {class_names[i]: float(prediction1[i]) for i in range(2)}, \
                {class_names[i]: float(prediction2[i]) for i in range(2)}, \
                {class_names[i]: float(prediction3[i]) for i in range(2)}, \
                {class_names[i]: float(ensemble[i]) for i in range(2)},


    gr.Interface(
        fn=predict_image,
        inputs=image,
        outputs=[label, label2, label3, label4,],
        interpretation='default'
    ).launch(debug='True')
