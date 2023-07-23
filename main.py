# necessary libraries
import pathlib
import gradio as gr
import tensorflow as tf
import numpy as np

from models.inceptionv3 import Inception
from models.inception import InceptionVv3
from models.chexnet import CheXNetModel
from models.densenet201 import DenseNet201Model
# from models.resnet import ResNet50Model
from gen.data import Generate
from util.utils import Utils
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score

base_path = "./train"
base_path = pathlib.Path(base_path)

# Generating the dataset
data = Generate(base_path=base_path, target_size=(75, 75), batch_size=32)
li, train_data, val_data = data.generate_data()

# Instance of the inception model
inception = Inception(train_data=train_data, val_data=val_data, num_epochs=15, verbose=1)
densenet121 = CheXNetModel(train_data=train_data, val_data=val_data, num_epochs=15, verbose=1)
densenet201 = DenseNet201Model(train_data=train_data, val_data=val_data, num_epochs=15, verbose=1)
ic = InceptionVv3(train_data = train_data, val_data=val_data, num_epochs=15, verbose=1)
# resnet = ResNet50Model(train_data=train_data, val_data=val_data, num_epochs=10, verbose=1)
is_train = True
if is_train:
    # Training the model
    history = densenet121.forward()
    print(history.history.keys())

    # Statistics about the model, (plots, etc)
    utilities = Utils(hist=history, model=densenet121, validation_data=val_data, li=li)
    utilities.forward()
else:
    # Load model weight
    densenet201.load_weights()
    inception.load_weights()
    densenet121.load_weights()
    ic.load_weights()
   # resnet.load_weights()
    image = gr.inputs.Image(shape=(75, 75))
    label = gr.outputs.Label(num_top_classes=2)
    label2 = gr.outputs.Label(num_top_classes=2)
    label3 = gr.outputs.Label(num_top_classes=2)
    label4 = gr.outputs.Label(num_top_classes=2)
   #  label_vc = gr.outputs.Label(num_top_classes=2)



    def predict_image(input_img):
        class_names = ['Normal', 'Pneumonia']

        input_img = input_img.reshape(75, 75, -1)
        input_img = tf.keras.utils.img_to_array(input_img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = input_img / 255
        prediction1 = densenet201.model.predict(input_img)
        prediction2 = inception.model.predict(input_img)
        prediction3 = densenet121.model.predict(input_img)
        prediction4 = ic.model.predict(input_img)
        m1 = prediction1.flatten()
        m2 = prediction2.flatten()
        m3 = prediction3.flatten()
        m4 = prediction4.flatten()

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

            # ic
            if m4 < 0.5:
                d = 1 - prediction3[0]
                prediction4 = np.insert(prediction4, 0, d)
            else:
                d = 1 - prediction4[0]
                prediction4 = np.insert(prediction4, 0, d)

        return {class_names[i]: float(prediction1[i]) for i in range(2)}, \
            {class_names[i]: float(prediction2[i]) for i in range(2)}, \
            {class_names[i]: float(prediction3[i]) for i in range(2)}, \
            {class_names[i]: float(prediction4[i]) for i in range(2)},




    gr.Interface(
        fn=predict_image,
        inputs=image,
        outputs=[label, label2, label3, label4],
        interpretation='default'
    ).launch(debug='True')
