import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from mlxtend.plotting import plot_confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics


class Utils():
    def __init__(self, hist, model, validation_data, li):
        self.history = hist
        self.model = model
        self.val_data = validation_data
        self.epochs = None
        if self.history:
            self.epochs = range(1, len(self.history.history["accuracy"]) + 1)
        self.cm = ""
        self.li = li

    def accuracy_plot(self):
        plt.plot(self.epochs, self.history.history["accuracy"], color="purple")
        plt.plot(self.epochs, self.history.history["val_accuracy"], color="pink")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.title("Accuracy plot")
        plt.legend(["train_acc", "val_acc"])
        plt.show()

    def loss_plot(self):
        plt.plot(self.epochs, self.history.history["loss"], color="purple")
        plt.plot(self.epochs, self.history.history["val_loss"], color="pink")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss plot")
        plt.legend(["train_loss", "val_loss"])
        plt.show()

    def prediction_val_data(self):
        prediction = self.model.predict_all()
        prediction = (prediction > 0.5)
        return prediction

    def confusion_matrix_details(self):
        prediction = self.prediction_val_data()
        self.cm = confusion_matrix(self.val_data.classes, prediction)
        plot_confusion_matrix(self.cm, figsize=(5, 5))
        print(accuracy_score(self.val_data.classes, prediction))
        print(classification_report(self.val_data.classes, prediction))

        # Creating the confusion matrix
        self.cm = metrics.confusion_matrix(self.val_data.classes, prediction)
        # Assigning columns names
        cm_df = pd.DataFrame(self.cm,
                             columns=['Predicted Negative', 'Predicted Positive'],
                             index=['Actual Negative', 'Actual Positive'])
        # Showing the confusion matrix
        print(cm_df)

    # Creating a function to report confusion metrics
    def confusion_metrics(self):
        # save confusion matrix and slice into four pieces
        TP = self.cm[1][1]
        TN = self.cm[0][0]
        FP = self.cm[0][1]
        FN = self.cm[1][0]
        print('True Positives:', TP)
        print('True Negatives:', TN)
        print('False Positives:', FP)
        print('False Negatives:', FN)

        # calculate accuracy
        conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN))

        # calculate mis-classification
        conf_misclassification = 1 - conf_accuracy

        # calculate the sensitivity
        conf_sensitivity = (TP / float(TP + FN))
        # calculate the specificity
        conf_specificity = (TN / float(TN + FP))

        # calculate precision
        conf_precision = (TN / float(TN + FP))
        # calculate f_1 score
        conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
        print('-' * 50)
        print(f'Accuracy: {round(conf_accuracy, 2)}')
        print(f'Mis-Classification: {round(conf_misclassification, 2)}')
        print(f'Sensitivity: {round(conf_sensitivity, 2)}')
        print(f'Specificity: {round(conf_specificity, 2)}')
        print(f'Precision: {round(conf_precision, 2)}')
        print(f'f_1 Score: {round(conf_f1, 2)}')

    def roc_curve(self):
        prediction = self.prediction_val_data()
        fpr, tpr, threshold = roc_curve(self.val_data.classes, prediction)
        auc_inception = auc(fpr, tpr)

        plt.figure(figsize=(5, 5), dpi=100)
        plt.plot(fpr, tpr, linestyle='-', label='Xception (auc = %0.3f)' % auc_inception)

        plt.xlabel('False Positive Rate -->')
        plt.ylabel('True Positive Rate -->')

        plt.legend()
        plt.title("ROC curve")
        plt.show()

    def imshow_predict(self, prediction, new_img):
        print((prediction.shape))
        print(prediction)
        d = prediction.flatten()
        j = d.max()
        for index, item in enumerate(d):
            if item == j:
                class_name = self.li[index]

        plt.figure(figsize=(4, 4))
        plt.imshow(new_img)
        plt.axis('off')
        plt.title(class_name)
        plt.show()

    def forward(self):
        self.accuracy_plot()
        self.loss_plot()
        self.confusion_matrix_details()
        self.confusion_metrics()
        self.roc_curve()
