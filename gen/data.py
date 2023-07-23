from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


class Generate():
    def __init__(self, base_path, target_size, batch_size):
        self.base_path = base_path
        self.target_size = target_size
        self.batch_size = batch_size

    def generate_data(self):
        # Generating training data
        print("Training data : ")
        train_datagen = ImageDataGenerator(
            rescale=1 / 255,
            zoom_range=0.3,
            rotation_range=50,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        train_data = train_datagen.flow_from_directory(
            self.base_path,
            target_size=self.target_size,
            class_mode="binary",
            batch_size=self.batch_size,
            subset="training"
        )

        # Generating validation data
        print("\nvalidation data :")
        val_datagen = ImageDataGenerator(
            rescale=1 / 255,
            validation_split=0.2
        )

        val_data = val_datagen.flow_from_directory(
            self.base_path,
            target_size=self.target_size,
            class_mode="binary",
            batch_size=self.batch_size,
            shuffle=False,
            subset="validation"
        )

        # Storing the class names for classification
        class_names = train_data.class_indices
        li = list(class_names)
        return li, train_data, val_data
