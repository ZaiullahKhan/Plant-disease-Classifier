
# plant_classifier.py

import os
import pandas as pd
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Step 1: Organize Dataset
def prepare_dataset():
    df = pd.read_csv('data/train.csv')
    img_dir = 'data/images'
    output_dir = 'dataset/train'

    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        img_name = row['image_id'] + '.jpg'
        labels = row.drop('image_id')
        label = labels.idxmax()  # Get the label column name with value 1

        label_folder = os.path.join(output_dir, label)
        os.makedirs(label_folder, exist_ok=True)

        src = os.path.join(img_dir, img_name)
        dst = os.path.join(label_folder, img_name)

        if os.path.exists(src):
            shutil.copy(src, dst)
    print("✅ Dataset organized successfully.")


# Step 2: Training the Model
def train_model():
    image_size = 224
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=val_generator, epochs=10)

    os.makedirs('outputs', exist_ok=True)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Model Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    plt.show()

    os.makedirs('model', exist_ok=True)
    model.save('model/plant_disease_model.h5')
    print("✅ Model saved as 'model/plant_disease_model.h5'")


# Step 3: Prediction on a new image
def predict_image(img_path):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np

    image_size = 224
    model = load_model('model/plant_disease_model.h5')
    class_names = sorted(os.listdir('dataset/train'))

    img = image.load_img(img_path, target_size=(image_size, image_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    print(f"🟩 Prediction: {predicted_class}")


# Entry point
if __name__ == '__main__':
    prepare_dataset()
    train_model()
    # Example for testing after training:
    # predict_image('dataset/train/scab/xxxx.jpg')
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_images_in_folder(folder_path):
    model = load_model('model/plant_disease_model.h5')
    class_labels = list(train_generator.class_indices.keys())

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            print(f"🖼️ {filename}: {class_labels[class_index]} ({confidence:.2f}%)")
            predict_images_in_folder("test")

import matplotlib.pyplot as plt


# Plot training history
def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Acc')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('outputs/accuracy.png')
    print("📈 Saved training plot to 'outputs/accuracy.png'")
    plt.show()

# Call it right after model.fit()
plot_training(history)
# Reload model history from previous training (if saved) OR replot
plot_training(history)
predict_images_in_folder("test")
