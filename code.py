import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2  # Replace with 'from PIL import Image' if using Pillow
from sklearn.preprocessing import LabelEncoder  # Optional for categorical labels
import os  # For file path manipulation
import pandas as pd  # Assuming pandas for CSV handling (if applicable)


# Define data path (replace with your actual path)
data_path = r"C:\Users\aleyn\OneDrive\Masaüstü\deep l.1\Fashion144k_v1"
target_label = "fashionability_score"  # Adjust based on your data labels (if applicable)
label_file = None  # Set path to label file if using separate CSV


def load_data(data_path, target_label=None, label_file=None):
  # Assuming images are in subfolders (Scenario 1)
  if not label_file:
    images = []
    labels = []
    for folder in os.listdir(data_path):
      class_label = folder
      for filename in os.listdir(os.path.join(data_path, folder)):
        img_path = os.path.join(data_path, folder, filename)
        img = cv2.imread(img_path)  # Replace with your image loading function
        if img is not None:
          img = cv2.resize(img, (224, 224))
          images.append(img)
          if target_label:
            labels.append(class_label)
      print(f"Loaded {len(images)} images from subfolders so far")  # Print for verification

  # Load data from separate CSV (Scenario 2)
  else:
    data = pd.read_csv(label_file)
    images = []
    labels = []
    for index, row in data.iterrows():
      img_path = os.path.join(data_path, row['image_id'] + '.jpg')  # Adjust based on your CSV format
      img = cv2.imread(img_path)
      if img is not None:
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(row['label'])  # Assuming 'label' is the label column name
      print(f"Loaded image {index+1} from CSV")  # Print for verification

  # Preprocess images (e.g., normalize pixel values)
  images = np.array(images)
  images = images.astype('float32') / 255.0  # Normalize pixel values

  # Convert labels to categorical if needed (modify based on your data)
  if target_label:
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    labels = keras.utils.to_categorical(labels)

  return images, labels


def create_fashion_model(include_crf=False):
  # Define input layer (adjust for your image size)
  inputs = keras.layers.Input(shape=(224, 224, 3))

  # Pre-trained VGG16 model (include top=False to exclude final layers)
  base_model = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)

  # Freeze pre-trained layers for fine-tuning
  base_model.trainable = False

  # Add additional convolutional layers (optional)
  # ... (add your custom layers here)

  # Flatten the data
  x = base_model.output
  x = keras.layers.Flatten()(x)

  # Dense layers with ReLU activation
  x = keras.layers.Dense(128, activation="relu")(x)
  x = keras.layers.Dense(64, activation="relu")(x)

  # Output layer with single neuron for fashionability score (sigmoid for probability)
  outputs = keras.layers.Dense(1, activation="sigmoid")(x)

  # Model with or without CRF (Conditional Random Field)
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
"""
  # Include CRF (replace with your CRF implementation if desired)
  if include_crf:
    # ... (implement your CRF layer here)
    # This section is for reference only and requires further research

  return model
"""

# Train-test split with data augmentation (replace with your training logic)
def train_model(images, labels, test_size=0.2, validation_split=0.1):
  X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size)
  datagen = ImageDataGenerator(rotation_range=20, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
  datagen.fit(X_train)

  X_train, X_val = train_test_split(X_train, test_size=validation_split, shuffle=True)
  train_generator = datagen.flow(X_train, y_train, batch_size=32)
  validation_generator = datagen.flow(X_val, y_val, batch_size=32)

  # ... (training logic using the model, train_generator, and validation_generator)

# Example usage (replace with your training loop)
if __name__ == "__main__":
  # Load data (replace with your data loading logic)
  images, labels = load_data(data_path, target_label)

  # Train the model
  model = create_fashion_model()
  train_model(images, labels)

