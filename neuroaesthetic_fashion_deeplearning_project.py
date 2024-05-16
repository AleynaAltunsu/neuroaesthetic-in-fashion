"""import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from tensorflow.keras.applications import VGG16  # Pre-trained model (optional)

# Data preprocessing functions (replace with your implementation)
def load_images(data_dir):
    # Load images from the directory
    # ... (Your image loading and preprocessing logic) ...
    return images, labels

def preprocess_text(text):
    # Clean and standardize text descriptions (optional)
    # ... (Your text preprocessing logic) ...
    return preprocessed_text

def preprocess_user_interaction_data(data_path):
    # Load and preprocess user interaction data (likes, comments) (optional)
    # ... (Your user interaction data loading and preprocessing logic) ...
    return user_interaction_features

# Data loading and splitting
train_images, train_labels = load_images(train_data_dir)
val_images, val_labels = load_images(val_data_dir)
test_images, test_labels = load_images(test_data_dir)

# Preprocess text descriptions (optional)
train_descriptions = preprocess_text(train_text_descriptions)  # Replace if applicable
val_descriptions = preprocess_text(val_text_descriptions)  # Replace if applicable
test_descriptions = preprocess_text(test_text_descriptions)  # Replace if applicable

# Load user interaction data (optional)
if user_interaction_data_path:
    user_interaction_features = preprocess_user_interaction_data(user_interaction_data_path)

# Data augmentation (optional)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load images using data generators (or modify for other data)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'  # Assuming labels indicate fashionability (0/1)
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

# Model definition (choose baseline or advanced architecture)

# a) Baseline CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for fashion probability (0-1)
])

# b) Advanced Techniques (Optional - Modify based on data availability)

# i) Attention Mechanism (Example)
from tensorflow.keras.layers import Attention

# ... (Baseline CNN model definition up to the last convolutional layer) ...

last_convolutional_layer
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from tensorflow.keras.applications import VGG16  # Pre-trained model (optional)

# ... (Data preprocessing functions and data loading/splitting sections) ...

# Model definition (choose baseline or advanced architecture)

# a) Baseline CNN Model
model = Sequential([
    # ... (Baseline CNN layers) ...
])

# b) Advanced Techniques (Optional - Modify based on data availability)

# i) Attention Mechanism (Example)
from tensorflow.keras.layers import Attention

#if user_interaction_data_path:  # Include attention only if user interaction data exists

    # ... (Baseline CNN model definition up to the last convolutional layer) ...

#    last_convolutional_layer_output = ...  # Replace with the output of the last convolutional layer

    # Attention module example
#    attention_layer = Attention()([last_convolutional_layer_output, last_convolutional_layer_output])

    # ... (Following fully connected layers, incorporating the attention output) ...

# ii) Multimodal Learning (Optional - Modify based on data availability)
#if user_interaction_data_path:

    # ... (Baseline CNN model definition for image processing) ...

    # Load and preprocess user interaction data (already defined in data loading section)

    # Concatenate image features and user interaction features
 #   combined_features = tf.concat([image_features, user_interaction_features], axis=-1)

    # ... (Following fully connected layers for classification using combined features) ...

# Model compilation, training, and evaluation sections (unchanged)
"""Important Considerations:

Data Availability: The attention mechanism and multimodal learning sections are included conditionally based on the availability of user interaction data (user_interaction_data_path). If you don't have this data, you can remove these sections and focus on the baseline CNN model.
Attention Mechanism Implementation: The provided example with Attention() is a basic structure. You might need to explore more specific attention mechanisms depending on your research goals and dataset characteristics.
Hyperparameter Tuning and Regularization: Remember to experiment with hyperparameters (learning rate, optimizer, etc.) for optimal performance. Consider techniques like dropout layers or L1/L2 regularization to prevent overfitting, especially with deep models.
Ethical considerations:

If you do collect user interaction data, ensure it's done ethically (e.g., user consent, anonymization).
I hope this comprehensive response addresses the missing elements and provides a clearer project structure!

Kaynaklar
info
towardsdatascience.com/different-colorspaces-as-inputs-to-cnns-406ae62d1bd6
github.com/Aline-Costa/Artificial-Neural-N"""

"""
# HYPERPARAMETER VE FİNETUNİNG SONRASI FULL CODE:
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from tensorflow.keras.applications import VGG16  # Pre-trained model (optional)
from tensorflow.keras.layers import Attention  # Attention layer import

# ... (Data preprocessing functions and data loading/splitting sections) ...

# Model definition (choose baseline or advanced architecture)

# a) Baseline CNN Model
def create_baseline_model():
  model = Sequential([
      # ... (Define baseline CNN layers) ...
  ])
  return model

# b) Advanced Techniques (Optional - Modify based on data availability)

# i) Attention Mechanism (Example)
if user_interaction_data_path:  # Include attention only if user interaction data exists

  def create_attention_model():
    model = create_baseline_model()  # Call baseline model creation function

    # ... (Attention mechanism implementation using the output of the baseline model) ...

    return model

# Hyperparameter Tuning (Optional)
def tune_hyperparameters(model_fn):  # Function to accept model creation function
  # ... (Implement hyperparameter tuning logic using tools like Keras Tuner) ...
  # This function would likely use techniques like GridSearchCV or RandomizedSearchCV
  # to explore different learning rates, optimizers, batch sizes, etc. based on your needs.
  # It would then return the best performing model configuration.

# Regularization (Optional)
def add_regularization(model):
  # ... (Add regularization layers or techniques like dropout, L1/L2 regularization) ...
  # This function could modify the model passed as input to incorporate techniques
  # that help prevent overfitting, such as adding dropout layers after convolutional layers
  # or including L1/L2 regularization in the compile step.
  return model

# Model compilation, training, and evaluation sections (modify to incorporate tuning/regularization)

# Example usage (assuming tune_hyperparameters and add_regularization functions are implemented)
best_model = tune_hyperparameters(create_attention_model)  # Tune hyperparameters for attention model (if applicable)
best_model = add_regularization(best_model)  # Add regularization to the best model

# Compile the model (replace with your optimizer, loss, and metrics)
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# ... (Training code) ...
# Model compilation, training, and evaluation sections (modify to incorporate tuning/regularization)

# Example usage (assuming tune_hyperparameters and add_regularization functions are implemented)
best_model = tune_hyperparameters(create_attention_model)  # Tune hyperparameters for attention model (if applicable)
best_model = add_regularization(best_model)  # Add regularization to the best model

# Compile the model (replace with your optimizer, loss, and metrics)
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
train_generator.fit(best_model)  # Assuming you defined the data generator earlier

# Evaluate the model (on test data)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)
loss, accuracy = best_model.evaluate(test_generator)
print('Test accuracy:', accuracy)

# Additional evaluation and analysis (optional)
# - Compare performance with and without attention mechanism (if applicable)
# - Use visualization techniques like saliency maps or class activation maps to understand
#   which image regions receive higher attention.
# - Discuss the results in the context of neuroaesthetics theories and user perception of fashion.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from tensorflow.keras.applications import VGG16  # Pre-trained model (optional)
from tensorflow.keras.layers import Attention  # Attention layer import

# ... (Data preprocessing functions and data loading/splitting sections) ...

# Model definition (choose baseline or advanced architecture)

# a) Baseline CNN Model
def create_baseline_model():
    model = Sequential([
        # ... (Define baseline CNN layers) ...
    ])
    return model


# b) Advanced Techniques (Optional - Modify based on data availability)

# i) Attention Mechanism (Example)
if user_interaction_data_path:  # Include attention only if user interaction data exists

    def create_attention_model():
        model = create_baseline_model()  # Call baseline model creation function

        # ... (Attention mechanism implementation using the output of the baseline model) ...

        return model


# Hyperparameter Tuning (Optional)
#def tune_hyperparameters(model_fn):  # Function to accept model creation function
    # ... (Implement hyperparameter tuning logic using tools like Keras Tuner) ...
    # This function would likely use techniques like GridSearchCV or RandomizedSearchCV
    # to explore different learning rates, optimizers, batch sizes, etc. based on your needs.
    # It would then return the best performing model configuration.


# Regularization (Optional)
def add_regularization(model):
    # ... (Add regularization layers or techniques like dropout, L1/L2 regularization) ...
    # This function could modify the model passed as input to incorporate techniques
    # that help prevent overfitting, such as adding dropout layers after convolutional layers
    # or including L1/L2 regularization in the compile step.
    model.add(Dropout(0.2))  # Example dropout layer for regularization
    return model


# Model compilation, training, and evaluation sections (modify to incorporate tuning/regularization)

# Example usage (assuming tune_hyperparameters and add_regularization functions are implemented)
best_model = tune_hyperparameters(create_attention_model)  # Tune hyperparameters for attention model (if applicable)
best_model = add_regularization(best_model)  # Add regularization to the best model

# Compile the model (replace with your optimizer, loss, and metrics)
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# ... (Training code) ...

# Model compilation, training, and evaluation sections (modify to incorporate tuning/regularization)

# Example usage (assuming tune_hyperparameters and add_regularization functions are implemented)
best_model = tune_hyperparameters(create_attention_model)  # Tune hyperparameters for attention model (if applicable)
best_model = add_regularization(best_model)  # Add regularization to the best model

# Compile the model (replace with your optimizer, loss, and metrics)
best_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
train_generator.fit(best_model)  # Assuming you defined the data generator earlier

# Evaluate the model (on test data)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)
loss, accuracy = best_model.evaluate(test_generator)
print('Test accuracy:', accuracy)

# Additional evaluation and analysis (optional)
# - Compare performance with and without attention mechanism (if applicable)
# - Use visualization techniques like saliency maps or class activation maps to understand
#   which image regions receive higher attention.
# - Discuss the results in the context of neuroaesthetics theories and user perception of fashion.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For data augmentation
from tensorflow.keras.applications import VGG16  # Pre-trained model (optional)
from tensorflow.keras.layers import Attention  # Attention layer import

# Fashion144K Data Loading and Preprocessing

import os  # Import the os module at the beginning of your code

# ... (Rest of your code) ...

# Fashion144K Data Loading and Preprocessing

# Assuming you have downloaded the Fashion144K dataset and extracted it to a folder named 'fashion144k'
    #data_dir = 'fashion144k'

import tarfile
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf




# Define the train data directory
train_data_dir = os.path.join(data_dir, 'train')

# Ensure the directory exists
if not os.path.exists(train_data_dir):
    print(f"Directory does not exist: {train_data_dir}")
else:
    print(f"Directory exists: {train_data_dir}")

# List the contents of the train directory
try:
    for subdir in sorted(os.listdir(train_data_dir)):
        print(subdir)
except FileNotFoundError as e:
    print(f"Error: {e}")


# Define train, validation, and test data paths based on the dataset structure
train_data_dir = os.path.join(data_dir, 'train')
val_data_dir = os.path.join(data_dir, 'validation')
test_data_dir = os.path.join(data_dir, 'test')

# ... (Rest of your code) ...

# Assuming you have downloaded the Fashion144K dataset and extracted it to a folder named 'fashion144k'
data_dir = "C:\\Users\\aleyn\\OneDrive\\Masaüstü\\deep l.1\\Fashion144k_v1\\Fashion144k_v1"



# Define image width and height based on your dataset or desired image size
img_width = 224
img_height = 224


def create_baseline_model(img_width, img_height):
    # ... (Model definition using img_width and img_height) ...
    return model

# ... (Rest of your code) ...

# Assuming you defined image dimensions earlier
model = create_baseline_model(img_width, img_height)

# ... (Rest of your code) ...

# ... (Rest of your code) ...

train_generator = ImageDataGenerator(rescale=1./255)
train_generator = train_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# ... (Rest of your code) ...


# Define train, validation, and test data paths based on the dataset structure
train_data_dir = os.path.join(data_dir, 'train')
val_data_dir = os.path.join(data_dir, 'validation')
test_data_dir = os.path.join(data_dir, 'test')

# Load images using ImageDataGenerator
train_generator = ImageDataGenerator(rescale=1./255)
train_generator = train_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'  # Assuming 'fashion144k' has multiple fashion categories
)

val_generator = ImageDataGenerator(rescale=1./255)
val_generator = val_generator.flow_from_directory(
    val_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

test_generator = ImageDataGenerator(rescale=1./255)
test_generator = test_generator.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Define model architecture (baseline or advanced with attention if applicable)
# ... (Model definition code) ...

# Model compilation, training, and evaluation
# ... (Training and evaluation code) ...
"""