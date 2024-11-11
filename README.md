### **Project Overview**
The goal of this project is to build a model capable of classifying hand gestures representing ASL alphabet letters. The Sign Language MNIST dataset is used, which contains images labeled with each letter of the alphabet, enabling the model to recognize these signs from grayscale images of hand gestures.

**Requirements**
Python 3.7+
TensorFlow 2.x
Keras
NumPy
Matplotlib

**Install required libraries via pip:**
pip install tensorflow numpy matplotlib

**Dataset**
The dataset used in this project is the Sign Language MNIST dataset. This dataset contains two CSV files:
sign_mnist_train.csv: Training data (images and labels)
sign_mnist_test.csv: Validation data (images and labels)
Each CSV file contains 28x28 pixel images flattened into a row with labels for each image.

**Algorithm**
1. Data Parsing and Loading
The parse_data_from_input() function reads the CSV files, separates labels and image data, reshapes images to 28x28, and converts them into NumPy arrays for easier processing.
2. Data Visualization
plot_categories() function displays a sample of 10 images with their corresponding labels to visualize different hand signs in grayscale.
3. Data Augmentation and Generator Setup
train_val_generators() function utilizes Keras’s ImageDataGenerator to create data generators for both training and validation datasets. The training images are augmented with random transformations such as rotation, shifts, shearing, and zooming to improve the model's generalization.
4. Model Architecture
create_model() defines a Sequential CNN model:
Two convolutional layers with ReLU activation and max-pooling
A flattening layer followed by a fully connected layer
Output layer with softmax activation for classifying 26 categories (one for each letter of the alphabet)
The model is compiled with the Adam optimizer and sparse categorical cross-entropy as the loss function.
5. Model Training
The model is trained for 15 epochs with the training and validation generators, and accuracy and loss metrics are tracked.
6. Results Visualization
Training and validation accuracy and loss are plotted over the epochs to analyze the model's performance and identify any signs of overfitting.
