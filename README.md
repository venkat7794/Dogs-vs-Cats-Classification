# Dog and Cat Classification using CNN

This repository contains a TensorFlow and Keras-based convolutional neural network (CNN) model that classifies images of dogs and cats. The model is trained on a dataset of images and evaluates its performance using metrics such as training accuracy, training loss, validation accuracy, and validation loss.

### Requirements

- TensorFlow
- Keras
- Python
- ImageDataGenerator

### Usage

1. **Install Dependencies**: Install TensorFlow and Keras using pip:
   ```bash
   pip install tensorflow keras
   ```

2. **Set Paths**: Update the `train_dir` and `validation_dir` variables to point to the directories containing the training and validation images, respectively.

3. **Run the Code**: Run the Python script to train the model and evaluate its performance:
   ```python
   python dog_cat_classification.py
   ```

### Output

The script will print the following metrics:
- **Training Accuracy**: The accuracy of the model on the training set.
- **Training Loss**: The loss of the model on the training set.
- **Validation Accuracy**: The accuracy of the model on the validation set.
- **Validation Loss**: The loss of the model on the validation set.

### Model Architecture

The model consists of several convolutional and pooling layers, followed by a flatten layer and two dense layers. The output layer uses the softmax activation function to output probabilities for both dog and cat classes.

### Data Preparation

The model uses the `ImageDataGenerator` to load and preprocess the images. The images are resized to 150x150 pixels, and the data is augmented using random shear, zoom, and horizontal flip.

### Training

The model is trained using the Adam optimizer and categorical cross-entropy loss. The training process is run for 10 epochs, and the model is evaluated on the validation set after each epoch.

### Evaluation

The model's performance is evaluated using the accuracy and loss metrics on both the training and validation sets. The final accuracy and loss values are printed to the console

### contributions
Contributions to this project are welcome. Please submit pull requests or issues to improve the model's performance or add new features.

