# Foundational Models vs Pre-trained Models for Emotion Detection
In this project, we will build foundational models for emotion detection in humans and dogs and compare their performance with Pre-trained models with Transfer Learning.

## Requirements

Make sure you have the following libraries installed:

- Python 3.6+
- torch
- torchvision
- numpy
- tqdm
- tensorboard


You can install the required libraries using pip:

pip install torch torchvision numpy tqdm tensorboard


## Running the Code

# For Human Emotion Detection

Using Transfer learning:
python train_evaluate_human_TL.py --model_name <model_name> --pretrained <True/False> --learning_rate <learning_rate> --num_epochs <num_epochs> --batch_size <batch_size> --log_dir <log_dir>

Using Foundational Models:
python train_evaluate_human_CNN.py --mode <mode> --learning_rate <learning_rate> --num_epochs <num_epochs> --batch_size <batch_size> --log_dir <log_dir>

# For Dog Emotion Detection

Using Transfer Learning:
python train_evaluate_dog_TL.py --model_name <model_name> --pretrained <True/False> --learning_rate <learning_rate> --num_epochs <num_epochs> --batch_size <batch_size> --log_dir <log_dir>

Using Foundational Models:
python train_evaluate_dog_CNN.py --mode <mode> --learning_rate <learning_rate> --num_epochs <num_epochs> --batch_size <batch_size> --log_dir <log_dir>


# References for input

Model Names:
- Alex Net
- ResNet
- VGG
- SqueezeNet
- DenseNet
- GoogleNet
- ViT

Modes to Choose:
1: Fully Conencted layers
2: Convolutional layers followed by fully connected layers

Logging
Training and evaluation metrics will be logged to TensorBoard
tensorboard --logdir=<log_dir>

# Dataset
Make sure the dataset is placed in the following directories
Human dataset: ~/user/CV-Project/Data/Human/
Dog dataset: ~/user/CV-Project/Data/Dog/

The dataset should be organized in subdirectories for each class.


# Output
The output of the training and evaluation will be printed to the console and logged to TensorBoard. The best accuracy achieved during the training will also be printed at the end.



