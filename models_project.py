import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.models as M

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        # Definig required layers for Model 1
        self.fc1_m1 = nn.Linear(224 * 224 * 3, 1000) # Input size: 28x28 (flattened image), Output: 100 neurons
        self.fc2_m1 = nn.Linear(1000, 4)      # Input: 100 neurons, Output: 10 classes (digits 0-9)

        # Defining required layers for Model 2 & 3
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)  # Convolution layer: 1 input channel, 40 output channels, 5x5 filter
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)       # Max pooling layer: 2x2 filter with stride 2
        self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1) # Convolution layer: 40 input channels, 40 output channels, 5x5 filter 
        self.fc1_m2_3 = nn.Linear(40 * 4 * 4, 100)              # Flattened feature map to 100 neurons
        self.fc2_m2_3 = nn.Linear(100, 10)                      # Output: 10 classes (digits 0-9)

        # Defining required layers for Model 4
        self.fc1_m4 = nn.Linear(40 * 4 * 4, 100)  # First fully connected layer with 100 neurons
        self.fc2_m4 = nn.Linear(100, 100)         # Second fully connected layer with 100 neurons
        self.fc3_m4 = nn.Linear(100, 10)

        # Defining required layers for Model 5
        self.fc1_m5 = nn.Linear(40 * 4 * 4, 1000) # First fully connected layer with 1000 neurons
        self.fc2_m5 = nn.Linear(1000, 1000)       # Second fully connected layer with 1000 neurons
        self.fc3_m5 = nn.Linear(1000, 10)         # Output layer: 10 classes (digits 0-9)
        self.dropout = nn.Dropout(0.5)            # Dropout layer with 50% probability


        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        #################################
        ### One fully connected layer ###
        #################################
        X = torch.flatten(X, start_dim=1) # Flatten the input tensor
        X = self.fc1_m1(X)                # Apply first fully connected layer                          
        X = F.sigmoid(X)                  # Apply sigmoid activation function
        X = self.fc2_m1(X)                # Apply second fully connected layer
        return X

    # Use two convolutional layers.
    def model_2(self, X):
        #############################################################
        ### Two convolutional layers + one fully connnected layer ###
        #############################################################
        X = self.conv1(X)                 # Apply first convolutional layer
        X = F.sigmoid(X)                  # Apply sigmoid activation function
        X = self.pool(X)                  # Apply max pooling
        X = self.conv2(X)                 # Apply second convolutional layer
        X = F.sigmoid(X)                  # Apply sigmoid activation function
        X = self.pool(X)                  # Apply max pooling
        X = torch.flatten(X, start_dim=1) # Flatten the tensor
        X = self.fc1_m2_3(X)              # Apply first fully connected layer
        X = F.sigmoid(X)                  # Apply sigmoid activation function
        X = self.fc2_m2_3(X)              # Apply second fully connected layer
        return X

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        #######################################################################
        ### Two convolutional layers + one fully connected layer, with ReLU ###
        #######################################################################
        X = self.conv1(X)                  # Apply first convolutional layer
        X = F.relu(X)                      # Apply ReLU activation function
        X = self.pool(X)                   # Apply max pooling
        X = self.conv2(X)                  # Apply second convolutional layer
        X = F.relu(X)                      # Apply ReLU activation function
        X = self.pool(X)                   # Apply max pooling
        X = torch.flatten(X, start_dim=1)  # Flatten the tensor
        X = F.relu(X)                      # Apply ReLU activation function
        X = self.fc2_m2_3(X)               # Apply second fully connected layer
        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        ########################################################################
        ### Two convolutional layers + two fully connected layers, with ReLU ###
        ########################################################################
        X = self.conv1(X)                 # Apply first convolutional layer
        X = F.relu(X)                     # Apply ReLU activation function
        X = self.pool(X)                  # Apply max pooling
        X = self.conv2(X)                 # Apply second convolutional layer
        X = F.relu(X)                     # Apply ReLU activation function
        X = self.pool(X)                  # Apply max pooling
        X = torch.flatten(X, start_dim=1) # Flatten the tensor
        X = self.fc1_m4(X)                # Apply first fully connected layer
        X = F.relu(X)                     # Apply ReLU activation function
        X = self.fc2_m4(X)                # Apply second fully connected layer
        X = F.relu(X)                     # Apply ReLU activation function
        X = self.fc3_m4(X)                # Apply third fully connected layer
        return X

    # Use Dropout now.
    def model_5(self, X):
        ##################################################################################
        ### Two convolutional layers + two fully connected layers, with ReLU + Dropout ###
        ##################################################################################
        X = self.conv1(X)                 # Apply first convolutional layer
        X = F.relu(X)                     # Apply ReLU activation function
        X = self.pool(X)                  # Apply max pooling
        X = self.conv2(X)                 # Apply second convolutional layer
        X = F.relu(X)                     # Apply ReLU activation function
        X = self.pool(X)                  # Apply max pooling
        X = torch.flatten(X, start_dim=1) # Flatten the tensor
        X = self.fc1_m5(X)                # Apply first fully connected layer
        X = F.relu(X)                     # Apply ReLU activation function
        X = self.dropout(X)               # Apply dropout
        X = self.fc2_m5(X)                # Apply second fully connected layer
        X = F.relu(X)                     # Apply ReLU activation function
        X = self.dropout(X)               # Apply dropout
        X = self.fc3_m5(X)                # Apply third fully connected layer
        return X

    
class PreTrainedModels(nn.Module):
    def __init__(self, model_name, pretrained):
        super(PreTrainedModels, self).__init__()
        self.pretrained = pretrained
        self.model_name = model_name
        if model_name in ['AlexNet', 'alexnet']:
            self.model = self.AlexNet()
        elif model_name in ['ResNet', 'resnet']:
            self.model = self.ResNet()
        elif model_name in ['VGG', 'vgg']:
            self.model = self.VGG()
        elif model_name in ['SqueezeNet', 'squeezenet']:
            self.model = self.SqueezeNet()
        elif model_name in ['DenseNet', 'densenet']:
            self.model = self.DenseNet()
        elif model_name in ['GoogleNet', 'googlenet']:
            self.model = self.GoogleNet()
        elif model_name in ['ViT', 'vit']:
            self.model = self.ViT()
        else:
            print("Invalid model name ", model_name, "selected. Select from ['AlexNet', 'ResNet', 'VGG', 'SqueezeNet', 'DenseNet', 'GoogleNet', 'ViT']")
            exit(0)

    def forward(self, x):
        return self.model(x)

    # AlexNet model with configurable weights and classes
    def AlexNet(self, pretrained=True):
        # Load the AlexNet model with or without pretrained weights
        weights = M.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.alexnet(weights=weights)
        # Modify the classifier to match the number of classes
        model.classifier[6] = nn.Linear(4096, 4)
        return model

    # ResNet18 model with configurable weights and classes
    def ResNet(self, pretrained=True):
        # Load the ResNet18 model with or without pretrained weights
        weights = M.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        # Modify the classifier to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, 4)
        return model
    
    # VGG16 model with configurable weights and classes
    def VGG(self, pretrained=True):
        # Load the VGG16 model with or without pretrained weights
        weights = M.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        # Modify the classifier to match the number of classes
        model.classifier[6] = nn.Linear(4096, 4)
        return model
    
    # SqueezeNet model with configurable weights and classes
    def SqueezeNet(self, pretrained=True):
        # Load the SqueezeNet model with or without pretrained weights
        weights = M.SqueezeNet1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.squeezenet1_0(weights=weights)
        # Modify the classifier to match the number of classes
        model.classifier[1] = nn.Conv2d(512, 4, kernel_size=(1,1), stride=(1,1))
        return model

    # DenseNet model with configurable weights and classes
    def DenseNet(self, pretrained=True):
        # Load the DenseNet model with or without pretrained weights
        weights = M.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        # Modify the classifier to match the number of classes
        model.classifier = nn.Linear(model.classifier.in_features, 4)
        return model
    
    # GoogleNet model with configurable weights and classes
    def GoogleNet(self, pretrained=True):
        # Load the GoogleNet model with or without pretrained weights
        weights = M.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.googlenet(weights=weights)
        # Modify the classifier to match the number of classes
        model.fc = nn.Linear(model.fc.in_features, 4)
        return model
    
    # Vision Transformer model with configurable weights and classes
    def ViT(self, pretrained=True):
        # Load the Vision Transformer model with or without pretrained weights
        weights = M.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        # Modify the head to match the number of classes
        model.heads.head = nn.Linear(model.heads.head.in_features, 4)
        return model
