import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.models as M

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()
        # Definig linear layers for Model 1
        self.fc1_m1 = nn.Linear(224 * 224 * 3, 1000) # Input size: 224 * 224 * 3(flattened colored image), Output: 1000 neurons
        self.fc2_m1 = nn.Linear(1000, 500)           # Input: 1000 neurons, Output: 500 neurons
        self.fc3_m1 = nn.Linear(500, 100)            # Input: 500 neurons, Output: 100 neurons
        self.fc4_m1 = nn.Linear(100, 4)              # Input: 100 neurons, Output: 4 classes 

        
        # Defining required convolutional and linear layers for Model 2
        # Convolutional layers
        self.conv1_m2 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, stride=1, padding=1)     # Input: (3, 224, 224)   | Output: (512, 224, 224)
        self.conv2_m2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)   # Input: (512, 224, 224) | Output: (256, 224, 224)
        self.conv3_m2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)   # Input: (256, 224, 224) | Output: (128, 224, 224)
        self.conv4_m2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)    # Input: (128, 224, 224) | Output: (64, 224, 224)
        self.conv5_m2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)     # Input: (64, 224, 224)  | Output: (32, 224, 224)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples by a factor of 2
        
        # Fully connected layers
        self.fc1_m2 = nn.Linear(32 * 7 * 7, 1000)   # Input: 32 * 7 * 7, Output: 1000 neurons
        self.fc2_m2 = nn.Linear(1000, 100)          # Input: 1000 neurons, Output: 100 neurons
        self.fc3_m2 = nn.Linear(100, 4)             # Input: 100 neurons, Output: 4 classes

        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        else: 
            print("Invalid mode ", mode, "selected. Select 1 or 2")
            exit(0)
        
    #############################
    ### fully connected layer ###
    ############################# 
    def model_1(self, X):
        X = torch.flatten(X, start_dim=1) # Flatten the input tensor
        X = F.relu(self.fc1_m1(X))        # Apply first fully connected layer                          
        X = F.relu(self.fc2_m1(X))        # Apply second fully connected layer     
        X = F.relu(self.fc3_m1(X))        # Apply third fully connected layer
        X = F.relu(self.fc4_m1(X))        # Apply fourth fully connected layer
        return X

    #################################################################
    ### five convolutional layers + three fully connnected layers ###
    ### kernel size = 3, stride = 1, padding = 1, max pooling = 2 ###
    #################################################################
    def model_2(self, X):
        X = self.pool(F.relu(self.conv1_m2(X)))   # Apply first convolutional layer
        X = self.pool(F.relu(self.conv2_m2(X)))   # Apply second convolutional layer
        X = self.pool(F.relu(self.conv3_m2(X)))   # Apply third convolutional layer
        X = self.pool(F.relu(self.conv4_m2(X)))   # Apply fourth convolutional layer
        X = self.pool(F.relu(self.conv5_m2(X)))   # Apply fifth convolutional layer
        X = torch.flatten(X, start_dim=1)         # Flatten the tensor
        X = F.relu(self.fc1_m2(X))                # Apply first fully connected layer                          
        X = F.relu(self.fc2_m2(X))                # Apply second fully connected layer     
        X = F.relu(self.fc3_m2(X))                # Apply third fully connected layer
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
