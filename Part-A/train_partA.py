# Part-A

### lmport  the relevant libraries and inbuilt functions
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np    
import matplotlib.pyplot as plt
from PIL import Image
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU, GELU, SELU, Mish
from torch.nn import LogSoftmax
from torch import flatten

### Selecting the device to run our code(GPU or CPU)
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Giving the arg parse command option

# Using argparse, I have define the arguments and options that my program accepts,
# and argparse will run the code, pass arguments from command line and 
# automatically generate help messages. I have given the defaults values for 
# all the arguments, so code can be run without passing any arguments.
# lastly returning the arguments to be used in the running of the code.

import argparse

parser = argparse.ArgumentParser(description="Stores all the hyperpamaters for the model.")
parser.add_argument("-wp", "--wandb_project",default="cs6910_assignment2_new" ,type=str,
                    help="Enter the Name of your Wandb Project")
parser.add_argument("-we", "--wandb_entity", default="am22s020",type=str,
                    help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("-ws", "--wandb_sweep", default="False", type=bool,
                    help="If you want to run wandb sweep then give True")
parser.add_argument("-e", "--epochs",default="1", type=int, help="Number of epochs to train neural network.")
parser.add_argument("-b", "--batch_size",default="4", type=int, help="Batch size used to train neural network.")
parser.add_argument("-a", "--activation",default="selu", type=str, choices=["selu", "gelu", "relu"])
parser.add_argument("-dl", "--dense_layer",default="100", type=int, 
                    choices=[50,100,150,200], help="Choose number of neuron in dense layer")
parser.add_argument("-da", "--data_augmentation", default="True", type=bool)
parser.add_argument("-bn", "--batch_normalisation", default="True", type=bool)

args = parser.parse_args()

wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
wandb_sweep = args.wandb_sweep
num_epochs = args.epochs
batch_size = args.batch_size
act_fu = args.activation
size_denseLayer = args.dense_layer
data_augmentation = args.data_augmentation
batch_normalisation = args.batch_normalisation

print("wandb_project :", wandb_project , "wandb_entity: ", wandb_entity,"wandb_sweep: ",wandb_sweep,
      "epochs: ",num_epochs,"batch_size: ",batch_size, "dense_layer: ", size_denseLayer,
      "activation_function: ", act_fu,"data augmentation: ", data_augmentation, 
      "batch Normalization: ", batch_normalisation)


###  Uploading and transforming the dataset to train our CNN model 

# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use, find the size, mean and std for each channel of our dataset
def data_pre_processing(batch_size, data_augmentation):
    
    """
    This function will upload the downloaded datasets(train and test).
    Apply transformation on the dataset like resize to make each image of same size
    as we know inaturalist datasets are having different sizes.
    Split the train dataset into train(80%) and validation(20%).
    Use transforms.compose method to reformat images for modeling,and save to variable
    all_transforms for later use, find the size, mean and std for each channel of our dataset.
    Again uploaded the train dataset and taken only 20% of it to make
    augmented dataset.
    I have applied horizontalFlip, Randomrotation to augment dataset with resize
    and the normalization.
    Then i concated the train and augmented datasets if data_augmentation=True.
    I have created dataloader for train, validation and test datasets. dataloader 
    takes data in batches which saves our memory.
    Then I have returned the dataloaders and datasets.
    
    """
    
    # Transformation on the dataset. Resize=> numpy to tensor => Normalizing the pixel values
    all_transforms = transforms.Compose([transforms.Resize((256,256)),
                                         transforms.ToTensor(),        #0-255 to 0-1 & numpy to tensor
                                         transforms.Normalize(mean=[0.4713, 0.4600, 0.3897],  #0-1 to [-1,1]
                                                              std=[0.2373, 0.2266, 0.2374])                                        
                                         ])

    # path for training and testing dataset directory
    train_path = r"C:\Users\HICLIPS-ASK\nature_12K\inaturalist_12K\train"
    test_path = r"C:\Users\HICLIPS-ASK\nature_12K\inaturalist_12K\val"
    
    # uploading the train data with above transformation applied
    train_dataset = torchvision.datasets.ImageFolder(root = train_path, transform = all_transforms)
    
    # converting train dataset into train and validation for hyperparameter tuning
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # splitting the train data inti train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # data augmentation
    if data_augmentation == True:
        # These transformation will be applied on the augmented dataset 
        augment_transforms = transforms.Compose([transforms.Resize((256,256)),                                         
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomRotation((-60,60)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.4713, 0.4600, 0.3897], 
                                                              std=[0.2373, 0.2266, 0.2374])
                                                 
                                         ])
        
        
        # uploading train dataset to take a portion as augment dataset, then concate with train dataset
        aug_dataset = torchvision.datasets.ImageFolder(root = train_path, transform = augment_transforms)
        discrad_size = int(0.8 * len(aug_dataset))
        aug_size = len(aug_dataset) - discrad_size
        
        _ , transformed_dataset = torch.utils.data.random_split(aug_dataset, [discrad_size, aug_size])
        train_dataset = torch.utils.data.ConcatDataset([transformed_dataset, train_dataset])
    
    # uploading the test dataset
    test_dataset = torchvision.datasets.ImageFolder(root = test_path, transform = all_transforms)

    # Instantiate loader objects to facilitate processing
    # shuffle= True, will ensure data of each class present in each batch
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True)


    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                               batch_size = batch_size,
                                               shuffle = True)
    
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                               batch_size = batch_size,
                                               shuffle = True)
    
    return train_loader, test_loader, val_loader, train_dataset,test_dataset


### Creating CNN Model from scratch using pytorch
class ConvNeuNet(Module):
    
    """
    In __init__, I have initialized the layers to be used in the model.
    My model has five consecutive convolutional layers, each layer has set of
    2D convolution => Batch Nromalisation => non-linear activation function => Dropout => max pool layers.
    The code is flexible such that the number of filters, size of filters, and activation function of the 
    convolution layers and dense layers can be changed. We can change the number of neurons in the dense layer.
    I have created a class named ConvNeuNet(nn.Modeule), in this nitialised the init function with arguments
    having flexible inputs. In init initialised all the convolution layers, dense layer and output layer.
    Within the class, I have created forward() function in which all the layers are arranged in sequence.
    The output of forward is a tensor having probability of ten classes.
    
    """
    
    def __init__(self, size_kernel, num_stride, act_fu, size_denseLayer,
                 data_augmentation, batch_normalisation,padding, dropout_rate,
                 num_filters,classes=10,input_channels=3):
        
        # call the parent constructor
        super(ConvNeuNet, self).__init__()
        
        self.batch_norm = batch_normalisation
        self.data_aug = data_augmentation
        width = 256
        height = 256
        
        #(batch_size = 64, input_channels=3, width=150, height=150)
         # initialize second set of CONV => update dim => Batch Nrom => RELU => Dropout => POOL layers
        self.conv1 = Conv2d(in_channels=input_channels, out_channels=num_filters[0],
                    kernel_size=size_kernel[0], stride=num_stride, padding=padding)
        width = int((width- size_kernel[0][0] + 2*padding)/num_stride) + 1
        height = int((height- size_kernel[0][0] + 2*padding)/num_stride) + 1
        self.bn1 = nn.BatchNorm2d(num_features=num_filters[0])
        self.af1 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # updating width and height of the next layer after maxpool
        width = int((width- 2 + 2*0)/2) + 1
        height = int((height- 2 + 2*0)/2) + 1
        # initialize second set of CONV => update dim => Batch Nrom => RELU => Dropout => POOL layers
        
        self.conv2 = Conv2d(in_channels=num_filters[0], out_channels=num_filters[1],
                     kernel_size=size_kernel[1], stride=num_stride, padding=padding)
        width = int((width- size_kernel[1][0] + 2*padding)/num_stride) + 1
        height = int((height- size_kernel[1][0] + 2*padding)/num_stride) + 1
        self.bn2 = nn.BatchNorm2d(num_features=num_filters[1])
        self.af2 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # updating width and height of the next layer after maxpool
        width = int((width- 2 + 2*0)/2) + 1
        height = int((height- 2 + 2*0)/2) + 1
        
         # initialize second set of CONV => update dim => Batch Nrom => RELU => Dropout => POOL layers
        self.conv3 = Conv2d(in_channels=num_filters[1], out_channels=num_filters[2],
                     kernel_size=size_kernel[2], stride=num_stride, padding=padding)
        width = int((width- size_kernel[2][0] + 2*padding)/num_stride) + 1
        height = int((height- size_kernel[2][0] + 2*padding)/num_stride) + 1
        self.bn3 = nn.BatchNorm2d(num_features=num_filters[2])
        self.af3 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.maxpool3 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
         # updating width and height of the next layer after maxpool
        width = int((width- 2 + 2*0)/2) + 1
        height = int((height- 2 + 2*0)/2) + 1
        
         # initialize second set of CONV => update dim => Batch Nrom => RELU => Dropout => POOL layers
        self.conv4 = Conv2d(in_channels=num_filters[2], out_channels=num_filters[3],
                     kernel_size=size_kernel[3], stride=num_stride, padding=padding)
        width = int((width- size_kernel[3][0] + 2*padding)/num_stride) + 1
        height = int((height- size_kernel[3][0] + 2*padding)/num_stride) + 1
        self.bn4 = nn.BatchNorm2d(num_features=num_filters[3])
        self.af4 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.dropout4 = nn.Dropout(p=dropout_rate)
        self.maxpool4 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # updating width and height of the next layer after maxpool
        width = int((width- 2 + 2*0)/2) + 1
        height = int((height- 2 + 2*0)/2) + 1
        
         # initialize second set of CONV => update dim => Batch Nrom => RELU => Dropout => POOL layers
        self.conv5 = Conv2d(in_channels=num_filters[3], out_channels=num_filters[4],
                     kernel_size=size_kernel[4], stride=num_stride, padding=padding)
        width = int((width- size_kernel[4][0] + 2*padding)/num_stride) + 1
        height = int((height- size_kernel[4][0] + 2*padding)/num_stride) + 1
        self.bn5 = nn.BatchNorm2d(num_features=num_filters[4])
        self.af5 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.dropout5 = nn.Dropout(p=dropout_rate)
        self.maxpool5 = MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # updating width and height of the next layer after maxpool
        width = int((width- 2 + 2*0)/2) + 1
        height = int((height- 2 + 2*0)/2) + 1
        #initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=int(num_filters[4]*width*height), out_features=size_denseLayer)
        self.af6 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.dropout6 = nn.Dropout(p=dropout_rate)
        
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=size_denseLayer, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
        
    def forward(self, x):
        
        # pass the input through our first set of CONV => Batch_norm => RELU =>
        # POOL layers
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.af1(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)
       
        # pass the output from the previous layer through the second
        # set of CONV => Batch_norm => RELU => layers
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.af2(x)
        x = self.dropout2(x)
        x = self.maxpool2(x)
        # pass the output from the previous layer through the third
        # set of CONV => Batch_norm => RELU => POOL layers
        x = self.conv3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.af3(x)
        x = self.dropout3(x)
        x = self.maxpool3(x)
        # pass the output from the previous layer through the fourth
        # set of CONV => Batch_norm => RELU => POOL layers
        x = self.conv4(x)
        if self.batch_norm:
            x = self.bn4(x)
        x = self.af4(x)
        x = self.dropout4(x)
        x = self.maxpool4(x)
        # pass the output from the previous layer through the fifth
        # set of CONV => Batch_norm => RELU => POOL layers
        x = self.conv5(x)
        if self.batch_norm:
            x = self.bn5(x)
        x = self.af5(x)
        x = self.dropout5(x)
        x = self.maxpool5(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.af6(x)
        x = self.dropout6(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output
    
    ### Define the function to find the accuracy of the model

    # Define evaluation function
def evaluate(model, dataloader):
    """
    This function will calculate the accuracy on the given dataloader dataset.
    It takes model and dataloader as arguments and return the accuracy.
    First model is set into the .eval() mode to deactivate backpropagation.
    Then images in the dataloader are send through forward function and 
    compared with the labels to match max value in output with the label class.
    Correct term initialized to collect the number of correct prediction and total
    is to count the total number of images.
    finally we get the accuracy by, accuracy = 100 * correct / total formula.
    
    """
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

### Creating function to train our model
# install and import the wandb 
if wandb_sweep == True:
    #!pip install wandb
    import wandb

# don't comment it
num_filters, size_kernel,num_stride,padding,dropout_rate = [12,12,12,12,12],[(3,3),(3,3),(3,3),(3,3),(3,3)],1,1,0.3

def train_CNN(num_filters, size_kernel,num_stride,padding,dropout_rate,
             wandb_sweep, wandb_project, num_epochs, batch_size,data_augmentation,
              wandb_entity, act_fu,size_denseLayer,batch_normalisation):
    
    """
    Created train_CNN() function to train our model.
    In this, first model is defined and it is exported to the device (either GPU or CPU).
    Optimizer and loss function is imported using torch library. 
    we have choosen Adam as optimizer with learning rate= 1e-4, and weight
    decay = 1e-4. Then cross entropy loss is choosen as the loss function.
    I have also include the commands needed to integrate the wandb sweep. 
    If wandb_sweep == True, then sweep will start otherwise training will be shown only in our environment.
    I have login to wandb account. I have already imported the wandb,
    now I am giving the default values of our variable for sweep. 
    After that I have defined the wandb run name which will be assign to each run.
    Values like epoch, train loss, train accuracy and validation accuracy are login to wandb.
    Saving the wandb run and finishing the run.
    
    """
    
    
    if wandb_sweep == True:
        #default values for wandb run
        config_defaults = {
            'num_filters': [12,12,12,12,12],
            'act_fu': 'relu',
            'size_kernel': [(3,3),(3,3),(3,3),(3,3),(3,3)],
            'data_augmentation': True,
            'batch_normalisation': True,
            'dropout_rate': 0.2,
            'size_denseLayer':200
             }
    

        #initialize wandb
        wandb.init(project = wandb_project ,config=config_defaults)

        # config is a data structure that holds hyperparameters and inputs
        config = wandb.config

        # Local variables, values obtained from wandb config
        num_filters = config.num_filters
        act_fu = config.act_fu
        size_kernel = config.size_kernel
        data_augmentation = config.data_augmentation
        batch_normalisation = config.batch_normalisation
        dropout_rate = config.dropout_rate
        size_denseLayer = config.size_denseLayer

        # Defining the run name in wandb sweep
        wandb.run.name  = "FSize_{}_af_{}_NF_{}_DA_{}_BN_{}_Drp_{}_DLayer_{}_".format(size_kernel,
                                                                              act_fu,
                                                                              num_filters,
                                                                              data_augmentation,
                                                                              batch_normalisation,
                                                                              dropout_rate,
                                                                              size_denseLayer)


        print(wandb.run.name )
    

    #loading the dataloaders and dataset
    train_loader, test_loader, val_loader, train_dataset, test_dataset = data_pre_processing(batch_size,
                                                                                        data_augmentation=True)
    # Loading the cnn model    
    model = ConvNeuNet(size_kernel, num_stride, act_fu, size_denseLayer,
                 data_augmentation, batch_normalisation,padding, dropout_rate,
                 num_filters,classes=10, input_channels=3).to(device)
    
    #Optimizer and loss function
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.0001)
    loss_function=nn.CrossEntropyLoss()
    

       
    # Training on training dataset
    # setting model to train mode
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # loading the images and labels to the device
            inputs, labels = data[0].to(device), data[1].to(device)
            # for each train loader initializing the gradient to zero
            optimizer.zero_grad()
            # getting output from the model
            outputs = model(inputs)
            # calculating the loss for each batch
            loss = loss_function(outputs, labels)
            # doing backprop
            loss.backward()
            # Updating the parameters
            optimizer.step()
            # adding the train loss of each batch
            running_loss += loss.item()
            # storing the loss of last 100 batches
            if i % 100 == 99:
                train_loss=running_loss/100   
                running_loss = 0.0

        # Evaluate training set accuracy
        train_accuracy = evaluate(model, train_loader)

        # Evaluate test set accuracy
        val_accuracy = evaluate(model, val_loader)


        print("Epoch: "+str(epoch+1)+ ' Train Loss:'+ str(train_loss) +' Train Accuracy:'+
              str(train_accuracy) + ' Validation Accuracy: '+ str(val_accuracy)) 


        if wandb_sweep == True:
        
            wandb.log({"validation accuracy": val_accuracy, "train accuracy": train_accuracy, 
                        "train loss": train_loss, 'epoch': epoch+1})
            
    if wandb_sweep == True:
        wandb.run.name 
        wandb.run.save()
        wandb.run.finish()
        
    if wandb_sweep == False:
        return model

       
if wandb_sweep == False:
    model = train_CNN(num_filters, size_kernel,num_stride,padding,dropout_rate,
             wandb_sweep, wandb_project, num_epochs, batch_size,data_augmentation,
              wandb_entity, act_fu,size_denseLayer,batch_normalisation) 
    
# Running the wandb sweep
def sweep():
    
    """
    This function is used to exploit the wandb hyperparameter sweep 
    function to get the best hyperparameters.

    It takes in no inputs and gives no outputs.
  
    Instead it logs everything into the wandb workspace'''

    """
    sweep_config = {"name": wandb_project, "method": "bayes"}   
    sweep_config["metric"] = {"name": "val_accuracy", "goal": "maximize"}

    parameters_dict = {
                  "num_filters": {"values": [[12,12,12,12,12],[4,8,16,32,64],[64,32,16,8,4]]},
                  "act_fu": {"values": ["relu","selu","gelu","mish"]},
                  "size_kernel": {"values": [[(3,3),(3,3),(3,3),(3,3),(3,3)], [(3,3),(5,5),(5,5),(7,7),(7,7)],
                                             [(7,7),(7,7),(5,5),(5,5),(3,3)]]}, 
                    "data_augmentation": {"values": [True, False]} ,
                    "batch_normalisation": {"values": [True, False]} ,
                    "dropout_rate": {"values": [0, 0.2, 0.3]},
                    "size_denseLayer": {"values": [50, 100, 150, 200]}
                    }
    sweep_config["parameters"] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, entity=wandb_entity, project=wandb_project)
    wandb.agent(sweep_id, train_CNN(num_filters, size_kernel,num_stride,padding,dropout_rate,
             wandb_sweep, wandb_project, num_epochs, batch_size,data_augmentation,
              wandb_entity, act_fu,size_denseLayer,batch_normalisation), count=100)
    
if wandb_sweep == True:
    sweep()

### testing the model on best sweep parameters
def test_model():
    
    """
    This function test the model on the best parameters find out after 
    hyper parameter tuning using wandb.
    set the arguments of the best validation accuracy run in the ConvNeuNet().
    This function train the model on the best parameters and give the test accuracy 
    on the test dataset.
    
    """
    
    print("Testing the model on the best parameters===>>>")
    
    model = ConvNeuNet(size_kernel=[(3,3),(3,3),(3,3),(3,3),(3,3)], num_stride=1, act_fu='selu', size_denseLayer=200,
                     data_augmentation=True, batch_normalisation=True, input_channels=3,
                     classes=10, padding=1, dropout_rate=0.3, num_filters=[12,12,12,12,12]).to(device)

    #Optimizer and loss function
    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.0001)
    loss_function=nn.CrossEntropyLoss()

    num_epochs=5

    train_loader, test_loader, val_loader, train_dataset, test_dataset = data_pre_processing(batch_size=16,
                                                                                        data_augmentation=True)
    
    

     # Training on training dataset
    best_accuracy = 0.0
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            if epoch == 9:
                print(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            if epoch == 9:
                print(outputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                train_loss=running_loss/100   
                running_loss = 0.0

        # Evaluate training set accuracy
        train_accuracy = evaluate(model, train_loader)

        # Evaluate test set accuracy
        test_accuracy = evaluate(model, test_loader)
        
        #Save the best model
        if test_accuracy>best_accuracy:
            torch.save(model.state_dict(), 'best_partA_checkpoint.model')
            best_accuracy=test_accuracy


        print("Epoch: "+str(epoch+1)+ ' Train Loss:'+ str(train_loss) +' Train Accuracy:'+
              str(train_accuracy) + ' Test Accuracy: '+ str(test_accuracy))
        
    return model
        
        
### testing the model on the test dataset
model = test_model()