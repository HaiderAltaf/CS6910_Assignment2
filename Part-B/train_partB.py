## Importing the Libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU, GELU, SELU, Mish
from torch.nn import LogSoftmax
from torch import flatten

## Selecting the device to run our code(GPU or CPU)
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Giving the arg parse command option

# Using argparse, I have define the arguments and options that my program accepts,
# and argparse will run the code, pass arguments from command line and 
# automatically generate help messages. I have given the defaults values for 
# all the arguments, so code can be run without passing any arguments.



import argparse

# initializing the arguments to be passed
parser = argparse.ArgumentParser(description="Stores all the hyperpamaters for the model.")
parser.add_argument("-wp", "--wandb_project",default="cs6910_assignment2" ,type=str,
                    help="Enter the Name of your Wandb Project")
parser.add_argument("-we", "--wandb_entity", default="am22s020",type=str,
                    help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
parser.add_argument("-ws", "--wandb_sweep", default="False", type=bool,
                    help="If you want to run wandb sweep then give True")
parser.add_argument("-e", "--epochs",default="1", type=int, help="Number of epochs to train neural network.")
parser.add_argument("-b", "--batch_size",default="16", type=int, help="Batch size used to train neural network.")
parser.add_argument("-da", "--data_augmentation", default="True", type=bool, choices=[True, False])
parser.add_argument("-opt", "--optimizer", default="adam", type=str, choices=["adam", "sgd"])
parser.add_argument("-opt", "--learning_rate", default="0.0001", type=int, choices=[0.001, 0.0001])

args = parser.parse_args()

wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
wandb_sweep = args.wandb_sweep
num_epochs = args.epochs
batch_size = args.batch_size
data_augmentation = args.data_augmentation
optimizer = args.optimizer
learning_rate = args.learning_rate

print("wandb_project: ",wandb_project,"wandb_entity: ",wandb_entity,"wandb_sweep: ",wandb_sweep,
      "num_epochs :", num_epochs , "batch_size: ", batch_size, "data_augmentation",
      data_augmentation,  "optimizer: ", optimizer, "learning rate: ",learning_rate) 

## Uploading and transforming the dataset to train our CNN model 
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
    
    all_transforms = transforms.Compose([transforms.Resize((256,256)),
                                         transforms.ToTensor(),        #0-255 to 0-1 & numpy to tensor
                                         transforms.Normalize(mean=[0.4713, 0.4600, 0.3897],  #0-1 to [-1,1]
                                                              std=[0.2373, 0.2266, 0.2374])                                        
                                         ])

    # path for training and testing dataset directory
    train_path = r"C:\Users\HICLIPS-ASK\nature_12K\inaturalist_12K\train"
    test_path = r"C:\Users\HICLIPS-ASK\nature_12K\inaturalist_12K\val"

    train_dataset = torchvision.datasets.ImageFolder(root = train_path, transform = all_transforms)
    
    # converting train dataset into train and validation for hyperparameter tuning
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # data augmentation
    if data_augmentation == True:
        augment_transforms = transforms.Compose([transforms.Resize((256,256)),                                  
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomRotation((-60,60)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.4713, 0.4600, 0.3897], 
                                                              std=[0.2373, 0.2266, 0.2374])
                                                 
                                         ])
        
        
        # uploading train dataset to take a portion and augment it then concate with train dataset
        aug_dataset = torchvision.datasets.ImageFolder(root = train_path, transform = augment_transforms)
        discrad_size = int(0.8 * len(aug_dataset))
        aug_size = len(aug_dataset) - discrad_size
        
        _ , transformed_dataset = torch.utils.data.random_split(aug_dataset, [discrad_size, aug_size])
        train_dataset = torch.utils.data.ConcatDataset([transformed_dataset, train_dataset])

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

## Fine-tuning the VGG16 CNN Model(Pretrained) to make compatible with our datasets
def modified_model(model):
    
    """
    This function will modify the pretrained imported model as VGG16, resnet, inception
    as these huge network and fine tuning on even small dataset like iNaturalist
    is very expensive. 
    VGG16 has 13 Convolutional layers and 3 linear dense layer, 
    i will freeze all the convolutional layers except the last two.
    
    """
    
    
    # Replacing the last layer VGG16 to make it compatible with iNaturalist ten class dataset
    model.classifier[6] = Linear(in_features=4096, out_features=10, bias=True)
    # freezing the parameters of the convolutional layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # defreezing 13th conv layer of vgg16
    for param in model.features[28].parameters():
        param.requires_grad = True
        
    # defreezing 12th conv layer of vgg16
    for param in model.features[26].parameters():
        param.requires_grad = True

        
        
    return model
        
## Function to find the accuracy
# # Define evaluation function
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

## Creating training function
def train_CNN(optimizer,batch_size, data_augmentation,device,num_epochs,learning_rate):
    
    """
    This function is created to bring all the inputs needed to train the modified VGG16
    CNN model and calculate the train loss, train accuracy and validation accuracy.
    First we have choosen Adam as optimizer with learning rate= 1e-4, and weight
    decay = 1e-4. Then cross entropy loss is choosen as the loss function.
    data_pre_processing function initialized to give the dataloders and the datasets
    needed to train and evaluate the model.
    model is set to .train() mode to do backprop with forward prop to train the model.
    In training the images in batch are taken and feed to forward prop to get loss 
    value then backprop is performed then the weights and biases are updated.
    After training for each epoch train accuracy and validation accuracy are calculated.
    Best model is saved for further use like to calculate test accuracy.
    
    """
   
    print("Training the model in progress==>>")
    if wandb_sweep == True:
        #default values for wandb run
        config_defaults = {
            'optimizer': 'adam',
            'data_augmentation': True,
            'num_epochs': 5,
            'learning_rate': 0.0001
             }

        #initialize wandb
        wandb.init(project = wandb_project,config=config_defaults)

        # config is a data structure that holds hyperparameters and inputs
        config = wandb.config

        # Local variables, values obtained from wandb config
        data_augmentation = config.data_augmentation
        optimizer = config.optimizer
        num_epochs = config.num_epochs
        learning_rate = config.learning_rate

        # Defining the run name in wandb sweep
        wandb.run.name  = "lr_{}_DA_{}_opt_{}_epoch_{}_".format(learning_rate,
                                                                data_augmentation,
                                                                  optimizer,                                                                         
                                                                  num_epochs)                                                                                                                                                 




        print(wandb.run.name )
    
    # uploading the vgg16 pretrained model
    model = torchvision.models.vgg16(pretrained=True)
    # modify the VGG16 model 
    model = modified_model(model)
    # sending model to device(GPU or CPU)
    model = model.to(device)

    #Optimizer and loss function
    if optimizer == 'adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.0001)
    elif optimizer == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
        
    loss_function=nn.CrossEntropyLoss()
    
    # uploading the dataloader and datasets
    train_loader, test_loader, val_loader, train_dataset,test_dataset = data_pre_processing(batch_size, data_augmentation)
    
    
    # Training on training dataset
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)               
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
        val_accuracy = evaluate(model, val_loader)

        # printing the epoch, train loss, train accuracy and validation accuracy after each epoch
        print("Epoch: "+str(epoch+1)+ ' Train Loss: '+ str(train_loss) +' Train Accuracy: '+
              str(train_accuracy) + ' Validation Accuracy: '+ str(val_accuracy)) 
        
        if wandb_sweep == True:
            wandb.log({"validation accuracy": val_accuracy, "train accuracy": train_accuracy, 
                        "train loss": train_loss, 'epoch': epoch+1})

    if wandb_sweep == True:
        wandb.run.name 
        wandb.save()
        wandb.run.finish()
        
    # if code running on local machine returning the model    
    if wandb_sweep == False:
        return model
    
if wandb_sweep == False:
    train_CNN(optimizer,batch_size, data_augmentation,device, 
              num_epochs,learning_rate) 
    
if wandb_sweep == True:
    import wandb

### Wandb Sweep
def sweep():
    
    """
    This function is used to exploit the wandb hyperparameter sweep 
    function to get the best hyperparameters.
    It takes in no inputs and gives no outputs.
    Instead it logs everything into the wandb workspace'''

    """
    sweep_config = {"name": wandb_project, "method": "bayes"}   
    sweep_config["metric"] = {"name": "val_accuracy", "goal": "maximize"}

    #Declaring the dictionary of all choices for the hyperparameters.
    parameters_dict = {
                  'optimizer': {"values": ['adam', 'sgd']}, 
                  "data_augmentation": {"values": [True, False]},
                  "num_epochs": {"values": [5, 10]},
                 "learning_rate": {"values": [0.0001, 0.001]}
                    }
    sweep_config["parameters"] = parameters_dict
    
    # creating the sweep id and starting the sweep agent to run the hyper parameter configuration
    sweep_id = wandb.sweep(sweep_config, entity=wandb_entity, project=wandb_project)
    wandb.agent(sweep_id, train_CNN(optimizer,batch_size, data_augmentation,device, 
              num_epochs,learning_rate) )
    
### Training the model
if wandb_sweep == True:
    sweep()

### Test the model on the best configuration of hyper parameters
def test_model(optimizer, learning_rate,batch_size,
                               num_epochs, data_augmentation):
    
    """
    This function test the model on the best parameters find out after 
    hyper parameter tuning using wandb.
    set the arguments of the best validation accuracy run in the ConvNeuNet().
    This function train the model on the best parameters and give the test accuracy 
    on the test dataset.
    
    """
    
    print("Testing the model===>>>")
    
     # uploading the vgg16 pretrained model
    model = torchvision.models.vgg16(pretrained=True)
    # modify the VGG16 model 
    model = modified_model(model)
    # sending model to device(GPU or CPU)
    model = model.to(device)

    #Optimizer and loss function
    if optimizer == 'adam':
        optimizer=torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.0001)
    elif optimizer == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(),lr=0.0001)
        
    loss_function=nn.CrossEntropyLoss()
    
    # uploading the dataloader and datasets
    train_loader, test_loader, val_loader, train_dataset,test_dataset = data_pre_processing(batch_size, data_augmentation)
    
    
    

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
            torch.save(model.state_dict(), 'best_partB_checkpoint.model')
            best_accuracy=test_accuracy


        print("Epoch: "+str(epoch+1)+ ' Train Loss:'+ str(train_loss) +' Train Accuracy:'+
              str(train_accuracy) + ' Test Accuracy: '+ str(test_accuracy))
        
    return model
        
# Testing the model at the best parameter
model = test_model(optimizer='adam', learning_rate=0.0001,batch_size=16,
                               num_epochs=5, data_augmentation=False)
