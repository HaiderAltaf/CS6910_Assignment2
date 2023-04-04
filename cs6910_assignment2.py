#!/usr/bin/env python
# coding: utf-8

# # Part-A

# ## Question-1

# Source1- https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/ 

# source2- https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

# ### load Relevant libraries

# In[1]:


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU, GELU, SELU, Mish
from torch.nn import LogSoftmax
from torch import flatten


# In[3]:


# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


print(device)


# In[ ]:





# ### Data Pre-Processing

# In[5]:


# Use transforms.compose method to reformat images for modeling,
# and save to variable all_transforms for later use, find the size, mean and std for each channel of our dataset
def data_pre_processing(batch_size=64, data_augmentation=False):
    
    all_transforms = transforms.Compose([transforms.Resize((150,150)),
                                         transforms.ToTensor(),        #0-255 to 0-1 & numpy to tensor
                                         #transforms.Normalize(mean=[0.4713, 0.4600, 0.3897],  #0-1 to [-1,1]
                                                              #std=[0.2373, 0.2266, 0.2374])
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5],  #0-1 to [-1,1]
                                                              std=[0.5, 0.5, 0.5])
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
        augment_transforms = transforms.Compose([transforms.Resize((150,150)),                                         
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomRotation((-60,60)),
                                                 transforms.ToTensor(),
#                                                  transforms.Normalize(mean=[0.4713, 0.4600, 0.3897], 
#                                                               std=[0.2373, 0.2266, 0.2374]),
                                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],  #0-1 to [-1,1]
                                                              std=[0.5, 0.5, 0.5])
                                                 
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


# In[6]:


train_loader, test_loader, val_loader, train_dataset, test_dataset = data_pre_processing(batch_size=64,
                                                                                        data_augmentation=False)


# ### Finding the mean and std of our dataset

# In[7]:


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# In[8]:


#get_mean_and_std(train_loader)
#[0.4713, 0.4600, 0.3897], [0.2373, 0.2266, 0.2374] for iNaturalist dataset at resize=[150,150]


# ### Show image of samples

# In[9]:


def show_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
    batch = next(iter(loader))
    images, labels = batch
    print(images.shape)
    
    grid = torchvision.utils.make_grid(images, nrow = 3)
    plt.figure(figsize= (11,11))
    plt.imshow(np.transpose(grid, (1,2,0)))
    print('labels', labels)                                   


# In[10]:


#show_images(train_dataset)


# In[11]:


class Addition():
    def __init__(self, x,y):
        self.x = x
        self.y = y
    def add(self):
        return self.x +self.y


# In[12]:


#Addition(0,4).add()


# ### CNN Model

# In[13]:


class ConvNeuNet(Module):
    
    def __init__(self, size_kernel=3, num_stride=1, act_fu='gelu', size_denseLayer=500,
                 data_augmentation=True, batch_normalisation=True, input_channels=3,
                 classes=10, padding=0, kernel_org=1, dropout_rate=0.2, num_filters=10):
        
        # call the parent constructor
        super(ConvNeuNet, self).__init__()
        
        self.batch_norm = batch_normalisation
        self.data_aug = data_augmentation
        width = 150
        height = 150
        
        #(batch_size = 64, input_channels=3, width=150, height=150)
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=input_channels, out_channels=num_filters,
                    kernel_size=size_kernel, stride=num_stride, padding=padding)
        width = ((width- size_kernel + 2*padding)/num_stride) + 1
        height = ((height- size_kernel + 2*padding)/num_stride) + 1
        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        self.af1 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.maxpool1 = MaxPool2d(kernel_size=size_kernel, stride=num_stride, padding=padding)
        
        # updating width and height of the next layer after maxpool
        width = ((width- size_kernel)/num_stride) + 1
        height = ((height- size_kernel)/num_stride) + 1
        
        # Add dropout layer
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # initialize second set of CONV => RELU => POOL layers
        size_kernel = size_kernel*kernel_org
        self.conv2 = Conv2d(in_channels=num_filters, out_channels=num_filters,
                     kernel_size=size_kernel, stride=num_stride, padding=padding)
        width = ((width- size_kernel + 2*padding)/num_stride) + 1
        height = ((height- size_kernel + 2*padding)/num_stride) + 1
        self.bn2 = nn.BatchNorm2d(num_features=num_filters)
        self.af2 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.maxpool2 = MaxPool2d(kernel_size=size_kernel, stride=num_stride, padding=padding)
        
        # updating width and height of the next layer after maxpool
        width = ((width- size_kernel)/num_stride) + 1
        height = ((height- size_kernel)/num_stride) + 1
        # initialize third set of CONV => RELU => POOL layers
        size_kernel = size_kernel*kernel_org
        self.conv3 = Conv2d(in_channels=num_filters, out_channels=num_filters,
                     kernel_size=size_kernel, stride=num_stride, padding=padding)
        width = ((width- size_kernel + 2*padding)/num_stride) + 1
        height = ((height- size_kernel + 2*padding)/num_stride) + 1
        self.bn3 = nn.BatchNorm2d(num_features=num_filters)
        self.af3 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.maxpool3 = MaxPool2d(kernel_size=size_kernel, stride=num_stride, padding=padding)
        
         # updating width and height of the next layer after maxpool
        width = ((width- size_kernel)/num_stride) + 1
        height = ((height- size_kernel)/num_stride) + 1
        # initialize fourth set of CONV => RELU => POOL layers
        size_kernel = size_kernel*kernel_org
        self.conv4 = Conv2d(in_channels=num_filters, out_channels=num_filters,
                     kernel_size=size_kernel, stride=num_stride, padding=padding)
        width = ((width- size_kernel + 2*padding)/num_stride) + 1
        height = ((height- size_kernel + 2*padding)/num_stride) + 1
        self.bn4 = nn.BatchNorm2d(num_features=num_filters)
        self.af4 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.maxpool4 = MaxPool2d(kernel_size=size_kernel, stride=num_stride, padding=padding)
        
        # updating width and height of the next layer after maxpool
        width = ((width- size_kernel)/num_stride) + 1
        height = ((height- size_kernel)/num_stride) + 1
        # initialize fifth set of CONV => RELU => POOL layers
        size_kernel = size_kernel*kernel_org
        self.conv5 = Conv2d(in_channels=num_filters, out_channels=num_filters,
                     kernel_size=size_kernel, stride=num_stride, padding=padding)
        width = ((width- size_kernel + 2*padding)/num_stride) + 1
        height = ((height- size_kernel + 2*padding)/num_stride) + 1
        self.bn5 = nn.BatchNorm2d(num_features=num_filters)
        self.af5 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        self.maxpool5 = MaxPool2d(kernel_size=size_kernel, stride=num_stride, padding=padding)
        
        # updating width and height of the next layer after maxpool
        width = ((width- size_kernel)/num_stride) + 1
        height = ((height- size_kernel)/num_stride) + 1
        #initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=int(num_filters*width*height), out_features=size_denseLayer)
        self.af6 = ReLU() if act_fu=='relu' else GELU() if act_fu=='gelu' else SELU() if act_fu=='selu' else Mish()
        
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
        x = self.maxpool1(x)
       
        # pass the output from the previous layer through the second
        # set of CONV => Batch_norm => RELU => layers
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.af2(x)
        x = self.maxpool2(x)
        # pass the output from the previous layer through the third
        # set of CONV => Batch_norm => RELU => POOL layers
        x = self.conv3(x)
        if self.batch_norm:
            x = self.bn3(x)
        x = self.af3(x)
        x = self.maxpool3(x)
        # pass the output from the previous layer through the fourth
        # set of CONV => Batch_norm => RELU => POOL layers
        x = self.conv4(x)
        if self.batch_norm:
            x = self.bn4(x)
        x = self.af4(x)
        x = self.maxpool4(x)
        # pass the output from the previous layer through the fifth
        # set of CONV => Batch_norm => RELU => POOL layers
        x = self.conv5(x)
        if self.batch_norm:
            x = self.bn5(x)
        x = self.af5(x)
        x = self.maxpool5(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.af6(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


# In[14]:


model = ConvNeuNet().to(device)


# In[15]:


#Optimizer and loss function
optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()


# In[16]:


num_epochs=10


# In[17]:


#Model training and saving best model

best_accuracy=0.0
for epoch in range(num_epochs):
    
    #Evaluating and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        #print(images.shape())
        if torch.cuda.is_available():
            images=images.cuda()
            labels=labels.cuda()
        
        # make grad to zero after each batch
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/len(train_dataset)
    train_loss=train_loss/len(train_dataset)
        
    
    #Evaluating on validation dataset
    model.eval()
    
    test_accuracy=0.0
    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=images.cuda()
            labels=labels.cuda()
            
        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))
        
    test_accuracy=test_accuracy/len(test_dataset)
    
    print("Epoch: "+str(epoch)+ 'Train Loss: '+str(int(train_loss))+'Train Accuracy: '+
          str(int(train_accuracy))+ 'Test Accuracy: '+str(int(test_accuracy)))  
    
    
    # save the best model
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'best_checkpoint.model')
        best_accuracy=test_accuracy


# In[ ]:





# In[ ]:




