## Part B : Fine-tuning a pre-trained model

### 1. References:

- Source1- https://pytorch.org/docs/master/notes/autograd.html

- source2- https://www.youtube.com/watch?v=qaDe0qQZ5AQ&list=PL2zZiFciIyS1RYRbav7DIwcwEEw8ir_QU&index=6

### 2. These things are same as in Part-A

- Libraries used 

- device

- Arg parse

- data pre-processing

- evaluate function

- installing and importing wandb

### 3. CNN Model

I have imported the __pre-trained VGG16__ CNN model available in pytorch. The VGG16 is trained on ImageNet dataset.

 - Code to upload the model: model = torchvision.models.vgg16(__pretrained=True__)

I will __modify__ the model to make it suitable for __iNaturalist__ dataset (only 10,000 training samples).

### 4. Modification in the VGG16

I have created a function named as __modified_model()__ to modify the pretrained model, it take model and modification choice as input. 

First, I have modified the last layer of the vgg16 model to make the ouput having 10 classes instead of 1000.

    Code :  model.classifier[6] = Linear(in_features=4096, out_features=10, bias=True)

Then, I have given options to freeze the convolutional layers to prevent from updating of the parameters during the training using iNaturalist dataset

VGG16 has 13 Convolutional layers and 3 linear dense layer.
  
I will freeze only the convolutional layers.

Foloowing options are available for freezing.

    option1 = "freeze all", Freeze all layers except the fully connected layers.
    option2 = "freeze first 12", Freezing all convolutional layers except last convolutional layer.
    option3 = "freeze first 10", Freeze the first 10 convolutional layers except 3.

### 5. Train the modified model on __iNaturalist__ dataset

I have created a function named as __train_model()__.

- This function is created to bring all the inputs needed to train the modified VGG16 CNN model 

- calculate the train loss, train accuracy and validation accuracy.
    
- First we have choosen Adam as optimizer with __learning rate= 1e-4__, and __weight decay = 1e-4__. 

- Then __cross entropy loss__ is choosen as the loss function.
    
- __data_pre_processing__ function initialized to give the dataloders and the datasets needed to train and evaluate the model.

- model is set to __.train() mode__ to do backprop with forward prop to train the model.

- for training,  the images in batch are taken and feed to forward prop to get loss value then backprop is performed then the weights and biases are updated.


- After training for each epoch train accuracy and validation accuracy are calculated and printed

### 6. Arg parse 

created function __arg_parse()__ to pass the command line arguments.

- Using argparse, I have define the arguments and options that my program accepts,
- argparse will run the code, pass arguments from command line and 
    automatically generate help messages.
- __I have given the defaults values for 
    all the arguments, so code can be run without passing any arguments.__
    
Description of various command line arguments

    --wandb_sweep : Do you want to sweep or not: Enter True or False. Default value is False. 
    --wandb_entity : Login username for wandb. Default is given but if you are already using wandb, you will be logged in automatically.
    --wandb_project : name to initialize your run. No need to mention if you are just trying the code.
    --data_augmentation : Data Augmentation: True or False
    --epochs : Number of Epochs: integer value
    --batch_size : Batch Size: integer value
    --optimizer : choice of optimizer for backprop : "adam" or "sgd"
    --learning_rate: learning rate of optimizer: 0.0001 or 0.001
   
    
### 7. Training our CNN model directly by running the command line arguments
I have created training function called __train_partB.py__ file it has everything needed for training and testing our model, we can run the code using command line arguments. 

Or we may also use .ipynb file for partB problem to train the model and test it.

### 8. Best Model checked on test dataset

You can download the model from the link  google driven given below.

    https://drive.google.com/file/d/1YNrTsK7mum6IavcyN4NqPJd-nfpr_TK_/view?usp=sharing
## Part B : Fine-tuning a pre-trained model

### 1. References:

- Source1- https://pytorch.org/docs/master/notes/autograd.html

- source2- https://www.youtube.com/watch?v=qaDe0qQZ5AQ&list=PL2zZiFciIyS1RYRbav7DIwcwEEw8ir_QU&index=6

### 2. These things are same as in Part-A

- Libraries used 

- device

- Arg parse

- data pre-processing

- evaluate function

- installing and importing wandb

### 3. CNN Model

I have imported the __pre-trained VGG16__ CNN model available in pytorch. The VGG16 is trained on ImageNet dataset.

 - Code to upload the model: model = torchvision.models.vgg16(__pretrained=True__)

I will __modify__ the model to make it suitable for __iNaturalist__ dataset (only 10,000 training samples).

### 4. Modification in the VGG16

I have created a function named as __modified_model()__ to modify the pretrained model, it take model and modification choice as input. 

First, I have modified the last layer of the vgg16 model to make the ouput having 10 classes instead of 1000.

    Code :  model.classifier[6] = Linear(in_features=4096, out_features=10, bias=True)

Then, I have given options to freeze the convolutional layers to prevent from updating of the parameters during the training using iNaturalist dataset

VGG16 has 13 Convolutional layers and 3 linear dense layer.
  
I will freeze only the convolutional layers.

Foloowing options are available for freezing.

    option1 = "freeze all", Freeze all layers except the fully connected layers.
    option2 = "freeze first 12", Freezing all convolutional layers except last convolutional layer.
    option3 = "freeze first 10", Freeze the first 10 convolutional layers except 3.

### 5. Train the modified model on __iNaturalist__ dataset

I have created a function named as __train_model()__.

- This function is created to bring all the inputs needed to train the modified VGG16 CNN model 

- calculate the train loss, train accuracy and validation accuracy.
    
- First we have choosen Adam as optimizer with __learning rate= 1e-4__, and __weight decay = 1e-4__. 

- Then __cross entropy loss__ is choosen as the loss function.
    
- __data_pre_processing__ function initialized to give the dataloders and the datasets needed to train and evaluate the model.

- model is set to __.train() mode__ to do backprop with forward prop to train the model.

- for training,  the images in batch are taken and feed to forward prop to get loss value then backprop is performed then the weights and biases are updated.


- After training for each epoch train accuracy and validation accuracy are calculated and printed

### 6. Arg parse 

created function __arg_parse()__ to pass the command line arguments.

- Using argparse, I have define the arguments and options that my program accepts,
- argparse will run the code, pass arguments from command line and 
    automatically generate help messages.
- __I have given the defaults values for 
    all the arguments, so code can be run without passing any arguments.__
    
Description of various command line arguments

    --wandb_sweep : Do you want to sweep or not: Enter True or False. Default value is False. 
    --wandb_entity : Login username for wandb. Default is given but if you are already using wandb, you will be logged in automatically.
    --wandb_project : name to initialize your run. No need to mention if you are just trying the code.
    --data_augmentation : Data Augmentation: True or False
    --epochs : Number of Epochs: integer value
    --batch_size : Batch Size: integer value
    --optimizer : choice of optimizer for backprop : "adam" or "sgd"
    --learning_rate: learning rate of optimizer: 0.0001 or 0.001
   
    
### 7. Training our CNN model directly by running the command line arguments
I have created training function called __train_partB.py__ file it has everything needed for training and testing our model, we can run the code using command line arguments. 

Or we may also use .ipynb file for partB problem to train the model and test it.

### 8. Best Model checked on test dataset

You can download the model from the link  google driven given below.

    https://drive.google.com/file/d/1YNrTsK7mum6IavcyN4NqPJd-nfpr_TK_/view?usp=sharing
