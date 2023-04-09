
# CS6910 Assignment-2

The assignment contains two parts.
In Part-A, I have build CNN model from scratch using pytorch to train on iNaturalist dataset.
In Part-B, I have imported pretrained VGG16 CNN model using pytorch and modified it to make suitable for iNaturalist dataset(having only 10k train images).


## Authors

 [@Haider Altaf am22s020](https://www.github.com/HaiderAltaf)


## Part-A

### References:

- Source1- https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/

- source2- https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/


- source3- https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

### Libraries Used: 

- Used the pytoch, __pytorchvision__ and other necessary libraries for the CNN model buiding and training.

### Device:
I have write and run the code in Jupyter notebook and used GPU of my system for training.

- Device will determine whether to run the training on GPU or CPU.

  code: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Data Pre-processing:

- Created a function named __data_pre_processing__ to upload the downloaded datasets(train and test).
- Give path of the downloaded train and test dataset in specified places to upload it.
- The output of the function are __train_loader__, __test_loader__, __val_loader__, __train_dataset__, __test_dataset__.

### Installing wandb and importing wandb:
Run the below code 
- !pip install wandb
- import wandb

### CNN model:
- My model has five consecutive convolutional layers, each layer has set of __2D convolution__ => __Batch Nromalisation__ => __non-linear activation function__ => __Dropout__ => __max pool layers__.
- The code is  flexible such that the number of filters, size of filters, and activation function of the convolution layers and dense layers can be changed. We can change the number of neurons in the dense layer.
- I have created a class named __ConvNeuNet(nn.Modeule)__, in this nitialised the __init__ function with arguments having flexible inputs. In __init__ initialised all the convolution layers, dense layer and output layer.
- Within the class, I have created __forward()__ function in which all the layers are arranged in sequence. The output of forward is a tensor having probability of ten classes.

### Define evaluate function:
- __evalute()__ function created to find the accuracy of the any dataloader(train_loader etc).
- This function will be used to find accuracy during training and testing phase.

### Function for training the model:
- Created __train_CNN()__ function to train our model.
- In this, first model is defined and it is exported to the device (either GPU or CPU).
- Optimizer and loss function is imported using __torch__ library.
  __optimizer__=torch.optim.__Adam__(model.parameters(),__lr__=0.0001,__weight_decay__=0.0001)
  __loss_function__=nn.__CrossEntropyLoss__()
- I have also include the commands needed to integrate the __wandb__ __sweep__. I have __login__ to wandb account. I have already imported the __wandb__, now I am giving the __default values__ of our variable for __sweep__. After that I have defined the __wandb run name__ which will be assign to each run. Values like __epoch__, __train loss__, __train accuracy__ and __validation accuracy__ are login to wandb.
- __Saving__ the wandb run and __finishing__ the run

### Running the wandb sweep:

The wandb configuration for sweep:
- sweep_config = {"name": "cs6910_assignment2", "method": "bayes"}   
- sweep_config["metric"] = {"name": "val_accuracy", "goal": "maximize"}

- parameters_dict = {
            
             "num_filters": {"values": [[12,12,12,12,12],[4,8,16,32,64],[64,32,16,8,4]},
              "act_fu": {"values": ["relu","selu","mish"]},
              "size_kernel": {"values": [[(3,3),(3,3),(3,3),(3,3),(3,3)], [(3,3),(5,5),(5,5),(7,7),(7,7)],
                                         [(7,7),(7,7),(5,5),(5,5),(3,3)]]}, 
                "data_augmentation": {"values": [True, False]} ,
                "batch_normalisation": {"values": [True, False]} ,
                "dropout_rate": {"values": [0, 0.2, 0.3]},
                "size_denseLayer": {"values": [50, 100, 150, 200]}
                }
- sweep_config["parameters"] = parameters_dict

- sweep_id = wandb.sweep(sweep_config, entity="am22s020", project="cs6910_assignment2")
- wandb.agent(sweep_id, train_CNN, count=150)

### Testing the best model on Test dataset:
I have created function named __test_model()__ to test the trained model using test dataset.

We need to configure diffrenet parameters of the best model that we found using sweep runs. In default I have given the parameters which give best results for my runs.

After running the 54 sweep runs, I found the following parameter as giving best validation accuracy:

- __Validation_accuracy__: 35.7%

- epochs = 10

- __Filter Size__:  [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]] (In corresponding layers)

- __Activation function__ : selu

- __Numner of Fiters__: [12, 12, 12, 12, 12]  (In corresponding layers)

- __Data Augmentation__: True

- __Batch Normalization__: True

- __Dropout__: 0.3

- __Number of neurons in Dense Layer__ : 200

After setting the above parameters, I have run the __test_model()__ to check the test accuracy.

- __Test Accuracy__ found is __35.15%__



## Appendix

Any additional information goes here

