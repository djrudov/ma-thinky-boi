# Import necessary modules

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import argparse


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # set transforms
    # Transforms for the training set
    train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    # Transforms for the validation and test sets
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms) 
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader, train_data


def train_model(trainloader, validloader, save_dir, arch, learning_rate, input_size, hidden_units1, hidden_units2, epochs, gpu, train_data):
    
    
    # If you asked for it and we have a GPU... wish granted
    if gpu == True:
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Retrive pretrained model
    print("Retrieving model")
    model = eval('models.{}(pretrained=True)'.format(arch))
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
        
    #Define classifier architecture, or, how big we want our boi
    classifier = nn.Sequential(nn.Linear(input_size, hidden_units1),
                              nn.ReLU(),
                              nn.Dropout(p=0.3),
                              nn.Linear(hidden_units1, hidden_units2),
                              nn.ReLU(),
                              nn.Dropout(p=0.2),
                              nn.Linear(hidden_units2, 102),
                              nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()

    model.classifier = classifier

    #Only train classifier parameters. Feature params are frozen. Let It Go.
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    model.to(device)
    
    if str(device) == 'cpu':
        print("Training on CPU. Must specify '--gpu' for CUDA")
    else:
        print("Training on '{}' device".format(device.upper()))
    
    
    #--Train this whooshy boi--
    
    # Set initial parameters
    epochs = epochs
    
    # I have it set to print every 17 steps so it prints 6x per epoch
    print_every = 17
    steps = 0

    print("Training for {} epochs and printing results every {} steps".format(epochs, print_every))
    
    # Loop through the epochs
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            # Send images, labels to specified training device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print update at 'print_every' interval
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                #Turn off gradients for validations
                with torch.no_grad():
                    #pop that boy to eval mode
                    model.eval()
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        valid_loss += criterion(logps, labels)

                        #calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                #pop ya boi back to training
                model.train()

                # Print out the current accuracy data
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
       
    
    def save_checkpoint(arch, input_size, model, train_data):
    # save the checkpoint
        model.class_to_idx = train_data.class_to_idx
        
        # Set up a dictionary to store things in the checkpoint.pth file
        checkpoint_dict = {'arch': arch,
                  'input_size': input_size,
                  'hidden_units1': hidden_units1,
                  'hidden_units2': hidden_units2,
                  'classifier': classifier,
                  'learning_rate': learning_rate,
                  'output_size': 102,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }
        print("Saving checkpoint as {}checkpoint.pth".format(save_dir))
        torch.save(checkpoint_dict, '{}checkpoint.pth'.format(save_dir))
    
    # Run the save method
    save_checkpoint(arch, input_size, model, train_data)           
   
    return model


    


def main():
    
    # Define our argument parser
    parser = argparse.ArgumentParser()
   
    # Parse incoming arguments and set defaults
    parser.add_argument('data_dir', type=str, default='flowers/', help='Path of parent data directory')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory in which to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Torchvision models. Default is \'vgg16\'')
    parser.add_argument('--learning_rate', type=float, default=0.0015, help='Learning rate. Default is 0.0015')
    parser.add_argument('--input_size', type=int, default=25088, help='# of inputs into model. Default is 25088.')
    parser.add_argument('--hidden_units1', type=int, default=8192, help='# of units outputted from first hidden layer. Default is 8192. WARNING: If you\'re getting a size mismatch error, it\'s probably because you need to specify these. Both of them.')
    parser.add_argument('--hidden_units2', type=int, default=512, help='# of units outputted from second layer, into output layer processing. Default is 512.')
    parser.add_argument('--epochs', type=int, default=3, help='How long training is, in epochs. Default is 3.')
    parser.add_argument('--gpu', action='store_true', help='Whether to use GPU, if available')
    
    # Run through the parser
    args, _ = parser.parse_known_args()
    
    # Assign the variables
    data_dir = args.data_dir
    
    save_dir = './'
    if args.save_dir:
        save_dir = args.save_dir
        
    arch = 'vgg16'
    if args.arch:
        arch = args.arch
        
    learning_rate = 0.0015
    if args.learning_rate:
        learning_rate = args.learning_rate
        
    input_size = 25088
    if args.input_size:
        input_size = args.input_size
        
    hidden_units1 = 8192
    if args.hidden_units1:
        hidden_units1 = args.hidden_units1
        
    hidden_units2 = 512
    if args.hidden_units2:
        hidden_units2 = args.hidden_units2
        
    epochs = 3
    if args.epochs:
        epochs = args.epochs
    
    
    gpu = False
    if args.gpu:
        if torch.cuda.is_available():
            gpu = True
        else:
            print("No CUDA GPU available")
            
    
    # Get the loader data from the load_data method (needs 'data_dir')
    trainloader, validloader, testloader, train_data = load_data(data_dir)
    
    # Train the model
    model = train_model(trainloader, validloader, save_dir, arch, learning_rate, input_size, hidden_units1, hidden_units2, epochs, gpu, train_data)
    
if __name__ == '__main__':
    main()