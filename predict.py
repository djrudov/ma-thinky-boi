import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import argparse
import json



def load_checkpoint(filepath, device):
    # This old so-and-so was trained on a GPU, but I got you if you've only got a CPU...
    if str(device) == 'cpu':
        checkpoint = torch.load(filepath, map_location='cpu')
    else:
        checkpoint = torch.load(filepath)
    
    # Load architecture type specified in checkpoint dictionary
    arch = checkpoint['arch']        
    model = models.__dict__[arch](pretrained=True)
    #Freeze paramaters
    for param in model.parameters():
        param.requires_grad = False
    
    # Keep our guy nice and cozy when he pops in from the abyss, gotta make sure the architecture fits his phatness
    classifier = checkpoint['classifier']
    
    # Load more things from checkpoint dictionary
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    class_to_idx = checkpoint['class_to_idx']
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
        
    # Define the ol' transform switcheroos, so everything goes in all neat and tidy
    switcheroonies = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    image = switcheroonies(im)
    return image



def prediction(file_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    # Ignore your request for GPU because this VM makes it messy and CPU is fine for this
    model.to('cpu')
        
    #reset those badbois for when you swap flowers on me, you sneaky mcsneakerson
    probbies = 0
    classies = 0
    
    #Take specified image and run it through process_image
    image = process_image(file_path)    
    
    #unSQUEEEEEEZE it through the model
    outies = model(image.unsqueeze(0))
    
    #Convert that to a probabobabullitity
    ps = torch.exp(outies).data
    ps_top = ps.topk(topk)
    
    #Flop those indeces around whoospy-doodle
    uno_reverso = {value: key for key, value in model.class_to_idx.items()}
    
    # Assign the probabilities and class definitions to lists
    probbies = ps_top[0].tolist()[0]
    classies = [uno_reverso[i] for i in ps_top[1].tolist()[0]]
    return probbies, classies



def predict_image(model, image_path, checkpoint, top_k, category_names):
    # Just give 'er a lil' name swap
    file_path = image_path

    # Predict what the image is through the upstairs method
    probbies, classies = prediction(file_path, model, top_k)

    # Translate class to names, save in a list
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[i] for i in classies]
   
    # Assign our 'best guess'
    best_guess = class_names[0]

    # Print the requested number of guesses
    print("\nOur {} best guesses (in order) are:".format(top_k))
    for i in range(len(class_names)):
        print(str(i + 1) + ". " + class_names[i])
    print("\n")
    

    # a simple fluorish
    modifier = ""
    prob1 = probbies[0]
    if 0.5 <= prob1 <= 0.7:
        modifier = " big"
    elif 0.7 < prob1 <= 0.95:
        modifier = " HUGE"
    elif prob1 > 0.95:
        modifier = " an insane amount of"
    else:
        modifier = ""

    # 'a' vs 'an' depending on if flower starts with a vowel
    vowel = ""
    if best_guess[0] in ['a', 'e', 'i', 'o', 'u']:
        vowel = "n"
    

    print("I would bet{} money that that's a{} {}.".format(modifier, vowel, best_guess))
    print("")



def main():
    
    # Define the argument parser
    parser = argparse.ArgumentParser()
    
    # Set parsing and defaults for incoming arguments
    parser.add_argument('image_path', type=str, default='flowers/', help='Path to the image to predict')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth', help='Path of checkpoint.pth file')
    parser.add_argument('--top_k', type=int, default = 5, help='Return top X most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path of \'Category to Names\' file')
    parser.add_argument('--gpu', action='store_true', help='Whether to use GPU, if available')
    
    # Run the parse
    args, _ = parser.parse_known_args()

    # Load args from argparser
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    
    category_names = 'cat_to_name.json'
    if args.category_names:
        category_names = args.category_names
    
    device = 'cpu'
    gpu = False
    if args.gpu:
        if torch.cuda.is_available():
            gpu = True
            device = 'cuda'
        else:
            print("No CUDA GPU available")
            device = 'cpu'
    
    # Load the checkpoint into the model
    model = load_checkpoint('checkpoint.pth', device)
    
    #Run the image through the model to get our class predictions
    predict_image(model, image_path, checkpoint, top_k, category_names)
    

    
    
if __name__ == '__main__':
    main()