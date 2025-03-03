# import torch
# import pandas as pd


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2



def load_model(path):
    """
    Loads a trained PyTorch Vision Transformer model
    
    path ==> path to the model weights (.pth file)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.vit_b_16(weights=None)

    # Modify the classifier for binary classification
    base_model.heads = nn.Sequential(
        nn.Linear(base_model.hidden_dim, 512),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    base_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    base_model = base_model.to(device)

    return base_model

# def predict(image, model):
#     # Labels used during training
#     labels = {0: 'Normal', 1: 'Glaucoma'}
    
    # # Model expected image dimensions
    # img_width, img_height = 224, 224
    
    # # Define the same image transformation used during testing
    # test_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((img_width, img_height)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    # # Convert BGR to RGB format
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # # Apply transformations
    # image_tensor = test_transform(image)
    
    # # Add batch dimension
    # image_tensor = image_tensor.unsqueeze(0)
    
#     # Predict with the model
#     with torch.no_grad():
#         output = model(image_tensor)
        
#     # Convert the output probability (sigmoid) to binary prediction
#     probability = output.item()  # Get the raw probability value
#     prediction = 1 if probability >= 0.5 else 0
    
#     # Get the class label
#     result = labels.get(prediction)
    
#     # Create dataframe of prediction results
#     raw_data = [
#         ['Normal', 1 - probability],
#         ['Glaucoma', probability]
#     ]
    
#     # Make dataframe
#     dataframe = pd.DataFrame(raw_data, columns=["condition", "probability"])
    
#     return result, dataframe

def predict(image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Model expected image dimensions
    img_width, img_height = 224, 224
        
        # Define the same image transformation used during testing
    test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert BGR to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
    image_tensor = test_transform(image)
        
        # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    result = model(image_tensor).item()
    return result
