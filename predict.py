import torch
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image

def load_model(model_path='resnet18_pneumonia.pth'):
    # Initialize ResNet18
    model = models.resnet18(weights=None)
    # Modify final layer as done in training
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image_path, model):
    # Define the same transformations used in training
    transform_mean = [0.485, 0.456, 0.406]
    transform_std =[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std)
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension
    
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
        
    prediction = "Pneumonia" if probability > 0.5 else "Normal"
    confidence = probability if prediction == "Pneumonia" else 1 - probability
    
    return {"prediction": prediction, "confidence": round(confidence, 4)}

if __name__ == "__main__":
    # Test script locally with a dummy path if desired
    # print(predict_image("path_to_sample_xray.jpg", load_model()))
    pass
