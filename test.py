import torch
from torchvision import transforms, models
from PIL import Image

# Load the saved model
model_path = 'flare_classifier_model.pth'
model = torch.load(model_path)
model.eval()  # Set the model to evaluation mode

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the same transforms as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the image
image_path = 'Test_image/Flare1.jpg'
image = Image.open(image_path)
# Add batch dimension
image = transform(image).unsqueeze(0)

# Predict the image
image = image.to(device)
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# Decode the prediction
class_names = ['Flare', 'No_Flare']
prediction = class_names[predicted.item()]

print(f'The image is predicted as: {prediction}')
