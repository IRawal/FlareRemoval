from flask import Flask, request, render_template, send_file
from PIL import Image
import torch
from torchvision import transforms
import subprocess
import os

app = Flask(__name__)

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

class_names = ['Flare', 'No_Flare']

def process_image(image):
    # Apply the transformations
    image = transform(image).unsqueeze(0)
    # Move the image to the device
    image = image.to(device)
    # Predict the image
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    # Decode the prediction
    prediction = class_names[predicted.item()]
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        image = Image.open(file.stream).convert('RGB')
        prediction = process_image(image)

        if prediction == 'Flare':

            # Save the image temporarily in the same format
            temp_path = os.path.join('uploads', f'temp_image.{file.filename.split(".")[-1]}')
            processed_path = os.path.join('uploads', f'processed_image.{file.filename.split(".")[-1]}')
            image.save(temp_path)

            # Absolute path to the streak_removal.py script
            streak_removal_script = "streak_removal.py"

            # Run the streak_removal script with the required arguments
            result = subprocess.run(['python', streak_removal_script, '-i', temp_path, '-o', processed_path], capture_output=True, text=True)

            if result.returncode == 0:
                # Determine MIME type based on file extension
                if processed_path.endswith('.png'):
                    mime_type = 'image/png'
                elif processed_path.endswith('.jpg') or processed_path.endswith('.jpeg'):
                    mime_type = 'image/jpeg'
                else:
                    mime_type = 'application/octet-stream'  # Default binary type
                
                # Send the processed image file as a response for download
                return send_file(processed_path, mimetype=mime_type, as_attachment=True)
            else:
                return f'Error in streak_removal.py: {result.stderr}'
        
        return f'The image is predicted as: {prediction}'

if __name__ == '__main__':
    app.run(port=3000, debug=True)
