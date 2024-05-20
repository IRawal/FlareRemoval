# Flare/Streak Light Reduction

About our project:
1. Detection of Flares: Training image classification model to identify images that contain flare streaks. The model should accurately classify images into 'flare' and 'no flare' categories that has 89% accuracy.

2. Image Processing: For images identified as containing flares, apply a specialized script, such as 'streak_removal.py,' to remove the flare streaks. This script will process the image to reduce or eliminate the visual artifacts caused by the flares.

3. Webpage User Interface: Build a webpage for user interface where they can upload the image and use the pre-trained model to process and output new image.

How to set up:
- run load_and_test_model.py to save the classifier path
- Go to app.py and run the file
- After the environment has been set up, open your web browser and type "localhost:3000"
- The webpage will be rendered, upload your image file and click upload
- The processed image will be given to you. Save it in your preferred location
