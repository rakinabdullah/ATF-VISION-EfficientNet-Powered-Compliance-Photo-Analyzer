# **ATF-VISION: EfficientNet-Powered Compliance Photo Analyzer**

## Description

The **ATF-VISION** project aims to develop an automated photo approval system using EfficientNet for image classification. This project enhances the model with data augmentation techniques and provides interactive result visualization through a Dash web application. The goal is to create a robust tool for compliance photo analysis, ensuring efficient and accurate classification of images.

## Features

- **EfficientNet Model:** Utilizes EfficientNet for high-performance image classification.
- **Data Augmentation:** Incorporates various data augmentation techniques to improve model robustness.
- **Interactive Visualization:** Provides an interactive web application built with Dash for visualizing results.
- **Custom Dataset Class:** Facilitates easy loading and transformation of image data.

## Class Names

The model classifies images into the following categories:

1. Approved
2. Background Issues
3. Black and White
4. Blurry
5. Digitally Altered
6. Hats
7. Picture of a Picture
8. Too Close
9. Too Far

## Installation

To get started with the ATF-VISION project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/rakinabdullah/ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer.git
    cd ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate    # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your dataset:**
   Ensure your dataset is structured correctly and place it in the designated folder.

2. **Run the training script:**
    ```bash
    python train.py
    ```

3. **Launch the Dash web application:**
    ```bash
    python app.py
    ```
    Open your web browser and go to `http://127.0.0.1:8050` to interact with the application.

## Inference Code

The following Python script is used for deploying the inference model on Azure:

```python
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import json
import os
import logging
import base64
import io
import cv2
from rembg import remove

logging.basicConfig(level=logging.INFO)

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.efficientnet = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, num_classes)
    
    def forward(self, x):
        return self.efficientnet(x)

# Class names mapping
class_names = {
    0: "Approved", 1: "Background Issues", 2: "Black and White",
    3: "Blurry", 4: "Digitally Altered", 5: "Hats",
    6: "Picture of a Picture", 7: "Too Close", 8: "Too Far"
}

# Global variables
model = None
transform = None

def init():
    global model, transform
    
    logging.info("Starting init()")
    
    try:
        # Initialize the model
        num_classes = 9
        model = EfficientNetModel(num_classes)
        logging.info("Model initialized")

        # Load the model weights
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_model_hats_progress.pth')
        logging.info(f"Attempting to load model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logging.info("Model loaded successfully")

        # Define the image transformation
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        logging.info("Transform defined")
    except Exception as e:
        logging.error(f"Error in init(): {str(e)}")
        raise

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces[0] if len(faces) > 0 else None

def adjust_subject_position(image):
    face = detect_face(image)
    height, width = image.shape[:2]
    
    if face is None:
        return image, False
    
    x, y, w, h = face
    face_center = (x + w//2, y + h//2)
    
    crop_height = int(h * 3.2)
    crop_width = int(crop_height * 3/4)
    
    crop_top = max(0, face_center[1] - crop_height//2)
    crop_bottom = min(height, crop_top + crop_height)
    crop_left = max(0, face_center[0] - crop_width//2)
    crop_right = min(width, crop_left + crop_width)
    
    if crop_bottom > height:
        crop_top = max(0, height - crop_height)
        crop_bottom = height
    if crop_right > width:
        crop_left = max(0, width - crop_width)
        crop_right = width
    
    cropped = image[crop_top:crop_bottom, crop_left:crop_right]
    resized = cv2.resize(cropped, (413, 531))
    
    return resized, True

def remove_background(image):
    output = remove(image)
    output_np = np.array(output)
    
    white_background = np.ones_like(output_np[:,:,:3]) * 255
    alpha = output_np[:,:,3:] / 255.0
    result = (output_np[:,:,:3] * alpha + white_background * (1 - alpha)).astype(np.uint8)
    
    return result

def run(raw_data):
    logging.info("Starting prediction")
    try:
        # Parse the raw_data JSON
        json_data = json.loads(raw_data)
        logging.info("Input JSON parsed successfully")

        # Check if 'data' key exists and is a list
        if 'data' not in json_data or not isinstance(json_data['data'], list):
            raise ValueError("Input JSON must contain a 'data' key with a list value")

        # Get the first item in the data list
        data_item = json_data['data'][0]

        # Check if 'image' key exists in the data item
        if 'image' not in data_item:
            raise ValueError("Each item in 'data' must contain an 'image' key")

        # Decode the base64 image
        image_bytes = base64.b64decode(data_item['image'])
        image = Image.open(io.BytesIO(image_bytes))
        logging.info("Image decoded successfully")

        # Convert image to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)

        # Apply new processing steps
        processed_image = remove_background(image_np)
        processed_image, was_adjusted = adjust_subject_position(processed_image)
        logging.info("Image processing steps applied")

        # Apply the transformation
        image_transformed = transform(image=processed_image)['image']
        logging.info("Transformation applied")
        
        # Add batch dimension
        image_tensor = image_transformed.unsqueeze(0)
        
        # Get the prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        # Get the class name
        predicted_class = class_names[predicted.item()]
        logging.info(f"Prediction made: {predicted_class}")
        
        # Return the result as JSON
        return json.dumps({
            "predicted_class": predicted_class,
            "background_removed": True,
            "subject_adjusted": was_adjusted
        })
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {str(e)}")
        return json.dumps({"error": "Invalid JSON input"})
    except ValueError as e:
        logging.error(f"Value Error: {str(e)}")
        return json.dumps({"error": str(e)})
    except Exception as e:
        logging.error(f"Unexpected error in prediction: {str(e)}")
        return json.dumps({"error": f"Unexpected error: {str(e)}"})

# This is added to ensure the script can be run locally for testing
if __name__ == "__main__":
    init()
    # You can add test code here to run predictions locally
```


## Visualizations

### Original Data Distribution
<p align="center">
  <img src="https://github.com/rakinabdullah/ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer/blob/main/images/Class%20Distribution.png" alt="Class Distribution">
</p>
The distribution of classes in the original dataset, showing the frequency of each category.

### Confidence Scores
<p align="center">
  <img src="https://github.com/rakinabdullah/ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer/blob/main/images/Confidence%20Scores.PNG" alt="Confidence Scores">
</p>
The confidence scores of the model predictions, highlighting the model's certainty in its classifications.

### Confusion Matrix
<p align="center">
  <img src="https://github.com/rakinabdullah/ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer/blob/main/images/Confusion%20Matrix.png" alt="Confusion Matrix">
</p>
A confusion matrix illustrating the performance of the classification model, showing true positives, false positives, true negatives, and false negatives.

### Feature Enhancement
<p align="center">
  <img src="https://github.com/rakinabdullah/ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer/blob/main/images/Feature%20Enhancement.PNG" alt="Feature Enhancement">
</p>
Visualization of feature enhancement techniques applied to the dataset to improve model performance.

### ROC AUC
<p align="center">
  <img src="https://github.com/rakinabdullah/ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer/blob/main/images/ROC%20AUC.png" alt="ROC AUC">
</p>
Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) metric, demonstrating the model's capability to distinguish between classes.

### Test Results
<p align="center">
  <img src="https://github.com/rakinabdullah/ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer/blob/main/images/Test%20Results.PNG" alt="Test Results">
</p>
Results from the test dataset, summarizing the model's performance on unseen data.

### Training Metrics
<p align="center">
  <img src="https://github.com/rakinabdullah/ATF-VISION-EfficientNet-Powered-Compliance-Photo-Analyzer/blob/main/images/Training%20Metrics.png" alt="Training Metrics">
</p>
Metrics from the training process, including accuracy and loss over epochs, indicating the model's learning curve.

### Contributing

We welcome contributions to enhance the ATF-VISION project. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

MIT License

```
Copyright (c) 2024 Abdullah Al Rakin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


### Contact Information

For questions, feedback, or collaborations, please reach out to:

- **Abdullah Al Rakin**
- **Email:** [abdullahalrakin@gmail.com](mailto:abdullahalrakin@gmail.com)
- **GitHub:** [rakinabdullah](https://github.com/rakinabdullah)
