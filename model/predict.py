import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
from torchvision import models, transforms
from sklearn.preprocessing import MinMaxScaler,Normalizer,StandardScaler
import joblib
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from collections import Counter
import uuid
import boto3
#from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, S3_BUCKET_NAME
import os
from dotenv import load_dotenv
from filehandler import upload_file_to_s3,get_presigned_file_url,delete_folder
from joblib import load
import io
load_dotenv()

AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY=os.getenv('AWS_SECRET_KEY')
S3_BUCKET_NAME=os.getenv('S3_BUCKET_NAME')
AWS_REGION=os.getenv('AWS_REGION')

s3=boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,

)

transform = transforms.Compose([
    transforms.Resize(224),  # Resize images to EfficientNetB0 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

# Load pre-trained EfficientNet model for feature extraction
model = models.efficientnet_b0(pretrained=True)
labels = {0: 'Apple-Apple Scab', 1: 'Apple-Black Rot', 2: 'Apple-Cedar Apple Rust', 3: 'Apple-Healthy', 4: 'Bell Pepper-Bacterial Spot', 5: 'Bell Pepper-Healthy', 6: 'Cherry-Healthy', 7: 'Cherry-Powdery Mildew', 8: 'Corn (Maize)-Cercospora Leaf Spot', 9: 'Corn (Maize)-Common Rust', 10: 'Corn (Maize)-Healthy', 11: 'Corn (Maize)-Northern Leaf Blight', 12: 'Grape-Black Rot', 13: 'Grape-Esca (Black Measles)', 14: 'Grape-Healthy', 15: 'Grape-Leaf Blight', 16: 'Peach-Bacterial Spot', 17: 'Peach-Healthy', 18: 'Potato-Early Blight', 19: 'Potato-Healthy', 20: 'Potato-Late Blight', 21: 'Strawberry-Healthy', 22: 'Strawberry-Leaf Scorch', 23: 'Tomato-Bacterial Spot', 24: 'Tomato-Early Blight', 25: 'Tomato-Healthy', 26: 'Tomato-Late Blight', 27: 'Tomato-Septoria Leaf Spot', 28: 'Tomato-Yellow Leaf Curl Virus'}

model.eval()
# Load SVM classifier
resp=s3.get_object(Bucket='weights-leaf',Key='classsofication-svm/svm_pytorch.pkl')
body = resp['Body'].read()

    # Load the SVM classifier from the pickle file
svm_classifier = load(io.BytesIO(body))
# with open('model\svm_pytorch.pkl', 'rb') as f:
#     svm_classifier = joblib.load(f, mmap_mode=None)


def most_common_element(lst):
    # Count occurrences of each element in the list
    counts = Counter(lst)
    
    # Find the element with the highest count
    most_common = counts.most_common()
    
    # Return the most common element and its count
    elements=[]
    for i in most_common:
        if i[1]==most_common[0][1]:
            elements.append(i[0])
    return elements

# Function to perform feature extraction
def extract_features(image_path):
    
    image = (Image.open(BytesIO(requests.get(image_path).content)))
    image=image.convert("RGB")
    img=np.array(image)
    #print(img.shape)
    
    image=transform(image).unsqueeze(0)  # Add a batch dimension
    with torch.no_grad():
        features = model.features(image)  # Extract features from convolutional layers
        pooled_features = torch.nn.functional.adaptive_avg_pool2d(features, output_size=(1, 1)).squeeze()
        return pooled_features.numpy()

def get_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else None
    return tuple(shape)

# Function to perform classification
def classify_image(img):
    
    features=[]
    features1 = extract_features(img)
    features.append(features1)
    
    features=np.array(features)
    prediction = svm_classifier.predict(features)
    return prediction
def classify_image_dir(img_dir):
    
    features=[]
    for j in tqdm(os.listdir(img_dir)):
        if j.lower().endswith((".jpg", ".png", ".jpeg")):
            features1=extract_features(os.path.join(img_dir,j))
            features.append(features1)
            
    features=np.array(features)
    prediction = svm_classifier.predict(features)
    return prediction
def classify_image_urls(img_urls):
    
    features=[]
    for j in img_urls:
        
        #if j.lower().endswith((".jpg", ".png", ".jpeg")):
        features1=extract_features(j)
        features.append(features1)
            
    features=np.array(features)
    #print(features.shape)
    prediction = svm_classifier.predict(features)
    #print(prediction)
    elements=most_common_element(prediction)
    diseases=[]
    leaves=[]
    for i in elements:
        parts=labels[i].split('-')
        leaves.append(parts[0])
        diseases.append(parts[1])
    return leaves,diseases
   
# Example usage



    



#img_dir = r'model\images'


