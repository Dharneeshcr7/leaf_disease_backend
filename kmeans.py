import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time
from skimage.color import rgb2lab
from PIL import Image
import requests
from model_unet.segmenter import model
# Load the image
def read_image_from_url(url):
    # Download the image from the URL
    image = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), cv2.IMREAD_COLOR)
    
    # Resize the image
    image = cv2.resize(image, (224, 224))
    
    # Convert the OpenCV image to a PIL Image object
    pil_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #pil_image=pil_image.convert("RGB")
    
    pil_image = np.array(pil_image)
    

    # Convert RGB image to LAB color space
    # laab_image = rgb2lab(pil_image)

    
    # laab_image = laab_image.astype(np.int16)
    # lab_image = laab_image[:, :, 1:]
    # reshaped_lab = lab_image.reshape((-1, 2))

    return pil_image
def get_kmeans(img):
    num_clusters = 6
    laab_image = rgb2lab(img)

    
    lab_image = laab_image[:, :, 1:]
    reshaped_lab = lab_image.reshape((-1, 2))

    kmeans = KMeans(n_clusters=num_clusters)

    kmeans.fit(reshaped_lab)

    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    clustered_labels = labels.reshape(lab_image.shape[:2])
    cluster_masks = [(clustered_labels == i).astype(np.uint8) * 255 for i in range(num_clusters)]

    return cluster_centers,cluster_masks

def get_back(cluster_masks,rgb_image):
    rgb_values_main = []

    gray_values_main=[]

# Iterate over each cluster
    for i in range(6):
        
        cluster_gray_values=[]
        # Iterate over each pixel in the clustered masks
       # print(cluster_masks.shape)
        cluster_masks=np.array(cluster_masks)
        for y in range(cluster_masks.shape[1]):
            
            gray_values_row=[]
            for x in range(cluster_masks.shape[2]):
                # Get the cluster label at the current pixel
                cluster_label = cluster_masks[i, y, x]

                # If the pixel is not part of the background (cluster 0)
                if cluster_label > 0:
                    # Get the RGB values from the input image
                    rgb_value = rgb_image[y, x]
                    gray_value=0.3*rgb_value[0]+0.59*rgb_value[1]+0.11*rgb_value[2]
                    # Append the RGB values to the list for this row
                    
                    gray_values_row.append(gray_value)
                else:
                    # Append None for pixels that do not belong to the cluster
                    
                    gray_values_row.append(0)

            # Append the row to the cluster's RGB values
            
            cluster_gray_values.append(gray_values_row)

        # Append the cluster's RGB values to the main list
        
        gray_values_main.append(cluster_gray_values)


    # for i in range(6):
    #   rgb_values_main[i] = np.array(rgb_values_main[i])
    
    gray_values_main= np.array(gray_values_main)

    max_hist = []
    for i in range(6):
        hist, bins = np.histogram(gray_values_main[i], 256, [0, 256])
        max_hist.append(max(hist))

    min_hist_index = np.argmin(max_hist)
    return min_hist_index

    


def score_img(img):
    img_array = np.expand_dims(img, axis=0)
    preds=model.predict(img_array)
    preds=np.argmax(preds,axis=3)
    
    preds[0][preds[0]==0]=255
    

    cluster_centers,cluster_masks=get_kmeans(img)
    min_hist_index=get_back(cluster_masks,img)
    predefined_colors = [30,60,120]

# Step 1: Calculate hue (H) and saturation (S) for each cluster
    hue_values = []
    saturation_values = []
    for a, b in cluster_centers:
        H = np.arctan2(b, a) * 180 / np.pi  # Convert radians to degrees
        # if H < 0:
        #     H += 360  # Adjust negative angles
        hue_values.append(H)
        S = np.sqrt((a)*(a) + (b)*(b))
        saturation_values.append(S)
    
    max_saturation = max(saturation_values)
    
    normalized_saturation = [(s / 181.0) * 100 for s in saturation_values]
    
    # Step 4: Classify each cluster based on normalized angular distance and saturation
    cluster_classification = []

    #step 1
    for i in range(len(cluster_centers)):
        cluster_classification.append('None')
        if i==min_hist_index:
          cluster_classification[i]='background'
          continue


        if hue_values[i]<37 and hue_values[i]>15:
           cluster_classification[i]='orange'
        elif hue_values[i]<95:
        #cluster_classification.append('orange')
            if saturation_values[i]<=15:
                cluster_classification[i]='black'
            elif saturation_values[i]<=23:
                cluster_classification[i]='brown'
            else:
                cluster_classification[i]='yellow'
        elif hue_values[i]<140 and hue_values[i]>100:
            cluster_classification[i]='green'
        else:
          continue
    #step2

    for i in range(len(cluster_centers)):
        if cluster_classification[i]!='None':
            continue
        min_index=-1
        min_value=100
        for j in range(len(predefined_colors)):
            if abs(hue_values[i]-predefined_colors[j])<min_value:
                min_index=j
                min_value=abs(hue_values[i]-predefined_colors[j])
        if min_value>45:
            continue

        if min_index==0:
            cluster_classification[i]='orange'
        elif min_index==1:
            if saturation_values[i]<=15:
              cluster_classification[i]='black'
            elif saturation_values[i]<=23:
              cluster_classification[i]='brown'
            else:
              cluster_classification[i]='yellow'

        else:
            cluster_classification[i]='green'

    #step3

    for i in range(len(cluster_centers)):
        if cluster_classification[i]!='None':
            continue
        distances = np.sqrt(np.sum((cluster_centers - cluster_centers[i])**2, axis=1))

        # Exclude the distance to the given cluster center itself
        distances[i] = np.inf

        # Find the index of the nearest cluster center
        nearest_index = np.argmin(distances)

        cluster_classification[i]=cluster_classification[nearest_index]
    pixel_counts={
    'green':0,
    'yellow':0,
    'orange':0,
    'brown':0,
    'black':0,
    'background':0

    }
    area_affected=0
    leaf_area=0
    for i in range(len(cluster_masks)):

        x=np.sum(cluster_masks[i]>0)
        pixel_counts[cluster_classification[i]]+=x
        if(cluster_classification[i]!='background' and cluster_classification[i]!='green'):
            area_affected+=x
        if(cluster_classification[i]!='background'):
            leaf_area+=x
    for k in pixel_counts:
       pixel_counts[k]=pixel_counts[k]/leaf_area
    
    return pixel_counts



def score(urls):
    imgs=[]
    final_counts={
    'green':0,
    'yellow':0,
    'orange':0,
    'brown':0,
    'black':0,
    'background':0

    }
    i=0
    for url in urls:
        if(i>5):
            break
        img=read_image_from_url(url)
        imgs.append(img)
        i+=1
    imgs=np.array(imgs)

    for img in imgs:
        pixel_counts=score_img(img)
        for k in pixel_counts:
            final_counts[k]+=pixel_counts[k]
    
    for k in final_counts:
        final_counts[k]=final_counts[k]/min(len(urls),5)
    return final_counts



