import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tqdm.notebook import tqdm
from sklearn.cluster import MiniBatchKMeans

def extract_and_classify(image_og, square_size=32, batch_size=256):
    # Calculate padding size
    pad_size = square_size // 2

    # Pad the image with zeros
    image_padded = cv2.copyMakeBorder(image_og, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)

    # Initialize VGG16 model with weights pre-trained on ImageNet
    vgg16 = VGG16(weights='imagenet', include_top=False)

    # Initialize an empty array for the labels
    labels = np.zeros(image_og.shape[:2])

    # Create an array to hold the regions
    regions = np.zeros((batch_size, 224, 224, 3))

    # list of pixel coordinates
    pixel_coordinates = []

    # Iterate over each pixel in the original image
    i = 0
    for y in tqdm(range(image_og.shape[0])):
        for x in tqdm(range(image_og.shape[1]), leave=False):
            # Extract a square region around the pixel from the padded image
            region = image_padded[y:y+square_size, x:x+square_size]

            # Resize the region to the input size that VGG16 expects
            region = cv2.resize(region, (224, 224))

            # Add the region to the batch
            regions[i] = region

            # Store pixel coordinates
            pixel_coordinates.append((y, x))

            # If the batch is full, process it and reset the batch
            if i == batch_size - 1 or (y == image_og.shape[0] - 1 and x == image_og.shape[1] - 1):
                # Convert the batch to an array and preprocess it for VGG16
                regions = np.array(regions, dtype=np.float64)
                regions = preprocess_input(regions)

                # Extract features
                features = vgg16.predict(regions)

                # Assign the mean feature to the corresponding pixels
                for j, feature in enumerate(features):
                    y, x = pixel_coordinates[j]
                    labels[y, x] = np.mean(feature)

                # Reset the batch
                i = 0
                regions = np.zeros((batch_size, 224, 224, 3))
                pixel_coordinates = []
            else:
                i += 1

    return labels


def cluster_slic(image_og, labels, n_segments=100, compactness=10):
    # Perform SLIC segmentation
    segments = slic(image_og, n_segments=n_segments, compactness=compactness)
    
    # Classify each segment
    for segment_val in np.unique(segments):
        mask = np.zeros(image_og.shape[:2], dtype="bool")
        mask[segments == segment_val] = True
        clustered_label = np.mean(labels[mask])
        
        labels[segments == segment_val] = clustered_label

    return labels


def classify_kmeans(features, n_clusters=3):
    # Initialize MiniBatchKMeans algorithm
    minibatch_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100)

    # Fit the descriptors
    minibatch_kmeans.fit(features.reshape(-1, 1))

    # Get the cluster labels
    labels = minibatch_kmeans.labels_

    return labels.reshape(features.shape[:2])


def extract_orb_features(image_path, n_features=1000):
    # Load the image
    image = cv2.imread(image_path, 0) # 0 for grayscale

    # Initialize the ORB detector and detect keypoints
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints = orb.detect(image, None)

    # Compute the descriptors
    keypoints, descriptors = orb.compute(image, keypoints)
    return descriptors
