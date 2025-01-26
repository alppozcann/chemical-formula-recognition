import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")

def detect_arrow_heads(image_path, model_path):
    """
    Detects arrow heads in the given image using the specified model.
    
    Parameters:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained U-Net model.
        
    Returns:
        tuple: Coordinates of detected arrow heads and the original image.
    """
    model = tf.keras.models.load_model(model_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    print("Image loaded successfully. Resizing...")
    image_resized = cv2.resize(image, (512, 512))
    print("Image resized successfully!") 
    image_array = np.expand_dims(image_resized, axis=[0, -1]) / 255.0

    # Perform prediction using the model
    try:
        prediction = model.predict(image_array)[0, :, :, 0]
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

    # Apply threshold to detect arrow heads
    threshold = 0.3
    arrow_heads = np.where(prediction_resized > threshold)

    return arrow_heads, image

def plot_image_with_detections(image, arrow_heads, image_file):
    """
    Visualizes the original image and marks the detected arrow heads.
    
    Parameters:
        image (numpy.ndarray): Original grayscale image.
        arrow_heads (tuple): Coordinates of detected arrow heads.
        image_file (str): Name of the image file for the plot title.
    """
    plt.imshow(image, cmap='gray')
    for i in range(len(arrow_heads[0])):  # Y-coordinates
        plt.scatter(arrow_heads[1][i], arrow_heads[0][i], color='red', s=10)  # X and Y coordinates

    plt.title(f"Arrow Heads Detected in {image_file}")
    plt.axis('off')  # Hide axis
    plt.show()  # Wait until the window is closed

if __name__ == "__main__":
    # Get the paths of all images in the specified directory
    base_path = os.getcwd()
    image_dir = os.path.join(base_path, 'data/images')
    model_path = os.path.join(base_path, 'saved_models/unet_model_512.keras')

    # Retrieve all image files from the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Process and display each image
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # Call detect_arrow_heads function
        arrow_heads, image = detect_arrow_heads(image_path, model_path)
        
        # Log the detected arrow head positions
        print(f"Arrow heads detected in {image_file} at positions:", arrow_heads)

        # Visualize the results
        plot_image_with_detections(image, arrow_heads, image_file)