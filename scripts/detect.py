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
    model = tf.keras.models.load_model(model_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    print("Image loaded successfully. Resizing...")
    image_resized = cv2.resize(image, (512, 512))
    print("Image resized successfully!") 
    image_array = np.expand_dims(image_resized, axis=[0, -1]) / 255.0

    # Prediction
    print(type(model))
    print(dir(model))
    try:
        prediction = model.predict(image_array)[0, :, :, 0]
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

    # Threshold to detect arrow heads
    threshold = 0.4
    arrow_heads = np.where(prediction_resized > threshold)

    return arrow_heads


if __name__ == "__main__":
    # Specify a test image path
    base_path = os.getcwd()
    test_image_dir = 'notext_images'
    image_name = '3.jpeg'
    model_name = 'saved_models/unet_model_512.keras'
    image_path = os.path.join(base_path, test_image_dir, image_name)
    model_path = os.path.join(base_path, model_name)
    
    # Process image
    
    # Call detect_arrow_heads function
    arrow_heads = detect_arrow_heads(image_path, model_path)
    
    # Check the output
    print("Arrow heads detected at positions:", arrow_heads)
    
    # Reload the image and display it for visualization
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')

    # Mark the detected arrow heads
    for i in range(len(arrow_heads[0])):  # Y coordinates
        plt.scatter(arrow_heads[1][i], arrow_heads[0][i], color='red', s=10)  # X and Y coordinates

    plt.title("Arrow Heads Detected")
    plt.show()
    print(os.getcwd())