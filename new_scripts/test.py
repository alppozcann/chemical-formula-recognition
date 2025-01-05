import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# GPU bellek büyümesini etkinleştir
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
    try:
        prediction = model.predict(image_array)[0, :, :, 0]
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

    # Threshold for detecting arrow heads
    threshold = 0.3
    arrow_heads = np.where(prediction_resized > threshold)

    return arrow_heads, image

def plot_image_with_detections(image, arrow_heads, image_file):
    """ Görselleştirme ve ok başlarını işaretleme """
    plt.imshow(image, cmap='gray')
    for i in range(len(arrow_heads[0])):  # Y koordinatları
        plt.scatter(arrow_heads[1][i], arrow_heads[0][i], color='red', s=10)  # X ve Y koordinatları

    plt.title(f"Arrow Heads Detected in {image_file}")
    plt.axis('off')  # Ekseni gizle
    plt.show()  # Pencereyi kapatana kadar bekler

if __name__ == "__main__":
    # Belirtilen dizindeki tüm fotoğrafların yolunu al
    image_folder = '/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/arrowDetection/data/images/'
    model_path = '/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/saved_models/unet_model_512.keras'

    # Dizin içindeki tüm resimleri al
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Fotoğrafı her defasında görüntüle ve bir sonrakini hemen aç
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        # Detect_arrow_heads fonksiyonunu çağır
        arrow_heads, image = detect_arrow_heads(image_path, model_path)
        
        # Çıktıyı kontrol et
        print(f"Arrow heads detected in {image_file} at positions:", arrow_heads)

        # Görselleştirmeyi yap
        plot_image_with_detections(image, arrow_heads, image_file)
        
        # Sonraki resim için pencereyi kapatana kadar bekle