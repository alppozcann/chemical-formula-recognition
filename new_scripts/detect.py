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

    # Prédiction
    print(type(model))
    print(dir(model))
    try:
        prediction = model.predict(image_array)[0, :, :, 0]
        print("Prediction completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {e}")
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

    # Seuil pour détecter les têtes de flèches
    threshold = 0.4
    arrow_heads = np.where(prediction_resized > threshold)

    return arrow_heads

    

if __name__ == "__main__":
    # Test için bir resim yolu belirtin
    image_path = '/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/fichiers_sources/dossiers_de_test/test2/images/selected/image_0_8_2.jpeg'
    model_path = '/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/saved_models/unet_model_512.keras'
    
    #process image
    
    # Detect_arrow_heads fonksiyonunu çağır
    arrow_heads = detect_arrow_heads(image_path, model_path)
    
    # Çıktıyı kontrol et
    print("Arrow heads detected at positions:", arrow_heads)
    
    # Görüntüyü tekrar yükleyin ve görselleştirme için gösterin
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(image, cmap='gray')

    # Ok başlarını işaretleyin
    for i in range(len(arrow_heads[0])):  # Y koordinatları
        plt.scatter(arrow_heads[1][i], arrow_heads[0][i], color='red', s=10)  # X ve Y koordinatları

    plt.title("Arrow Heads Detected")
    plt.show()
    print(os.getcwd())

