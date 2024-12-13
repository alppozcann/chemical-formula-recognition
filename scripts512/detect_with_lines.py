import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
# GPU bellek büyümesini etkinleştir



def detect_arrow_heads(image_path, model_path):
    try:
        model = load_model(model_path)
        print("Model successfully loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    model.summary()
    print("Image loaded successfully. Resizing...") 
    image_resized = cv2.resize(image, (512, 512))
    print("Image resized successfully!")
    image_array = np.expand_dims(image_resized, axis=[0, -1]) / 255.0

    # Prediction
    try:
        prediction = model.predict(image_array)
        print("Prediction shape:", prediction.shape)
        prediction = prediction[0, :, :, 0]
    except Exception as e:
        print(f"Error during prediction: {e}")
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))

    # Binary mask oluştur
    threshold = 0.4
    binary_mask = (prediction_resized > threshold).astype(np.uint8)

    return binary_mask


def connected_components_analysis(binary_mask, original_image):
    # Bağlı bileşen analizi yap
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

    print(f"Total connected components (excluding background): {num_labels - 1}")

    return num_labels, labels, stats, centroids


def detect_lines(image):
    edges = cv2.Canny(image,50,150,apertureSize=5)

    lines_list =[]
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,minLineLength=5,maxLineGap=10)


    for points in lines:
        x1,y1,x2,y2=points[0]
        lines_list.append([(x1,y1),(x2,y2)])
    return lines_list


def visualize_results(image, centroids, lines):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Centroidleri işaretle
    for cx, cy in centroids[1:]:
        cv2.circle(output_image, (int(cx), int(cy)), 5, (0, 255, 0), -1)  # Yeşil noktalar
    
    
    # Çizgileri çiz
    for (x1, y1), (x2, y2) in lines:
        cv2.line(output_image, (x1,y1), (x2,y2), (255,0,0),2)
        
    # Sonuçları görselleştir
    plt.figure(figsize=(10, 10))
    plt.title("Centroids and Matching Lines")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
    
def find_lines_intersecting_components(detected_lines, labels):
    intersecting_lines = []
    
    for cx, cy in centroids[1:]:  # İlk centroid arka plan olduğundan atlanır
        for (x1, y1), (x2, y2) in detected_lines:
            # Çizginin centroid'e uzaklığını hesapla
            line_to_centroid_dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / \
                                     np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            
            # Belirli bir eşik uzaklık altında olan çizgileri seç
            if line_to_centroid_dist < 10:  # 10 pikselden az uzaklıkta
                intersecting_lines.append(((x1, y1), (x2, y2)))
                break  # Her centroid için yalnızca bir çizgiyi ekle
    
    return intersecting_lines

if __name__ == "__main__":
    # Test için bir resim yolu belirtin
    image_path = '/Users/alpozcan/Desktop/chemical-formula-recognition/data/image_line_test.jpg'
    model_path = '/Users/alpozcan/Desktop/chemical-formula-recognition/unet_model_512.keras'

    # Arrow head detection
    binary_mask = detect_arrow_heads(image_path, model_path)

    # Görüntüyü tekrar yükleyin
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Connected Components Analysis
    num_labels, labels, stats, centroids = connected_components_analysis(binary_mask, original_image)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    detected_lines = detect_lines(original_image)

    intersecting_lines = find_lines_intersecting_components(detected_lines, labels)

    # Sonuçları görselleştir
    visualize_results(original_image, centroids, intersecting_lines)
    