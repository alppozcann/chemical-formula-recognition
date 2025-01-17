import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
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

    # Kontrastı artırma
    image = cv2.convertScaleAbs(image, alpha = 1.3, beta = 0)  # alpha=1.5 kontrastı artırır, beta=0 parlaklık

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
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=10,maxLineGap=8)


    for points in lines:
        x1,y1,x2,y2=points[0]
        lines_list.append([(x1,y1),(x2,y2)])
    return lines_list


def visualize_results(image, centroids, intersecting_lines):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Centroidleri işaretle
    for cx, cy in centroids[1:]:
        cv2.circle(output_image, (int(cx), int(cy)), 5, (0, 255, 0), -1)  # Yeşil noktalar

    # Çizgileri çiz (sadece kesişenleri vurgula)
    for (x1, y1), (x2, y2) in intersecting_lines:
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Sarı çizgiler

    plt.figure(figsize=(8, 8))
    plt.title("Centroids and Intersecting Lines")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show
    cv2.imshow(output_image)
    cv2.waitKey(0)
    return output_image
    
# bounding box calculation
def get_rotated_bounding_box(x1, y1, x2, y2, threshold=20):
    # Çizgiyi bir nokta listesine dönüştür
    points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

    # Minimum alanlı dikdörtgeni (rotated bounding box) al
    rect = cv2.minAreaRect(points)

    # Dikdörtgenin boyutlarını genişlet (threshold kadar esneklik ekle)
    center, (width, height), angle = rect
    rect = (center, (width + threshold, height + threshold), angle)

    # Dikdörtgenin köşelerini al
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Köşe noktalarını tamsayıya dönüştür

    return box, rect


def find_lines_intersecting_components(detected_lines, labels, centroids, threshold=10):
    intersecting_lines = []
    visited_lines = set()
    # İlk centroid arka plan olduğundan, onu atlıyoruz
    for cx, cy in centroids[1:]:
        
         # Centroid'in x ve y koordinatları
        for (x1, y1), (x2, y2) in detected_lines:
            # Çizgi için bounding box hesapla
            x_min = min(x1, x2) - threshold
            x_max = max(x1, x2) + threshold
            y_min = min(y1, y2) - threshold
            y_max = max(y1, y2) + threshold
            
            # Eğer centroid bounding box içinde ise
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                intersecting_lines.append(((x1, y1), (x2, y2)))
                visited_lines.add(((x1, y1), (x2, y2)))
                find_intersecting_lines(detected_lines, intersecting_lines, visited_lines, x1, x2, y1, y2)
                break  # Her centroid için yalnızca bir çizgiyi ekle
    return intersecting_lines


def find_intersecting_lines(detected_lines, intersecting_lines, visited_lines, x1, x2, y1, y2, threshold = 10):
    
    # Çizginin bounding box'ını hesapla
            x_min = min(x1, x2) - threshold
            x_max = max(x1, x2) + threshold
            y_min = min(y1, y2) - threshold
            y_max = max(y1, y2) + threshold
            
                
            
            for (X1, Y1), (X2, Y2) in detected_lines:
                if ((X1, Y1), (X2, Y2)) not in visited_lines:
                    if x_min <= X1 <= x_max and y_min <= Y1 <= y_max or x_min <= X2 <= x_max and y_min <= Y2 <= y_max :
                        intersecting_lines.append(((X1, Y1), (X2, Y2)))
                        visited_lines.add(((X1, Y1), (X2, Y2)))
                        find_intersecting_lines(detected_lines, intersecting_lines, visited_lines, X1, X2, Y1, Y2)

'''

def find_intersecting_lines(detected_lines, intersecting_lines, visited_lines, x1, x2, y1, y2, threshold=15):
    # Çizginin bounding box'ını döndür
    box, rect = get_rotated_bounding_box(x1, y1, x2, y2, threshold=threshold)

    for (X1, Y1), (X2, Y2) in detected_lines:
        if ((X1, Y1), (X2, Y2)) not in visited_lines:
            # Yeni çizginin bounding box'unu döndür
            new_box, new_rect = get_rotated_bounding_box(X1, Y1, X2, Y2, threshold=threshold)

            # İki bounding box'un kesişip kesişmediğini kontrol et
            if cv2.rotatedRectangleIntersection(rect, new_rect)[0] != cv2.INTERSECT_NONE:
                intersecting_lines.append(((X1, Y1), (X2, Y2)))
                visited_lines.add(((X1, Y1), (X2, Y2)))
                find_intersecting_lines(detected_lines, intersecting_lines, visited_lines, X1, X2, Y1, Y2, threshold)
'''
def get_result(image_path, model_path, result_path):
    binary_mask = detect_arrow_heads(image_path, model_path)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    num_labels, labels, stats, centroids = connected_components_analysis(binary_mask, original_image)
    detected_lines = detect_lines(original_image)
    intersecting_lines = find_lines_intersecting_components(detected_lines, labels, centroids)  # centroids eklendi
    output_image = visualize_results(original_image, centroids, intersecting_lines)
    cv2.imwrite(result_path, output_image)


if __name__ == "__main__":
    # Image path
    base_path = os.getcwd()
    test_images = 'test_images'
    image_name = '1.jpg'
    model_name = 'unet_model_512.keras'
    image_path = os.path.join(base_path, test_images, image_name)
    model_path = os.path.join(base_path, model_name)
    
    
    # Arrow head detection
    binary_mask = detect_arrow_heads(image_path, model_path)

    # Görüntüyü tekrar yükleyin
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Connected Components Analysis
    num_labels, labels, stats, centroids = connected_components_analysis(binary_mask, original_image)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    detected_lines = detect_lines(original_image)

    #eskisi
    intersecting_lines = find_lines_intersecting_components(detected_lines, labels, centroids)
    
    # Sonuçları görselleştir
    visualize_results(original_image, centroids, intersecting_lines)
    