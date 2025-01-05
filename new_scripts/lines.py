import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image = cv2.imread("/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/arrowDetection/data/test_images/kimya.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Gri tonlamaya dönüştür

# Kenar tespiti (Canny Edge Detection)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Hough Transform ile çizgi tespiti
lines = cv2.HoughLines(edges, 1, np.pi/180, 180)

# Çizgileri çiz
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Kırmızı çizgiler

# Görüntüyü göster
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines")
plt.axis("off")
plt.show()