import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

# Tesseract'ın kurulu olduğu yolu belirtin (MacOS için gerekebilir)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Örneğin MacOS için


def remove_text_from_image(image_path):
    # Görüntüyü yükle
    image = cv2.imread(image_path)

    # Görüntünün bir kopyasını oluştur
    image_copy = image.copy()

    # Görüntüyü gri tonlara dönüştür
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Kontrastı arttırma
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)  # Kontrastı daha fazla artırdık

    # Görüntüdeki detayları yakalamak için biraz bulanıklaştırma ekleyebiliriz
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Tesseract ile metni tanı (daha hassas tanıma için farklı konfigürasyonlar kullanılabilir)
    custom_config = r'--oem 3 --psm 6'  # --psm 6 sayfayı tamamen metin olarak algılar
    text = pytesseract.image_to_string(gray, config=custom_config)

    # Metnin bulunduğu alanları tespit et
    h, w, _ = image_copy.shape
    boxes = pytesseract.image_to_boxes(gray)

    # Her karakterin etrafını çizen kutuları al
    for b in boxes.splitlines():
        b = b.split()
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

        # Her kutuyu beyaz (arka plan rengi) ile doldur
        image_copy[y:h, x:w] = (255, 255, 255)

    # Orijinal resme dokunmadık, sadece değiştirilen versiyonu döndürüyoruz
    return image_copy


def test_remove_text(image_path):
    # Text kaldırma fonksiyonunu çağır
    processed_image = remove_text_from_image(image_path)

    # İşlenmiş görüntüyü matplotlib ile göster
    plt.figure(figsize=(8, 8))
    plt.title("Processed Image (Text Removed)")
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Test için bir resim yolu belirtin
image_path = '/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/fichiers_sources/dossiers_de_test/test3/images/selected/image_0_7_2.jpeg'

# Testi çalıştır
test_remove_text(image_path)