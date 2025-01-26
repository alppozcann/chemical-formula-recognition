import easyocr
import numpy as np
import cv2
import os


def remove_text(input_path: str, output_path: str) -> None:
    """
    Remove text from an image using EasyOCR.
    """
    reader = easyocr.Reader(['en'])  # Initialize OCR reader

    # Load the image and preprocess it
    image = cv2.imread(input_path)
    _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)  # Binarize

    # Detect text in the binarized image
    results = reader.readtext(binary)

    for box in results:
        # Extract the four corner points of the bounding box
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = box[0]

        # Convert coordinates to integers
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        x3, y3 = int(x3), int(y3)

        # Create a polygon for the bounding box
        points = np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])

        # Always remove detected text
        cv2.fillPoly(image, pts=[points], color=(255, 255, 255))  # Fill with white color

    cv2.imwrite(output_path, image)  # Save the processed image
    print(f"Processed image saved to {output_path}")

# Example Usage
if __name__ == "__main__":
    base_path = os.getcwd()
    test_images = 'test_images'
    image_name = '1.jpg'
    output_images = 'notext_images'
    input_image_path = os.path.join(base_path, test_images, image_name)  # Input file
    output_image_path = os.path.join(base_path, output_images, image_name)  # Output file

    remove_text(input_image_path, output_image_path)
