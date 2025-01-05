import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from unetModel import *

def load_preprocessed_data():
    '''
    images = np.load('./Stage_LIRIS_INRAE/arrowDetection/data/preprocessed_images.npy')
    masks = np.load('./Stage_LIRIS_INRAE/arrowDetection/data/preprocessed_masks.npy')
    '''
    images = np.load('/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/arrowDetection/data/preprocessed_images_512.npy')
    masks = np.load('/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/arrowDetection/data/preprocessed_masks_512.npy')
    return images, masks

if __name__ == "__main__":
    print("Training U-Net model for arrow detection...")
    # Charger les données prétraitées
    images, masks = load_preprocessed_data()
    images = tf.convert_to_tensor(images)
    masks = tf.convert_to_tensor(masks)
    
    
    # Vérifier les dimensions des données
    assert images.shape[1:] == (512, 512, 1), f"Expected image shape (512, 512, 1), but got {images.shape[1:]}"
    assert masks.shape[1:] == (512, 512, 1), f"Expected mask shape (512, 512, 1), but got {masks.shape[1:]}"

    print("Data loaded successfully!")
    
    # Créer le modèle U-Net
    model = unet(input_size=(512, 512, 1))

    print("Model created successfully!")
    
    # Afficher le résumé du modèle pour vérifier l'architecture
    model.summary()
    
    # Entraîner le modèle
    model.fit(images, masks, batch_size=32, epochs=50, validation_split=0.1)
    
    # Sauvegarder le modèle entraîné
    # eski
    #model.save('/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/saved_models/unet_model_512.h5')
    model.save('/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/saved_models/unet_model_512.keras')