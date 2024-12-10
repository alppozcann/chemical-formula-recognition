import numpy as np
import cv2
import os

def load_data(image_dir, mask_dir, img_size=(512, 512)):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if (filename!='.DS_Store'):
            print(filename)
            #lecture des images et des masques
            image = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
            #application d'un seuillage pour obtenir des masques binaires
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            #redimensionnement des images et des masques
            image = cv2.resize(image, img_size)
            mask = cv2.resize(mask, img_size)
            images.append(image)
            masks.append(mask)

            #cv2.imshow('image', image)
            #cv2.imshow('mask', mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    #normalisation des images
    images = np.array(images).astype('float32') / 255.0 
    masks = np.array(masks).astype('float32') / 255.0
    #ajout d'une dimension pour les images pour les rendre compatibles avec le modèle
    images = np.expand_dims(images, axis=-1 )
    masks = np.expand_dims(masks, axis=-1)

    return images, masks

if __name__ == "__main__": 
    '''
    image_dir = './Stage_LIRIS_INRAE/arrowDetection/data/images'
    mask_dir = './Stage_LIRIS_INRAE/arrowDetection/data/masks'
    '''
    image_dir = '/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/arrowDetection/data/images'
    mask_dir = '/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/arrowDetection/data/masks'
    images, masks = load_data(image_dir, mask_dir)
    #sauvegarde des images et des masques prétraités dans un fichier numpy
    np.save('/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/arrowDetection/data/preprocessed_images_512.npy', images)
    np.save('/Users/alpates/Desktop/INSA_LYON/Duffner1/Stage_LIRIS_INRAE/arrowDetection/data/preprocessed_masks_512.npy', masks)

"""num_files = 20
    split_images = np.array_split(images, num_files)
    split_masks = np.array_split(masks, num_files)

    for i, split in enumerate(split_images):
        np.save(f'data/preprocessed_images_part_{i}.npy', split)
        np.save(f'data/preprocessed_masks_part_{i}.npy', split_masks[i])"""
