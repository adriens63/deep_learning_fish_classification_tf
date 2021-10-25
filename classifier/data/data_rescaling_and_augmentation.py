import tensorflow as tf
import tensorflow.keras.layers as tfnn

def generate_rescaler(h, w):
    """
    Comme on fait l'augmentation ici via le modele, elle va profiter 
    de l'accélération du GPU
    
    Et quand on exportera via model.save, les layers de preprocessing seront 
    aussi sauvegardés, donc quand on le déploira, le modele va standardizer 
    les images automatiquement

    Args:
        h ([type]): [description]
        w ([type]): [description]

    Returns:
        [type]: [description]
    """
    return tf.keras.Sequential([
                                tfnn.Resizing(h, w),
                                tfnn.Rescaling(1./255)])

def generate_augmentation():
    return tf.keras.Sequential([
                                tfnn.RandomFlip("horizontal_and_vertical"), #flip dans les 2 sens
                                tfnn.RandomRotation(0.2), #20% de 2 pi = 20% d'un tour dans les deux sens
                                #cf les tfnn.Random... pour voir toute l'augmentation
                                ])