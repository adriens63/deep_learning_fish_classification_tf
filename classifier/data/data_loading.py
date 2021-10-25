import tensorflow as tf
import tensorflow.keras.layers as tfnn

#********************globales********************
SEED = 101
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
data_dir = '/tf/coding/datasets/fish_recognition_gt/fish_image'
split = .2
h, w = 64, 64

#********************import des données********************

class DataLoader:
    def __init__(self, data_dir, split, h, w, BATCH_SIZE, SEED):
        self.data_dir = data_dir
        self.split = split
        self.h = h
        self.w = w
        self.BATCH_SIZE = BATCH_SIZE
        self.SEED = SEED
    
    
    def load(self):
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                                    self.data_dir,
                                                                    validation_split = self.split,
                                                                    subset = 'training',
                                                                    seed = self.SEED,
                                                                    batch_size = self.BATCH_SIZE)

        self.test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                                    self.data_dir,
                                                                    validation_split = self.split,
                                                                    subset = 'validation',
                                                                    seed = self.SEED,
                                                                    batch_size = self.BATCH_SIZE)
        
        self.train_ds = self.train_ds.prefetch(buffer_size = AUTOTUNE).cache()
        self.test_ds = self.test_ds.prefetch(buffer_size = AUTOTUNE).cache()
    
    
    def rescale_and_augment(self, shuffle = False, augment = False):
        """Si on ne veut pas integrer le preprocessing au modele, on utilise cette 
        methode, 
        
        l'augmentatoin de donnée sera fait asynchronement sur CPU sauf si on prefetch
        
        les layers de preprocessing ne seront pas exportés, on doit les attacher manuellement
        si on veut les sauvegarder
        
        On augmente que la train
        

        Args:
            shuffle (bool, optional): [description]. Defaults to False.
            augment (bool, optional): [description]. Defaults to False.
        """
        self.rescaler = tf.keras.Sequential([
                                tfnn.Resizing(self.h, self.w),
                                tfnn.Rescaling(1./255)])
        self.augmenter = tf.keras.Sequential([
                                tfnn.RandomFlip("horizontal_and_vertical"), #flip dans les 2 sens
                                tfnn.RandomRotation(0.2), #20% de 2 pi = 20% d'un tour dans les deux sens
                                #cf les tfnn.Random... pour voir toute l'augmentation
                                ])
        
        def prepare(ds, shuffle=False, augment=False):
            # Resize and rescale all datasets.
            ds = ds.unbatch().map(lambda x, y: (self.rescaler(x), y), 
                        num_parallel_calls=AUTOTUNE)

            if shuffle:
                ds = ds.shuffle(1000)

            # Batch all datasets.
            ds = ds.batch(self.BATCH_SIZE)

            # Seulement sur le training set.
            if augment:
                ds = ds.map(lambda x, y: (self.augmenter(x, training=True), y), 
                            num_parallel_calls=AUTOTUNE)

            # Use buffered prefetching on all datasets.
            return ds.prefetch(buffer_size=AUTOTUNE).cache()
        
        self.train_ds = prepare(self.train_ds, shuffle = shuffle, augment = augment)
        self.test_ds = prepare(self.test_ds)
    
    
    def prefetch_and_cache(self):
        self.train_ds = self.train_ds.prefetch(buffer_size = AUTOTUNE).cache()
        self.test_ds = self.test_ds.prefetch(buffer_size = AUTOTUNE).cache()
    
    

# dl = DataLoader(data_dir, split, h, w, BATCH_SIZE, SEED)
# dl.load()
# print(dl.train_ds)
# print(dl.train_ds.cardinality())
# dl.rescale_and_augment(augment = True)
# print('ok')

# x, y = next(dl.train_ds.as_numpy_iterator())
# print(x.shape, y)

# augmenter = tf.keras.Sequential([
#                                 tfnn.RandomFlip("horizontal_and_vertical"), #flip dans les 2 sens
#                                 tfnn.RandomRotation(0.2), #20% de 2 pi = 20% d'un tour dans les deux sens
#                                 #cf les tfnn.Random... pour voir toute l'augmentation
#                                 ])

# print(augmenter(x).shape)

# import matplotlib.pyplot as plt
# plt.imshow(x[0])
# plt.show()
