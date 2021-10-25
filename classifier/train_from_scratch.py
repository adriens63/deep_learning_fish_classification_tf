import tensorflow as tf

try:
    from models.cnns import generate_model
except ImportError:
    cc = None
    print('generate_model', ' non importé')

try:
    from data.data_loading import DataLoader
except ImportError:
    cc = None
    print('DataLoader', ' non importé')
    
try:
    from data.data_rescaling_and_augmentation import generate_augmentation, generate_rescaler
except ImportError:
    cc = None
    print('data_rescaling_and_augmentation', ' non importé')

#********************globales********************
SEED = 101
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
data_dir = '/tf/coding/datasets/fish_recognition_gt/fish_image'
split = .2
logs = '../logs'
h, w, d = 64, 64, 3


#********************import des données********************
dl = DataLoader(data_dir, split, h, w, BATCH_SIZE, SEED)
dl.load()
#on rescale et augmente dans le modele



#********************generation du model avec preprocessing intégré********************
mod = generate_model(h, w, d)
rescaler = generate_rescaler(h, w)
augmenter = generate_augmentation()
model = tf.keras.Sequential([rescaler,
                             #augmenter,
                             mod])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



#********************callbacks********************
callbacks = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                           histogram_freq = 10,
                                           write_graph = True,
                                           write_images = True,
                                           update_freq = 'epoch'
                                           )



#********************training********************
history = model.fit(dl.train_ds, epochs = 100, verbose = 1, validation_data = dl.test_ds, callbacks = callbacks,
                      batch_size = BATCH_SIZE)


#********************sauvegarde du modèle********************
model.save('weights/model_1.h5')