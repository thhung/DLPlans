# from keras import applications
# from keras.preprocessing.image import ImageDataGenerator
# from keras import optimizers
# from keras.models import Sequential, Model
# from keras.layers import Dropout, Flatten, Dense, Input

# from keras.models import load_model

# from keras import backend as K
# K.set_image_dim_ordering('tf')


# # class Tester:

# #     def __init__(self, path, batch_size = 16, zero_pad = False, extension = 'jpg'):
# #         self.path = path
# #         self.batch_size = batch_size
# #         self.zero_pad = zero_pad
# #         self.extension = extension
# #         self.nb_zero = 1
        

# #     def scout(self):
# #         path, dirs, files = next(os.walk(self.path))
# #         self.nb_zero = len(str(len(files)))

# model = load_model("cat_dog_model.h5")


# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode=None,
#     shuffle=False)

# validation_generator.filenames()

# model.predict_genrator()


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input

from keras.models import load_model
import numpy as np

from keras import backend as K
K.set_image_dim_ordering('tf')

# import bcolz
# def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
# def load_array(fname): return bcolz.open(fname)[:]



model = load_model("cat_dog_model_ep3.h5")

test_datagen = ImageDataGenerator(rescale=1. / 255)

nb_file = 1000
batch_size = 16
validation_data_dir = 'data/test'
img_width, img_height = 224, 224

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
print("Load done")

print(validation_generator.samples)

predicts = model.predict_generator(validation_generator,use_multiprocessing=True, verbose=1) # , validation_generator.samples
print("predict done")
our_predictions = 1 - predicts[:,0]

filenames = validation_generator.filenames
ids = np.array([int(f[8:f.find('.')]) for f in filenames])

subm = np.stack([ids, our_predictions], axis=1)
submission_file_name = 'submission1.csv'
np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')