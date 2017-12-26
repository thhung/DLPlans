from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input

from keras import backend as K
K.set_image_dim_ordering('tf')


train_data_dir = 'data/dog_cat_super_small/train'
validation_data_dir = 'data/dog_cat_super_small/validation'
nb_train_samples = 23000
nb_validation_samples = 2000
epochs = 3
batch_size = 16
img_width, img_height = 224, 224

input_tensor = Input(shape=(224,224,3))
base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
print('Model loaded.')

x = base_model.output
x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.1),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size)

model.save("cat_dog_model_ep3_RMSprop.h5")

print("Finish process")