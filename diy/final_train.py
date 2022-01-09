from keras.models import load_model

import tensorflow.keras.backend as K
import tensorflow as tf
import pre_process
from diy.Loader import DataLoader
import numpy as np
import load_csv
def build_v2():
    in1 = tf.keras.layers.Input(shape=(256, 256, 1))

    out1 = tf.keras.layers.Conv2D(64, (3, 3),
                                  activation="relu")(in1)

    out1 = tf.keras.layers.MaxPooling2D((2, 2))(out1)

    out1 = tf.keras.layers.Conv2D(64, (3, 3),
                                  activation="relu")(out1)

    out1 = tf.keras.layers.MaxPooling2D((2, 2))(out1)

    out1 = tf.keras.layers.Flatten()(out1)

    out2 = tf.keras.layers.Dense(128, activation="relu")(out1)

    out2 = tf.keras.layers.Dense(15,
                                 activation="sigmoid")(out2)

    model2 = tf.keras.Model(inputs=in1, outputs=out2)

    return model2


# model2 = build_v2()
# model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model2.summary()
# model2.save('my_model2.h5')  # creates a HDF5 file 'model.h5'

model2 = tf.keras.models.load_model("my_model2.h5")

chckpt = tf.keras.callbacks.ModelCheckpoint(f'./model2_f{fold}.hdf5',monitor='val_loss',mode='min',save_best_only=True)

train_set = np.load('../Chest_X-ray_Starter/dcm2numpy.npy')
model2.fit(train_set,
           epochs=10

           validation_data=(load_csv.get_train(), Y_eval),
           callbacks=[chckpt]
           )

