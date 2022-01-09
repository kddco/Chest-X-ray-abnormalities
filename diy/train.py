import os

import tensorflow as tf
# import tensorflow.keras.layers as L
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Recall, Precision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_v1():
    in1 = tf.keras.layers.Input(shape=(256, 256, 1))

    #     out1 = tf.keras.layers.Conv2D(4,(3,3),activation="relu")(in1)
    out1 = tf.keras.layers.Conv2D(32, (3, 3),
                                  activation="relu",
                                  padding='same')(in1)
    out1 = tf.keras.layers.MaxPooling2D((2, 2))(out1)

    out1 = tf.keras.layers.Conv2D(32, (3, 3),
                                  activation="relu",
                                  padding='same')(out1)
    out1 = tf.keras.layers.MaxPooling2D((2, 2))(out1)

    out1 = tf.keras.layers.Flatten()(out1)

    out2 = tf.keras.layers.Dense(30, activation="relu")(out1)
    out2 = tf.keras.layers.Dense(30, activation="relu")(out2)
    out2 = tf.keras.layers.Dense(15,
                                 activation="sigmoid",
                                 name='class_out',
                                 kernel_regularizer=regularizers.l2(0.01))(out2)

    model = tf.keras.Model(inputs=in1, outputs=out2)
    model.compile(loss={'class_out': 'categorical_crossentropy'},
                  optimizer="adam",
                  metrics=[Recall(), Precision(), 'accuracy'])

    return model


model = build_v1()
model.summary()
# tf.keras.utils.plot_model(model)


# import pre_process
#
# train_df = pre_process.get_train_df()
from keras.models import load_model
model.save('my_model.h5')  # creates a HDF5 file 'model.h5'






