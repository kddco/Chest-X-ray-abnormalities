
import keras
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
import tensorflow as tf

from diy import pre_process
from diy.Loader import DataLoader

for fold in range(3):
    X_train = pre_process.train_df[pre_process.train_df.kfold != fold]
    X_val = pre_process.train_df[pre_process.train_df.kfold == fold]
    dl = DataLoader('D:/machinelearning_xay_chest/download_from_kaggle_npy/train/', X_train, X_val)
    train_set = dl.flow(batch_size=32)
    X_eval, Y_eval = dl.getVal()

    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(256, 256, 1)))
    model.add(Flatten())
    model.add(Dense(64))  # 更改512
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16))  ##  新增加   # 更改64
    model.add(Activation('relu'))  ##  新增加
    model.add(Dense(1))  # 更改1
    model.add(Activation('sigmoid'))

    opt= tf.optimizers.Adam()

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit_generator(
        train_set,
        epochs=10,
        shuffle=True,
        steps_per_epoch=int(15000 / 32),
        validation_data=(X_eval, Y_eval),

    )

# save model
model.summary()
# model.save("train3.h5")
