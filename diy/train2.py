import keras
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, Activation, concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

import tensorflow.keras.backend as K
import tensorflow as tf
import pre_process
from diy.Loader import DataLoader
from keras import models
from keras import layers
def build_v2():
    model2=models.Sequential()
    model2.add(layers.Dense(128,activation='relu',input_shape=(256,256,1)))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same'))
    model2.add(Dense(32))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same'))

    model2.add(MaxPool2D(pool_size=(2, 2)))
    model2.add(Activation('sigmoid'))



    return model2


model2 = build_v2()
model2.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

for fold in range(3):
    print(f'\nFold: {fold}\n')

    #     X_train = train_df[train_df.kfold!=fold].drop('kfold',axis=1)
    #     X_val = train_df[train_df.kfold==fold].drop('kfold',axis=1)
    X_train = pre_process.train_df[pre_process.train_df.kfold != fold]
    X_val = pre_process.train_df[pre_process.train_df.kfold == fold]
    print('X_train.shape=', X_train.shape)
    print('X_train.head()=', X_train.head())

    print('-----------\n')

    print('X_val.shape=', X_val.shape)
    print('X_val.head()=', X_val.head())

    print('-----------\n')
    dl = DataLoader('D:/machinelearning_xay_chest/download_from_kaggle_npy/train/',X_train,X_val)
    train_set = dl.flow(batch_size=32)

    X_eval, Y_eval = dl.getVal()
    #     print('X_eval[0]=', X_eval[0])
    print('X_eval.shape=', X_eval.shape)

    #     print('Y_eval[0]=', Y_eval[0])g;
    print('Y_eval.shape=', Y_eval.shape)

    chckpt = tf.keras.callbacks.ModelCheckpoint(f'./model2_f{fold}.hdf5', monitor='val_loss', mode='min',
                                                save_best_only=True)

    K.clear_session()
    #     model = build_v1()

    print('-----------\n')
    model2.fit(X_eval, Y_eval,epochs=10,batch_size=512,validation_data=(X_eval,Y_eval))
model2.summary()
model2.save('my_model2.h5')  # creates a HDF5 file 'model.h5'

