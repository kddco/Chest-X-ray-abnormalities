from keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
import pre_process
from diy.Loader import DataLoader

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


model2 = build_v2()
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()
model2.save('my_model2.h5')  # creates a HDF5 file 'model.h5'
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
    dl = DataLoader('D:/machinelearning_xay_chest/download_from_kaggle/train/',X_train,X_val)
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
    model2.fit(train_set,
               epochs=10,
               steps_per_epoch=int(15000 / 32),
               validation_data=(X_eval, Y_eval),
               callbacks=[chckpt]
               )

    break

