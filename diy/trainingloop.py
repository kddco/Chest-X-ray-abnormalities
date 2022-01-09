import tensorflow as tf
import tensorflow.keras.backend as K

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from diy.Loader import DataLoader
from diy.train import build_v1


def loop(train_df):
    for fold in range(3):
        print(f'\nFold: {fold}\n')

        #     X_train = train_df[train_df.kfold!=fold].drop('kfold',axis=1)
        #     X_val = train_df[train_df.kfold==fold].drop('kfold',axis=1)
        X_train = train_df[train_df.kfold != fold]
        X_val = train_df[train_df.kfold == fold]
        print('X_train.shape=', X_train.shape)
        print('X_train.head()=', X_train.head())

        print('-----------\n')

        print('X_val.shape=', X_val.shape)
        print('X_val.head()=', X_val.head())

        print('-----------\n')
        dl = DataLoader('d:/machinelearning_xay_chest/train/', X_train, X_val)

        train_set = dl.flow(batch_size=32)

        X_eval, Y_eval = dl.getVal()
        #     print('X_eval[0]=', X_eval[0])
        print('X_eval.shape=', X_eval.shape)

        #     print('Y_eval[0]=', Y_eval[0])
        print('Y_eval.shape=', Y_eval.shape)

        chckpt = tf.keras.callbacks.ModelCheckpoint(f'./model_f{fold}.hdf5', monitor='val_loss', mode='min',
                                                    save_best_only=True)

        K.clear_session()
        model = build_v1()

        print('-----------\n')
        model.fit(train_set,
                  epochs=10,
                  steps_per_epoch=int(15000 / 32),
                  validation_data=(X_eval, Y_eval),
                  callbacks=[chckpt]
                  )

        break
