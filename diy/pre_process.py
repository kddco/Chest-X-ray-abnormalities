import pandas as pd
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import KFold

import load_csv

train = load_csv.get_train()
class_names = sorted(train.class_name.unique())
del class_names[class_names.index('No finding')]
class_names = class_names+['No finding']
classes = dict(zip(list(range(15)),class_names))



def prepareDataFrame(train_df=train):
    train_df = train_df.fillna(0)
    #     train_df = train_df.head(10)

    cols = ['image_id', 'label'] + list(range(4 * len(class_names[:-1])))
    return_df = pd.DataFrame(columns=cols)

    for image in tqdm(train_df.image_id.unique()):
        #         print('image=', image)
        df = train_df.query("image_id==@image")
        #         print('df=', df)

        label = np.zeros(15)
        for cls in df.class_id.unique():
            #             print('cls=', cls)
            label[int(cls)] = 1
        #             print('label=', label)

        bboxes_df = df.groupby('class_id')[['x_min', 'y_min', 'x_max', 'y_max']].mean().round()
        #         print('bboxes_df=', bboxes_df)

        bboxes_list = [0 for i in range(60)]
        for ind in list(bboxes_df.index):
            bboxes_list[4 * ind:4 * ind + 4] = list(bboxes_df.loc[ind, :].values)
        return_df.loc[len(return_df), :] = [image] + [label] + bboxes_list[:-4]

    #         print('===========\n')

    return return_df

def generateFolds(train_df,n_splits = None):
    kf = KFold(n_splits= n_splits)
    for id,(tr_,val_) in enumerate(kf.split(train_df["image_id"],train_df["label"])):
        train_df.loc[val_,'kfold'] = int(id)
    train_df["kfold"].astype(int)


def get_train_df():
    return train_df

train_df = prepareDataFrame()

my_cols = ['image_id',    'label']
train_df = train_df[my_cols]

generateFolds(train_df, n_splits=5)

