import numpy as np

import pre_process




class DataLoader:
    def __init__(self, path=None, train_df=pre_process.get_train_df(), val_df=None):
        self.path = path
        self.df = train_df
        self.val_df = val_df
        self.train_list = [f'{img}.npy' for img in train_df["image_id"].unique()]
        np.random.shuffle(self.train_list)
        self.test_list = [f'{img}.npy' for img in val_df["image_id"].unique()]
        np.random.shuffle(self.test_list)

    def read_image(self):
        for img in self.train_list:
            im_name = img.split('.npy')[0]
            image = np.load(self.path + img)
            temp = self.df[self.df.image_id == im_name]
            c_label, bb = temp.iloc[0, 1], temp.iloc[0, 2:].values.astype('float')
            yield image, c_label, bb

    def batch_generator(self, items, batch_size):
        a = []
        i = 0
        for item in items:
            a.append(item)
            i += 1

            if i % batch_size == 0:
                yield a
                a = []
        if len(a) is not 0:
            yield a

    def flow(self, batch_size):
        """
        flow from given directory in batches
        ==========================================
        batch_size: size of the batch
        """
        while True:
            for bat in self.batch_generator(self.read_image(), batch_size):
                batch_images = []
                batch_c_labels = []
                batch_bb = []
                for im, im_c_label, im_bb in bat:
                    batch_images.append(im)
                    batch_c_labels.append(im_c_label)
                    batch_bb.append(im_bb)
                batch_images = np.stack(batch_images, axis=0)

                #                 batch_labels =  (np.stack(batch_c_labels,axis=0),np.stack(batch_bb,axis=0))
                batch_labels = np.stack(batch_c_labels, axis=0)
                yield batch_images, batch_labels

    def getVal(self):
        images = []
        c_labels = []
        bb_labels = []
        for img in self.test_list:
            im_name = img.split('.npy')[0]
            image = np.load(self.path + img)
            temp = self.val_df[self.val_df.image_id == im_name]
            c_label, bb = temp.iloc[0, 1], temp.iloc[0, 2:].values.astype('float')
            images.append(image)
            c_labels.append(c_label)
            bb_labels.append(bb)

        #         return np.stack(images,axis=0),(np.stack(c_labels,axis=0),np.stack(bb_labels,axis=0))
        return np.stack(images, axis=0), np.stack(c_labels, axis=0)