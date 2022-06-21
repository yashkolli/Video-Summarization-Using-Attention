import h5py
import numpy as np
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split


class Sequence_Generator(Sequence):

    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataset)/self.batch_size))

    def __getitem__(self, idx):
        batch = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        X, dX, sdata, y = self.__data_generation(batch)

        # return X, y     # for model - 1
        return [X, sdata], y    # for model - 2, 4, 5
        # return [X, dX], y   # for model - 3

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch):
        X = np.array([self.dataset[video][0] for video in batch])
        dX = np.empty((self.batch_size, 320, 1))
        n = len(batch)
        sdata = np.ones((n, 1, 1))*(2)
        for i in range(n):
            dX[i, 0, ] = np.ones((1)) * (2)
            dX[i, 1:, ] = np.array(self.dataset[batch[i]][1])[
                :-1].reshape(-1, 1)
        y = np.array([self.dataset[video][1].reshape(320, 1)
                     for video in batch], dtype="float32")

        return X, dX, sdata, y


class Dataset_Maker():

    def __init__(self, data_path):
        self.data_file = h5py.File(data_path)

    def __len__(self):
        return len(self.data_file)

    def __getitem__(self, index):
        index += 1
        video = self.data_file['video_'+str(index)]
        feature = np.array(video['feature'][:])
        label = np.array(video['label'][:])

        return feature, label, index


def get_loader(path, batch_size):
    dataset = Dataset_Maker(path)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_sequence = Sequence_Generator(train_dataset, batch_size)

    return train_sequence, test_dataset


if __name__ == '__main__':

    loader = get_loader('tvsum.h5')
