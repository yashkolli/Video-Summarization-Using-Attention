import h5py
import numpy as np
import tensorflow.keras as Keras
from sklearn.model_selection import train_test_split

class DataGenerator(Keras.utils.Sequence):

    def __init__(self,dataset,batch_size=5,shuffle=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):

        return int(np.floor(len(self.dataset)/self.batch_size))

    def __getitem__(self,index):

        indexes = self.indices[index * self.batch_size : (index+1) * self.batch_size]
        feature, label = self.__data_generation(indexes)

        return feature, label

    def __data_generation(self,indexes):

        feature = np.empty((self.batch_size,320,1024))
        label = np.empty((self.batch_size,320,1))

        for i in range(len(indexes)):
            feature[i,] = np.array(self.dataset[indexes[i]][0])
            label[i,] = np.array(self.dataset[indexes[i]][1]).reshape(-1,1)

        return feature,label

    def on_epoch_end(self):

        self.indices = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

class DatasetMaker(object):

    def __init__(self,data_path):

        self.data_file = h5py.File(data_path)

    def __len__(self):

        return len(self.data_file)

    def __getitem__(self,index):

        index += 1
        video = self.data_file['video_'+str(index)]
        feature = np.array(video['feature'][:])
        label = np.array(video['label'][:])

        return feature,label,index

def get_loader(path, batch_size=5):

    dataset = DatasetMaker(path)
    train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2)
    train_loader = DataGenerator(train_dataset)

    return train_loader, test_dataset

if __name__ == '__main__':

    loader = get_loader('fcsn_dataset.h5')
