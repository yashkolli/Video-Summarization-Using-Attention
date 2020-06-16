import pprint

class Config():

    def __init__(self, **kwargs):

        self.data_path = 'C:/Users/Yashwanth/Desktop/Bennett DL Project/fcsn_tvsum.h5'
        self.save_dir = 'C:/Users/Yashwanth/Desktop/Bennett DL Project/save_dir'
        self.score_dir = 'C:/Users/Yashwanth/Desktop/Bennett DL Project/score_dir'

        self.n_epochs = 5
        self.batch_size = 5

        for a,b in kwargs.items():
            setattr(self,a,b)

    def __repr__(self):

        config_str = 'Configurations\n' + pprint.pformat(self.__dict__)

        return config_str

if __name__ == '__main__':
    config = Config()
    print(config)
