import pprint


class Config():

    def __init__(self, **kwargs):

        self.data_path_tvsum = ''     # path to tvsum dataset (h5 file)
        self.data_path_summe = ''     # path to summe dataset (h5 file)
        self.save_dir = ''      # path to save directory
        self.score_dir_tvsum = ''     # path to tvsum score directory
        self.score_dir_summe = ''     # path to summe score directory

        self.n_epochs = 5
        self.batch_size = 5

        for a, b in kwargs.items():
            setattr(self, a, b)

    def __repr__(self):

        config_str = 'Configurations\n' + pprint.pformat(self.__dict__)

        return config_str


if __name__ == '__main__':
    config = Config()
    print(config)
