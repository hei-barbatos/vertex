import numpy as np 

class SampleDataset():
    ''' data preprocess and generate
    
    '''
    def __init__(self, data, kernel={}, mode='train', batch_size=1):
        super(SampleDataset, self).__init__()
        self.sample_dat = []
        self.mode = mode
        self.kernel = kernel
        self.__preprocess__(data)

        self.batch_size = batch_size
        self.loader = self.__data_loader__()

    def __getitem__(self, idx):
        input_, label_ = self.sample_dat[idx]    
        return input_, label_

    def __len__(self):
        return len(self.sample_dat)

    def __get_encoded__(self, en_value,  en_parse):
        ''' Embedding  
        '''
        # slot values 编码
        if en_parse in self.kernel:
            if en_value in self.kernel[en_parse]['d2e'].keys():
                return self.kernel[en_parse]['d2e'][en_value]
            else:
                self.kernel[en_parse]['d2e'][en_value] = self.kernel[en_parse]['count']
                self.kernel[en_parse]['e2d'][self.kernel[en_parse]['count']] = en_value
                self.kernel[en_parse]['count'] = self.kernel[en_parse]['count'] + 1
                return self.kernel[en_parse]['d2e'][en_value]
        else:
            self.kernel[en_parse] = {"d2e":{en_value:1, 'empty':0}, 
                                        'e2d':{1:en_value, 0:'empty'}, 'count':2}
            return 1

    def __preprocess__(self, data):
        ''' self-define process logic
        '''
        for i, only_dat in enumerate(data):
            if self.mode == 'train':
                movie, user, score = only_dat
            else:
                movie, user = only_dat

            feasign = []
            feasign += [
                # user
                self.__get_encoded__(user[0], 'uid'),
                self.__get_encoded__(user[1], 'gid'),
                self.__get_encoded__(user[2], 'aid'),
                self.__get_encoded__(user[3], 'jid'),

                # movie
                self.__get_encoded__(movie[0], 'mid'),
                self.__get_encoded__(movie[1], 'tid')
            ]    

            feasign_cate = [0] * 3
            for i, c in enumerate(movie[2][:3]):
                feasign_cate[i] = self.__get_encoded__(c, 'cid')
            feasign += feasign_cate
            
            if self.mode == "train":
                self.sample_dat.append([
                    np.array(feasign).astype("int64"), 
                    np.array([int(score)/5]).astype("float32")
                ])
            else:
                self.sample_dat.append([
                    np.array(feasign).astype("int64")
                ])
        return 

    def __data_loader__(self):
        ''' only for train data
            return generator
        '''
        def reader():
            # 训练时随机打乱数据顺序
            # if mode == 'train':
            #     random.shuffle(filenames)

            # 定义列表存储要取得数据
            batch_input_ = []
            batch_label_ = []

            for line in self.sample_dat:
                input_, label_ = line
                batch_input_.append(input_)
                batch_label_.append(label_)

                if len(batch_label_) == self.batch_size:
                    yield np.array(batch_input_), np.array(batch_label_)
                    batch_input_, batch_label_ = [], []

            if len(batch_label_) > 0:
                yield np.array(batch_input_), np.array(batch_label_)
        return reader
