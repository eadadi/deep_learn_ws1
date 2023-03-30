import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    DS_FOLDER = "cifar-10-batches-py"
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    '''
    This receives array of size (n x 3072)
    and transform it to (n, 32, 32, 3)
    which is the form of 32x32 rgb matrice
    '''
    def prepare_to_visual(images):
        return images.reshape(len(images),3,32,32).transpose(0,2,3,1).astype("uint8")

    def show_img(image):
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()

    def load_all_data_batches():
        prefix = 'data_batch_'
        data_batches_array = []
        for i in range(1,6):
            data_batches_array.append(DataHandler.unpickle(f"{DataHandler.DS_FOLDER}/{prefix+str(i)}"))
        data_batches_array.append(DataHandler.unpickle(f"{DataHandler.DS_FOLDER}/test_batch"))
        return data_batches_array

    def sample_and_normalize_n_from_data_batch(batch, n):
        l = len(batch[b'data'][0])
        idxes = np.random.randint(l, size=n)
        X = batch[b'data'][idxes,:]
        #Normalize X to (0,1) range
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        Y = np.array(batch[b'labels'])
        Y = Y[idxes,]
        return (X,Y)

    def sample_n_from_batches(batches, n):
        N = len(batches)
        res = []
        while N>=2:
            rn = np.random.randint(n)
            n -= rn
            to_add = DataHandler.sample_and_normalize_n_from_data_batch(batches[N-1], rn)
            if to_add:
                res.append(to_add)
            N -= 1
        if n > 0:
            res.append(DataHandler.sample_and_normalize_n_from_data_batch(batches[0], n))
        X, Y = res[0][0], res[0][1]
        if len(res) == 0:
            return (X,Y)
        for batch in res[1:]:
            X = np.concatenate((X, batch[0]))
            Y = np.append(Y, batch[1])
        return (X,Y)

    def load_and_sample_data(training = 5000, test = 1000):
        batches = DataHandler.load_all_data_batches()
        training_batches = batches[:-1]
        test_batches = np.array([batches[-1]])
        sampled_training = DataHandler.sample_n_from_batches(training_batches, training)
        sampled_test = DataHandler.sample_n_from_batches(test_batches, test)
        return (sampled_training, sampled_test)

