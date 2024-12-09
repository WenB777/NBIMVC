import os, sys
import numpy as np
import scipy.io as sio


def generate_outliers(data, n_outliers):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # 生成离群值
    outliers = np.random.normal(loc=mean + 3 * std, scale=std, size=(n_outliers, data.shape[1]))
    
    return outliers

def load_data(config):
    """Load data """
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []
    print("shuffle")
    if data_name in ['MNIST-USPS']:
        mat = sio.loadmat('data/MNIST-USPS.mat')
        X1 = mat['X1'].astype('float32').reshape((5000, 784))  # (5000,784)
        X2 = mat['X2'].astype('float32').reshape((5000, 784))  # (5000,784)
        Y = np.squeeze(mat['Y'])
        X_list.append(X1)
        X_list.append(X2)
        Y_list.append(Y)

        print(Y_list[0])
    elif data_name in ['BDGP']:
        mat = sio.loadmat('data/BDGP.mat')
        x2 = np.copy(mat['X2'])
        x1 = np.copy(mat['X1'])
        y = np.copy(mat['Y'].T)
        print(np.squeeze(y))
        np.random.seed(2500)
        index = [i for i in range(2500)]
        np.random.shuffle(index)
        for i in range(2500):
            x2[i] = mat['X2'][index[i]]
            x1[i] = mat['X1'][index[i]]
            y[i] = mat['Y'].T[index[i]]

        from sklearn.preprocessing import normalize
        x1 = normalize(x1, axis=1, norm='max')
        x2 = normalize(x2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        x1 = min_max_scaler.fit_transform(x1)
        x2 = min_max_scaler.fit_transform(x2)

        X_list.append(x2.astype('float32'))                 # (2500,1750)
        X_list.append(x1.astype('float32'))                 # (2500,79)
        Y_list.append(np.squeeze(y))
        print(np.squeeze(y))
    elif data_name in ['Scene-15']:
        mat = sio.loadmat('data/Scene-15.mat')
        X = mat['X'][0]
        print(np.squeeze(mat['Y']))

        x1 = X[1]
        x2 = X[0]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(mat['Y'])
        index = [i for i in range(4485)]
        np.random.seed(4485)
        np.random.shuffle(index)
        for i in range(4485):
            xx1[i] = x1[index[i]]
            xx2[i] = x2[index[i]]
            Y[i] = mat['Y'][index[i]]

        from sklearn.preprocessing import normalize
        xx1 = normalize(xx1, axis=1, norm='max')
        xx2 = normalize(xx2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        xx1 = min_max_scaler.fit_transform(xx1)
        xx2 = min_max_scaler.fit_transform(xx2)

        X_list.append(xx1)
        X_list.append(xx2)
        y = np.squeeze(Y).astype('int')
        Y_list.append(y)
        print(y)
    elif data_name in ['LandUse-21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse-21.mat'))
        X = mat['X'][0]
        print(np.squeeze(mat['Y']))

        x1 = X[1]
        x2 = X[0]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(mat['Y'])
        index = [i for i in range(2100)]
        np.random.seed(2100)
        np.random.shuffle(index)
        for i in range(2100):
            xx1[i] = x1[index[i]]
            xx2[i] = x2[index[i]]
            Y[i] = mat['Y'][index[i]]

        from sklearn.preprocessing import normalize
        xx1 = normalize(xx1, axis=1, norm='max')
        xx2 = normalize(xx2, axis=1, norm='max')
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        xx1 = min_max_scaler.fit_transform(xx1)
        xx2 = min_max_scaler.fit_transform(xx2)

        X_list.append(xx1)
        X_list.append(xx2)
        y = np.squeeze(Y).astype('int')
        Y_list.append(y)
        print(y)
    elif data_name in ['HandWritten']:
        mat = sio.loadmat('data/HandWritten.mat')
        X = mat['X'][0]
        Y = np.copy(mat['Y'])

        index = np.arange(2000)
        np.random.seed(2000)
        np.random.shuffle(index)
        from sklearn.preprocessing import normalize
        from sklearn import preprocessing

        X_list = []
        for i in range(6):
            xx = np.copy(X[i])[index]
            xx = normalize(xx, axis=1, norm='max')
            xx = preprocessing.MinMaxScaler().fit_transform(xx)
            X_list.append(xx)

        y = np.squeeze(Y[index]).astype('int')
        Y_list.append(y)
        print(y)


    elif data_name in ['Caltech101-7']:
        mat = sio.loadmat('data/Caltech101-7.mat')
        X = mat['X']
        Y = np.copy(mat['Y'])

        index = np.arange(1474)
        np.random.seed(1474)
        np.random.shuffle(index)
        from sklearn.preprocessing import normalize
        from sklearn import preprocessing

        X_list = []
        for i in range(6):
            xx = np.copy(X[i][0])[index]
            xx = normalize(xx, axis=1, norm='max')
            xx = preprocessing.MinMaxScaler().fit_transform(xx)
            X_list.append(xx)

        y = np.squeeze(Y[index]).astype('int')
        Y_list.append(y)
        print(y)


    return X_list, Y_list

