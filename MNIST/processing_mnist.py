from tqdm import tqdm
import requests
import gzip
import numpy as np
import pickle
import os

'''
                         MNIST PROCESSING FILE
                
* MNIST data are downloaded from http://yann.lecun.com/exdb/mnist/

* The downloaded files are saved to a dataset directory

* The shape of the default file is (number of MNIST samples, 28*28 pixel or label)
    for examples:
        training image data : (60000, 784)
        test image data     : (10000, 784)
    
    If you change the argument (samples_pixel) of mnist_load function into "False",
        the shape becomes (28*28 pixel or label, number of MNIST samples).
            for example: mnist_load(samples_pixel=False) --> training image data : (784, 60000)

* Reference: https://github.com/oreilly-japan/deep-learning-from-scratch

'''

url_base = 'http://yann.lecun.com/exdb/mnist/'
mnist_file = {
    'train_img'  : 'train-images-idx3-ubyte.gz',
    'test_img'   : 't10k-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_label' : 't10k-labels-idx1-ubyte.gz',
}
file_dir = 'dataset/'
save_pkl_file = 'mnist_pickle.pkl'
chunk_size = 1024


def file_checker():
    os.makedirs(file_dir, exist_ok=True)
    for file_name in mnist_file.values():
        if not os.path.exists(file_dir + file_name):
            download(file_name)
    print('The MNIST datasets was downloaded !!')

    if not os.path.exists(save_pkl_file):
        data = {}
        for key in mnist_file:
            if 'img' in key:
                data[key] = file_read(mnist_file[key], offset=16)
            if 'label' in key:
                data[key] = file_read(mnist_file[key], offset=8)
        create_pickle_file(data)


def download(file_name):
    req = requests.get(url_base + file_name, stream=True)
    file_size = int(req.headers.get('Content-Length'))

    with tqdm(total=file_size, unit='B', desc=file_name, unit_scale=True) as qbar:
        with open(file_dir + file_name, 'wb') as f:
            for a in req.iter_content(chunk_size=chunk_size):
                f.write(a)
                qbar.update(chunk_size)


def file_read(file_name, offset):
    with gzip.open(file_dir + file_name, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='uint8', offset=offset)

    return data


def create_pickle_file(data):
    for key in data:
        if 'img' in key:
            data[key] = data[key].reshape(-1, 784)
        with open(file_dir + save_pkl_file, 'wb') as f:
            pickle.dump(data, f)

    print(save_pkl_file + ' was created.')
    print('These were saved to a dataset directory.\n')


def change_one_hot_label(label):
    one_hot = np.zeros((10, label.size))
    for i, row in enumerate(one_hot):
        row[label[i]] = 1

    return one_hot


def mnist_load(normalized=True, flatten=True, one_hot=False, samples_pixel=True):
    '''
    Loading MNIST data

    :param normalized: if "True", the pixel of the image is normalized.
    :param flatten: if "True", the image becomes one-dimensional array
    :param one_hot: if "True", the label becomes one-hot vector
    :param samples_pixel: if "True", the shape of the array is (number of samples, pixel or label)
                          if "False", the shape is (pixel or label, number of samples)

    :return: (training-image, training-label), (test-image, test-label)
    '''

    if not os.path.exists(file_dir + save_pkl_file):
        file_checker()

    with open(file_dir + save_pkl_file, 'rb') as f:
        data = pickle.load(f)

    if normalized:
        for key in ('train_img', 'test_img'):
            data[key] = data[key].astype(np.float32)
            data[key] /= 255.0

    if not flatten:
        for key in ('train_img', 'test_img'):
            data[key] = data[key].reshape(-1, 1, 28, 28)

    if one_hot:
        for key in ('train_label', 'test_label'):
            data[key] = change_one_hot_label(data[key])

    if not samples_pixel:
        data['train_img'] = data['train_img'].T
        data['train_label'] = data['train_label'].T
        data['test_img'] = data['test_img'].T
        data['test_label'] = data['test_label'].T

    return (data['train_img'], data['train_label']), (data['test_img'], data['test_label'])


if __name__ == '__main__':
    file_checker()
