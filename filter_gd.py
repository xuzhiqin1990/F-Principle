import os
import ml_collections

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def get_data(config):
    data = tf.keras.datasets.cifar10

    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    train_images_std = train_images / 255.0
    # test_images_std = test_images / 255.0
    train_labels0 = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    # test_labels0 = keras.utils.to_categorical(test_labels, num_classes=10)
    train_images_align = np.reshape(
        train_images_std, [train_images_std.shape[0], -1])

    idx_train = np.random.choice(50000, config.train_size, replace=False)
    train_images = train_images_std[idx_train]
    train_labels = train_labels0[idx_train]
    train_images_align = train_images_align[idx_train]

    # idx_test = np.random.choice(10000, config.test_size, replace=False)
    # test_images = test_images_std[idx_test]
    # test_labels = test_labels0[idx_test]
    return train_images, train_labels, train_images_align  # , test_images, test_labels


def compute_distance(x):
    dist = -2 * np.dot(x, x.T) + np.sum(x**2, axis=1) + \
        np.sum(x**2, axis=1)[:, np.newaxis]
    return dist


def normal_kernel(dist, filter_dict):
    kernel_dict = []
    for filter in filter_dict:
        kernel = np.exp(-dist / 2 / filter)
        mean = np.sum(kernel, axis=1, keepdims=True)
        kernel_dict.append(kernel/mean)
    return kernel_dict


def gauss_filiter(f_orig, kernel):
    return np.matmul(kernel, f_orig)


def get_freq_low_high(yy, kernel_dict):
    f_low = []
    f_high = []
    # diff_fil = []
    for filter in range(len(kernel_dict)):
        kernel = kernel_dict[filter]
        f_new_norm = gauss_filiter(yy, kernel)
        f_low.append(f_new_norm)
        # tmp_diff = np.mean(np.square(yy - f_new_norm))
        # diff_fil.append(tmp_diff)
        f_high_tmp = yy - f_new_norm
        f_high.append(f_high_tmp)

    return f_low, f_high  # , diff_fil


def build_model(config):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, 3))
    model.add(tf.keras.layers.Activation(config.actfun))
    model.add(tf.keras.layers.Conv2D(64, 3))
    model.add(tf.keras.layers.Activation(config.actfun))
    model.add(tf.keras.layers.MaxPool2D(2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(400))
    model.add(tf.keras.layers.Activation(config.actfun))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Softmax())
    return model


def main():
    config_dict = {
        "filter_start": 2,
        "filter_end": 100,
        "filter_num": 20,
        "train_size": 5000,
        "test_size": 500,
        "actfun": 'relu',
        "batch_size": 50,
        "lr": 0.001,
        "lossfunc": 'CategoricalCrossentropy',
        "epochs": 100,
        "result_dir": 'results/filter_gd'
    }
    config = ml_collections.ConfigDict(config_dict)

    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    filter_dict = np.linspace(
        config.filter_start, config.filter_end, num=config.filter_num)

    train_images, train_labels, train_images_align = get_data(config)

    dist = compute_distance(train_images_align)
    kernel_dict = normal_kernel(dist, filter_dict)
    f_low, f_high = get_freq_low_high(train_labels, kernel_dict)

    model = build_model(config)
    sgd = tf.keras.optimizers.SGD(
        lr=config.lr, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=config.lossfunc, optimizer=sgd, metrics=['accuracy'])

    loss = []
    acc = []
    # valloss = []
    # valacc = []
    lowdiff = [[] for _ in range(len(filter_dict))]
    highdiff = [[] for _ in range(len(filter_dict))]
    for i in range(config.epochs):
        print('Epoch {}/{}'.format(i, config.epochs))
        his = model.fit(train_images, train_labels, batch_size=config.batch_size,
                        epochs=1, shuffle=True)
        loss.append(his.history['loss'])
        acc.append(his.history['accuracy'])
        # valloss.append(his.history['val_loss'])
        # valacc.append(his.history['val_accuracy'])
        y_pred = model.predict(train_images, batch_size=config.batch_size)

        f_train_low, f_train_high = get_freq_low_high(
            y_pred, kernel_dict)
        for i in range(len(filter_dict)):
            lowdiff[i].append(np.linalg.norm(
                f_train_low[i] - f_low[i])/np.linalg.norm(f_low[i]))
            highdiff[i].append(np.linalg.norm(
                f_train_high[i] - f_high[i])/np.linalg.norm(f_high[i]))

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(loss, 'r-', label='train')
    # plt.plot(valloss, 'b-', label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(122)
    plt.semilogy(loss, label='train')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(config.result_dir + '/loss.png')
    plt.close()

    plt.figure()
    plt.plot(acc, 'r-', label='train')
    # plt.plot(valacc, 'b-', label='test')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(config.result_dir + '/acc.png')
    plt.close()

    for ff, filter in enumerate(filter_dict):
        plt.figure(figsize=(10, 4))
        plt.title('freq with filter {:.02f}'.format(filter))
        plt.subplot(121)
        plt.plot(lowdiff[ff], 'r-', label='low_{:.02f}'.format(filter))
        plt.plot(highdiff[ff], 'b-', label='high_{:0.2f}'.format(filter))
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('freq')
        plt.subplot(122)
        tmp = np.stack([lowdiff[i], highdiff[i]])
        plt.pcolor(tmp, cmap='RdBu', vmin=0.1, vmax=1)
        plt.colorbar()
        plt.yticks([0.6, 1.6], ('low freq', 'high freq'), rotation='vertical')
        plt.xlabel('epoch')
        plt.savefig(config.result_dir + '/hot_{:0.2f}.png'.format(filter))
        plt.close()


if __name__ == '__main__':
    main()
