from keras.datasets import mnist
import numpy as np
import random
import keras
from keras import backend as K
import cv2


#1:same, 0:different
def create_pairs(x, digit_indices): 
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    print('n:',n) 
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10) #1..9
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def contrastive_loss(y_true, y_pred): 
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def create_base_network(input_shape):
    input = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(6,(5,5),activation = 'relu', padding = 'valid')(input)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.MaxPooling2D(2,2)(x)
    x = keras.layers.Conv2D(16,(5,5),activation = 'relu', padding = 'valid')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.MaxPooling2D(2,2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    # x = keras.layers.Dropout(0.1)(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    # x = keras.layers.Dropout(0.1)(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    return keras.models.Model(input, x)


def euclidean_distance(vects): 
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    print(shape1[0])
    return (shape1[0], 1)




def train_model():
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images.shape = (60000, 28, 28)
    training_labels.shape = (60000, )
    test_images.shape = (10000, 28, 28)
    test_labels.shape = (10000, )
 
    training_images = training_images.astype('float32')/255
    test_images = test_images.astype('float32')/255
    input_shape = (28,28,1)

    digit_indices = [np.where(training_labels == i)[0] for i in range(10)] 

    tr_pairs, tr_y = create_pairs(training_images, digit_indices)
    print(tr_y, tr_y.shape)  
    # [1 0 1 ... 0 1 0] (108400,)

    digit_indices = [np.where(test_labels == i)[0] for i in range(10)]
    te_pairs, te_y = create_pairs(test_images, digit_indices)
    # n:891
    # print(te_pairs.shape) = (17820, 2, 28, 28)
    # print(te_y.shape) = (17820,)
    base_network = create_base_network(input_shape)
    base_network.summary()

    input_a = keras.Input(shape=input_shape)
    input_b = keras.Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = keras.layers.Lambda(euclidean_distance, 
                               output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = keras.models.Model([input_a, input_b], distance)
    model.summary()
    model.compile(loss=contrastive_loss , optimizer=keras.optimizers.RMSprop(), metrics=[accuracy])
    model.fit([tr_pairs[:, 0].reshape((108400,28,28,1)), tr_pairs[:, 1].reshape((108400,28,28,1))], tr_y,
          batch_size=128,
          epochs=10)

    model.save('./siamesenet.h5')
    print('-----------------------save successfully--------------')




def main():
    train_model()

if __name__ == '__main__':
    main()



