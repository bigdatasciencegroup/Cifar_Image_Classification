import tensorflow as tf
from numpy import array
import numpy as np

#mnist = tf.keras.datasets.mnist

#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def reshapeImage(imageArray):
    r = imageArray[0:1024].reshape((1,32,32))
    g = imageArray[1024:2048].reshape((1,32,32))
    b = imageArray[2048:3073].reshape((1,32,32))
    data = np.concatenate([r,g,b])
    return data

dataNames = ["data_batch_2","data_batch_3","data_batch_4","data_batch_5"]

batchData = unpickle("cifar-10-batches-py/data_batch_1")
TImageData = batchData[b'data']
TLabelData = batchData[b'labels']

# for dataName in dataNames:
#     batchData = unpickle("cifar-10-batches-py/"+dataName)
#     imageData = batchData[b'data']
#     labelData = batchData[b'labels']
#     TImageData = np.concatenate((TImageData,imageData))
#     TLabelData = np.concatenate((TLabelData,labelData))

x_train = array(TImageData)/255.0
y_train = array(TLabelData)
x_test = array(TImageData)/255.0
y_test = array(TLabelData)
#
# print(y_train)
#
# i = 0
# reshapedData = []
# for image in x_train:
#     reshapedData.append(reshapeImage(image))
# reshapedData = array(reshapedData)
# print(reshapedData.shape)

#print(x_train.shape())
#print(y_train.shape())

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

#test_loss, test_acc = model.evaluate(x_test, y_test)
#print('Test accuracy:', test_acc)