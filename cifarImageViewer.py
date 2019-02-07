from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.image as mpimg

#def displayImage(image):
#    plt.imshow(image)
#    plt.show()

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
dataNames = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
labelNames = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","trucks"]


batchData = unpickle("cifar-10-batches-py/"+dataNames[0])


for k in range(1,10):
    imageData = batchData[b'data'][k]
    labelIndex = batchData[b'labels'][k]

    r = imageData[0:1024]
    g = imageData[1024:2048]
    b = imageData[2048:3073]

    w, h = 32, 32
    data = np.zeros((h, w, 3), dtype=np.uint8)
    idx = 0
    for i in range(32):
        for j in range(32):
            data[i][j] = [r[idx],g[idx],b[idx]]
            idx = idx+1


#    img = Image.fromarray(data, 'RGB')

    imgplot = plt.imshow(data)   
    plt.title(labelNames[labelIndex])
    plt.show()
    #img.show()
    #input()
