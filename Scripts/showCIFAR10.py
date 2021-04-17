
# example of loading the cifar10 dataset
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

# plot first few images

f, axarr = plt.subplots(1, 4)
f.set_size_inches(16, 6)

for i in range(4):
    # define subplot
    #plt.subplot(330 + 1 + i)
    # plot raw pixel data
    #plt.imshow(trainX[i])
    axarr[i].imshow(trainX[i])
    axarr[i].title.set_text(cifar10_classes[int(trainy[i])])

# show the figure
plt.show()
