tensorflow-wavelets is an implementation of
- *Discrete Wavelets Transform Layer*
- *Duel Tree Complex Wavelets Transform Layer*
- *Multi Wavelets Transform Neural Networks Layer*


## Installation

```
pip install tensorflow-wavelets
```
# Usage
```
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.DMWT as DMWT
```

# Example
```
model = keras.Sequential()
model.add(keras.Input(shape=(28, 28, 1)))

model.add(DWT.DWT(name="haar",concat=0))
# name can be found in pywt.wavelist(family)
# concat=0 means to split to 4 smaller layers
# concat=1 will output 1 big layer - concat from 4 smaller layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(nb_classes, activation="softmax"))
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dwt_9_haar (DWT)             (None, 14, 14, 4)         0         
    _________________________________________________________________
    flatten_9 (Flatten)          (None, 784)               0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 10)                7850      
    =================================================================
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    _________________________________________________________________
	
	
**Free Software, Hell Yeah!**
