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
model.add(keras.Input(shape=input_shape))

model.add(DWT.DWT(name="haar",concat=0))
# name can be found in pywt.wavelist(family)
# concat=0 means to split to 4 smaller layers
# concat=1 will output 1 big layer - concat from 4 smaller layers
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(nb_classes, activation="softmax"))
model.summary()

```
**Free Software, Hell Yeah!**