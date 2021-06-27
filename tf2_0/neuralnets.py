import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0  # flatten the 2 dim
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0  # flatten the 2 dim

# x_train = tf.convert_to_tensor(x_train) # this will be automatics

print(x_train.shape)
print(x_test.shape)

# model = keras.Sequential(
#     [
#         keras.Input(shape=(28*28)),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(10),
#     ]
# )
#
# print(model.summary())
#
model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu', name='my_layer'))
model.add(layers.Dense(10))

# model = keras.Model(inputs=model.inputs, outputs=[model.get_layer('my_layer').output])
model = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
print(model.summary())


features = model.predict(x_train)
for feature in features:
    print(feature.shape)
# inputs = keras.Input(shape=28*28)
# x = layers.Dense(512, activation='relu')(inputs)
# x = layers.Dense(256, activation='relu')(x)
# output = layers.Dense(10, activation='softmax')(x)
# model = keras.Model(inputs=inputs, outputs=output)




# model.compile(
#     # if from_logics = false then softmax is defined in the output model
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # send to softmax and map to sparse categorical cross entropy
#     optimizer=keras.optimizers.Adam(lr=0.001),
#     metrics=["accuracy"],
# )
#
# model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
# model.evaluate(x_test, y_test, batch_size=32, verbose=2)