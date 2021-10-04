import numpy as np
from tensorflow.keras import losses
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot

from utils.data import *
from utils.plot import *
from utils.models import *

(x_train, y_train), (x_test, y_test) = load_mnist(remove_n_samples=0)

model = basic_dwt_idw(input_shape=(28, 28, 1), wave_name="db2", eagerly=True, soft_theshold=False)

model.compile(loss='mse',
              optimizer='adam',
              metrics=["mse"])

history = model.fit(x_train, x_train,
                    validation_split=0.2,
                    epochs=30,
                    batch_size=32,
                    verbose=2,
                    )

pyplot.plot(history.history['mse'])
pyplot.show()

plt_in = plot_n_examples(x_train, y_train)
out = model.predict(x_train)
plt_out = plot_n_examples(out, y_train)
plt_in.show()
plt_out.show()


print("Heye")