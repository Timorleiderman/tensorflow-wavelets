

from utils.data import *
from utils.plot import *
from utils.models import *
from tensorflow.keras import losses

(x_train, y_train), (x_test, y_test) = load_mnist(remove_n_samples=0)
# model = AutocodeBasic(latent_dim=64, width=28, height=28)
model = AutocodeBasicDWT(latent_dim=64, width=28, height=28)
model.compile(optimizer='adam', loss=losses.MeanSquaredError())

model.fit(x_train, x_train,
                epochs=2,
                shuffle=True,
                validation_data=(x_test, x_test))
model.summary()
encoded_imgs = model.encoder(x_test).numpy()
decoded_imgs = model.decoder(encoded_imgs).numpy()

plot_n_examples(decoded_imgs, y_test).show()
plot_n_examples(x_test, y_test).show()

print("Hey")