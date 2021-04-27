from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout


def cifar10CNN(input_size=(32, 32, 3), nb_classes=10):
    inputs = Input(shape=input_size)

    output = Conv2D(32, (3, 3), padding="same")(inputs)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = Conv2D(32, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = MaxPooling2D((2, 2))(output)

    output = Dropout(0.5)(output)  # added dropout to improve overfitting

    output = Conv2D(32, (3, 3), padding="same")(output)

    output = Dropout(0.25)(output)

    output = Conv2D(64, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = Conv2D(64, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = MaxPooling2D((2, 2))(output)

    output = Dropout(0.5)(output)  # added dropout to improve overfitting

    output = Conv2D(64, (3, 3), padding="same")(output)

    output = Flatten()(output)
    output = Dropout(0.25)(output)

    output = Dense(512, activation="relu")(output)
    output = Dropout(0.5)(output)

    output = Dense(nb_classes, activation="softmax")(output)

    model = Model(inputs=inputs, outputs=output)

    return model