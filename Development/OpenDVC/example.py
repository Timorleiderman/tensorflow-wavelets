from tensorflow.keras import layers, models, Model

def create_base_cnn(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
    # put more layers if you like
    model.add(layers.Dense(128, activation="relu"))
    return model

def create_head(input_shape, name):
    model = models.Sequential(name=name)
    model.add(layers.Dense(128, activation="relu", input_shape=input_shape))
    model.add(layers.Dense(64, activation="relu"))
    # put more layers if you like
    model.add(layers.Dense(1, activation="linear"))
    return model

# Create the model.
input_shape = (240, 180, 1)
base_model = create_base_cnn(input_shape)
head_model1 = create_head((128,), name="head1")
head_model2 = create_head((128,), name="head2")
model_input = layers.Input(shape=input_shape)

# Combine base with heads (using TF's functional API)
features = base_model(model_input)
model_output1 = head_model1(features)
model_output2 = head_model2(features)
model = Model(inputs=model_input, outputs=[model_output1, model_output2])


HEAD1_WEIGHT = 0.4
HEAD2_WEIGHT = 0.6
model.compile(
    optimizer="Adam",
    loss={"head1": "mse", "head2": "mse"},
    loss_weights={"head1": HEAD1_WEIGHT, "head2": HEAD2_WEIGHT},
    metrics={"head1": ["mae"], "head2": ["mae"]}
)
model.fit(dataset_training, validation_data, epochs)