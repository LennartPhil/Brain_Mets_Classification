import tensorflow as tf
import keras_tuner as kt

def build_hp_model(hp):

    n_conv_levels = hp.Int("n_conv_levels", min_value=1, max_value=5, default=3)
    n_kernel_size = hp.Int("n_kernel_size", min_value=2, max_value=7, default=3)
    n_filters = hp.Int("n_filters", min_value=32, max_value=256, default=64, step=32)
    n_pooling = hp.Int("n_pooling", min_value=1, max_value=4, default=2)
    n_strides = hp.Int("n_strides", min_value=1, max_value=4, default=1)
    n_img_dense_layers = hp.Int("n_img_dense_layers", min_value=1, max_value=3, default=2)
    n_img_dense_neurons = hp.Int("n_img_dense_neurons", min_value=32, max_value=200, default=64)
    n_end_dense_layers = hp.Int("n_end_dense_layers", min_value=1, max_value=3, default=2)
    n_end_dense_neurons = hp.Int("n_end_dense_neurons", min_value=32, max_value=200, default=64)
    img_dropout = hp.Boolean("img_dropout")
    end_dropout = hp.Boolean("end_dropout")
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, default=0.3)
    activation = hp.Choice("activation", values=["relu", "mish"], default="relu")
    learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-1, sampling="log")
    optimizer = hp.Choice("optimizer", values=["adam", "sgd"], default="adam")

    if optimizer == "adam":
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    
    # Define inputs
    image_input = tf.keras.layers.Input(shape=(155, 240, 240, 4))
    sex_input = tf.keras.layers.Input(shape=(2,))
    age_input = tf.keras.layers.Input(shape=(1,))

    x = tf.keras.layers.BatchNormalization()(image_input)

    for _ in range(n_conv_levels):
        x = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=n_kernel_size, strides=n_strides, activation=activation)(x)
        x = tf.keras.layers.MaxPool3D(pool_size=n_pooling)(x)

    x = tf.keras.layers.Flatten()(x)
    for _ in range(n_img_dense_layers):
        x = tf.keras.layers.Dense(n_img_dense_neurons, activation=activation)(x)
        if img_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    flattened_sex_input = tf.keras.layers.Flatten()(sex_input)
    age_input_reshaped = tf.keras.layers.Reshape((1,))(age_input)
    x = tf.keras.layers.Concatenate()([x, age_input_reshaped, flattened_sex_input])

    for _ in range(n_end_dense_layers):
        x = tf.keras.layers.Dense(n_end_dense_neurons, activation=activation)(x)
        if end_dropout:
            x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(1)(x)
    output = tf.keras.layers.Activation('sigmoid', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs=[image_input, sex_input, age_input], outputs=output)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", "RootMeanSquaredError", "AUC"])

    return model

    

