"""Plot CNN model architecture."""

model_dir = "model"


import os

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__":
    # Load model
    model = tf.keras.models.load_model(model_dir)

    # Save model to figure
    tf.keras.utils.plot_model(
        model,
        to_file="cnn_model.png",
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
