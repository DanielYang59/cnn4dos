"""Plot CNN model architecture."""

import tensorflow as tf

model_dir = "model"


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
