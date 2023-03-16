#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import keras_tuner
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from hp_model import hp_model


# Main Loop
if __name__ == "__main__":
    # Initiate Keras Tuner
    tuner = keras_tuner.Hyperband(
        hypermodel=hp_model,
        max_epochs=200,
        factor=3,
        overwrite=False,
        objective="val_mean_absolute_error",
        directory="hp_search",
        project_name="best_model",
        )
    
    
    # Load best model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    best_model.build(input_shape=(None, 4000, 9, 6))
    best_model.summary()
    
    # Save best model
    best_model.save("./best_model")
    