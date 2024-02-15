"""Check metrics for an already finished Keras-tuner run."""


import io
import sys

import keras_tuner
from hp_model import hp_model


# Main Loop
if __name__ == "__main__":
    # Reload Keras tuner
    tuner = keras_tuner.Hyperband(
        hypermodel=hp_model,
        max_epochs=200,
        factor=3,
        overwrite=False,
        objective="val_mean_absolute_error",
        directory="hp_search",
        project_name="best_model",
    )

    # Create a StringIO buffer to capture output
    output_buffer = io.StringIO()
    # Redirect stdout to the buffer
    original_stdout = sys.stdout
    sys.stdout = output_buffer

    # Load hyperparameter search summary
    summary = tuner.results_summary(num_trials=10000)

    # Restore the original stdout
    sys.stdout = original_stdout
    # Get the captured output as a string
    summary_string = output_buffer.getvalue()
    # Close the buffer
    output_buffer.close()

    # Parse metrics into dict
    metric_dict = {}
    trial_count = 0
    for line in summary_string.split("\n"):
        if line.strip().startswith("Score:"):
            trial_count += 1
            print(f"Trial No. {trial_count}, {line}.")
