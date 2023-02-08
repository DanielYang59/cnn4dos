#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import keras_tuner


def hp_model(hp, input_shape=(4000, 9, 6)):
    
    def branch(branch_input, drop_out_rate, numFilters=16):
        """Each branch of the CNN network.

        Args:
            branch_input: input of each branch
        
        Notes:
            expecting (batch_size, 4000, numOrbitals, numChannels) input
            
        """

        # Reshape (None, 4000, 6) to (None, 4000, 1, 6)
        branch_input = tf.keras.layers.Reshape(target_shape=(4000, 1, 6))(branch_input)
        
        # 1st Conv layer
        conv_x = tf.keras.layers.Conv2D(numFilters, (hp_branch_kernel_size, 1), activation="relu", padding="same")(branch_input) 
        conv_x = tf.keras.layers.Conv2D(numFilters, (hp_branch_kernel_size, 1), activation="relu", padding="same")(conv_x)
        conv_x = tf.keras.layers.Conv2D(numFilters, (8, 1), (2, 1), activation="relu", padding="same")(conv_x)
        conv_x = tf.keras.layers.Dropout(drop_out_rate)(conv_x)


        # 2nd Conv layer
        conv_x = tf.keras.layers.Conv2D(numFilters, (hp_branch_kernel_size, 1), activation="relu", padding="same")(conv_x)
        conv_x = tf.keras.layers.Conv2D(numFilters, (hp_branch_kernel_size, 1), activation="relu", padding="same")(conv_x)
        conv_x = tf.keras.layers.Conv2D(numFilters, (8, 1), (2, 1), activation="relu", padding="same")(conv_x)
        conv_x = tf.keras.layers.Dropout(drop_out_rate)(conv_x)


        # 3rd Conv layer
        conv_x = tf.keras.layers.Conv2D(numFilters, (hp_branch_kernel_size, 1), activation="relu", padding="same")(conv_x)
        conv_x = tf.keras.layers.Conv2D(numFilters, (hp_branch_kernel_size, 1), activation="relu", padding="same")(conv_x)
        conv_x = tf.keras.layers.Conv2D(numFilters, (8, 1), (2, 1), activation="relu", padding="same")(conv_x)
        conv_x = tf.keras.layers.Dropout(drop_out_rate)(conv_x)


        ## Flatten and dense
        conv_flat = tf.keras.layers.Flatten()(conv_x)

        # Branch output
        branch_output = tf.keras.layers.Dense(hp_branch_dense_units, activation="relu")(conv_flat)
        branch_output = tf.keras.layers.Dense(hp_branch_dense_units/2, activation="relu")(branch_output)
        branch_output = tf.keras.layers.Dense(1)(branch_output)
        
        return branch_output
    
    
    ############################## Hyper Tuning ##########################
    # Universal
    # hp_learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    # hp_drop_out_rate = hp.Float("drop_out_rate", min_value=0.1, max_value=0.5)
    hp_learning_rate = hp.Fixed("hp_learning_rate", value=0.001)
    hp_drop_out_rate = hp.Fixed("hp_drop_out_rate", value=0.3) 
    
     
    # Master Layer
    hp_master_1st_dense_units = hp.Choice("hp_master_1st_dense_units", [64, 128, 256, 512, 1024])
    hp_master_2nd_dense_units = hp.Choice("hp_master_2nd_dense_units", [64, 128, 256, 512, 1024])
     
    hp_master_3rd_dense_layer = hp.Boolean("hp_master_3rd_dense_layer", default=False)
    # hp_master_activation_function = hp.Choice("hp_master_act_func", ["linear", "relu"])  # relu seems to work better than linear
    hp_master_activation_function = hp.Fixed("hp_master_activation_function", value="relu")  
    
    
    # Branch
    hp_numFilters = hp.Int("hp_numFilters", min_value=2, max_value=128, sampling="log")
    hp_branch_kernel_size = hp.Int("hp_branch_kernel_size", min_value=2, max_value=32, step=2) 
    hp_branch_dense_units = hp.Choice("hp_branch_dense_units", [16, 32, 64, 128, 256, 512])
    
    
    ############################## Hyper Tuning Ends #####################
    
    
    # Master input layer
    master_input = tf.keras.Input(shape=input_shape, name="master_input")
    
    # Assign input and get output for each branch
    # branch_shape = input_shape[0]
    branch_output_0 = branch((master_input[:, :, 0]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)
    branch_output_1 = branch((master_input[:, :, 1]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)
    branch_output_2 = branch((master_input[:, :, 2]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)
    branch_output_3 = branch((master_input[:, :, 3]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)
    branch_output_4 = branch((master_input[:, :, 4]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)
    branch_output_5 = branch((master_input[:, :, 5]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)
    branch_output_6 = branch((master_input[:, :, 6]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)
    branch_output_7 = branch((master_input[:, :, 7]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)
    branch_output_8 = branch((master_input[:, :, 8]), drop_out_rate=hp_drop_out_rate, numFilters=hp_numFilters)


    # Concatenate branch outputs
    concat_output = tf.keras.layers.Concatenate(axis=-1)([branch_output_0, branch_output_1, branch_output_2, branch_output_3, branch_output_4, branch_output_5, branch_output_6, branch_output_7, branch_output_8])


    # Master output layer
    master_flat = tf.keras.layers.Flatten()(concat_output) 
    master_output = tf.keras.layers.Dense(hp_master_1st_dense_units, activation=hp_master_activation_function)(master_flat)
    master_output = tf.keras.layers.Dense(hp_master_2nd_dense_units, activation=hp_master_activation_function)(master_output)
    
    if hp_master_3rd_dense_layer:
       master_output = tf.keras.layers.Dense(hp_master_2nd_dense_units/4, activation=hp_master_activation_function)(master_output)
        
    
    master_output = tf.keras.layers.Dense(1)(master_output)





    # Assemble model
    model = tf.keras.Model(inputs=master_input,
                           outputs=master_output,
                           name="CNN4DOS_hypermodel",
                           )
    
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate),
        # optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.mean_absolute_error, ],
    )
    
    return model
