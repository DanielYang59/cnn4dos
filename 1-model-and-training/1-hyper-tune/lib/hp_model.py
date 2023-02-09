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
        conv_x = tf.keras.layers.Reshape(target_shape=(4000, 1, 6))(branch_input)


        # Dynamic amount of ConV blocks
        for i in range(hp_branch_numConvBlocks):
            for j in range(hp_branch_primary_numConvLayers):
                ## Primary ConV blocks
                conv_x = tf.keras.layers.Conv2D(numFilters, (hp_branch_primary_kernel_size, 1), activation="relu", padding="same")(conv_x)
            
            
            ## Ending ConV block
            conv_x = tf.keras.layers.Conv2D(numFilters, (hp_branch_ending_kernel_size, 1), (2, 1), activation="relu", padding="same")(conv_x)
            
            if hp_branch_include_pooling:
                conv_x = tf.keras.layers.AveragePooling2D(pool_size=(hp_branch_pool_size, 1), padding="same")(conv_x)
            
            conv_x = tf.keras.layers.Dropout(drop_out_rate)(conv_x)
        

        ## Flatten and dense
        conv_flat = tf.keras.layers.Flatten()(conv_x)

        # Branch output
        branch_output = tf.keras.layers.Dense(hp_branch_dense_units, activation=hp_branch_dense_activation_func)(conv_flat)
        branch_output = tf.keras.layers.Dense(hp_branch_dense_units/2, activation=hp_branch_dense_activation_func)(branch_output)
        branch_output = tf.keras.layers.Dense(1)(branch_output)
        
        return branch_output
    
    
    ############################## Hyper Tuning ##########################
    # Universal
    # hp_learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
    # hp_drop_out_rate = hp.Float("drop_out_rate", min_value=0.1, max_value=0.5)
    hp_learning_rate = hp.Fixed("hp_learning_rate", value=0.001)
    hp_drop_out_rate = hp.Fixed("hp_drop_out_rate", value=0.3)
     
     
    # Master Layer
    hp_master_1st_dense_units = hp.Choice("hp_master_1st_dense_units", [128, 256, 512])
    hp_master_2nd_dense_units = hp.Choice("hp_master_2nd_dense_units", [256, 512, 1024, 2048])
    hp_master_3rd_dense_layer = hp.Boolean("hp_master_3rd_dense_layer", default=False)
    # hp_master_activation_function = hp.Choice("hp_master_act_func", ["tanh", "relu", "sigmoid"])
    hp_master_activation_function = hp.Fixed("hp_master_activation_function", value="relu")  
    
    
    # Branch
    # hp_branch_dense_activation_func = hp.Choice("hp_branch_dense_activation_func", ["tanh", "relu", "sigmoid"]) 
    hp_branch_dense_activation_func = hp.Fixed("hp_branch_dense_activation_func", value="tanh")  
    hp_numFilters = hp.Int("hp_numFilters", min_value=8, max_value=64)
    # hp_numFilters = hp.Fixed("hp_numFilters", value=12)
    hp_branch_primary_kernel_size = hp.Int("hp_branch_primary_kernel_size", min_value=2, max_value=32, step=2)
    # hp_branch_primary_kernel_size = hp.Fixed("hp_branch_primary_kernel_size", value=22)
    hp_branch_ending_kernel_size = hp.Int("hp_branch_ending_kernel_size", min_value=2, max_value=32, step=2)
    hp_branch_primary_numConvLayers = hp.Int("hp_branch_primary_numConvLayers", min_value=1, max_value=16)
    
    
    hp_branch_numConvBlocks = hp.Int("hp_branch_numConvBlocks", min_value=1, max_value=16)
    
    
    hp_branch_include_pooling = hp.Boolean("hp_branch_include_pooling", default=False)
    if hp_branch_include_pooling:
        hp_branch_pool_size = hp.Int("hp_branch_pool_size", min_value=2, max_value=32, step=2)
        
    
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
