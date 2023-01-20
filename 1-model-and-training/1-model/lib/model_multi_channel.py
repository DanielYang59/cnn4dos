import tensorflow as tf


def cnn_for_dos(input_shape, drop_out_rate):

    def branch(branch_input, shape, drop_out_rate):
        """Each branch of the CNN network.

        Args:
            branch_input: input of each branch
            shape (int): length of each branch
        
        Notes:
            expecting (None, 4000, channels) input
            
        """

        # Reshape (None, 4000, 6) to (None, 4000, 1, 6)
        branch_input = tf.keras.layers.Reshape(target_shape=(4000, 1, 6))(branch_input)
        
        # 1st Conv layer
        conv_1 = tf.keras.layers.Conv2D(16, (3, 1), activation="relu", padding="same")(branch_input) 
        conv_1 = tf.keras.layers.Conv2D(16, (3, 1), activation="relu", padding="same")(conv_1)
        conv_1 = tf.keras.layers.Conv2D(16, (8, 1), (2, 1), activation="relu", padding="same")(conv_1)
        conv_1 = tf.keras.layers.Dropout(drop_out_rate)(conv_1)


        # 2nd Conv layer
        conv_2 = tf.keras.layers.Conv2D(16, (3, 1), activation="relu", padding="same")(conv_1)
        conv_2 = tf.keras.layers.Conv2D(16, (3, 1), activation="relu", padding="same")(conv_2)
        conv_2 = tf.keras.layers.Conv2D(16, (8, 1), (2, 1), activation="relu", padding="same")(conv_2)
        conv_2 = tf.keras.layers.Dropout(drop_out_rate)(conv_2)


        # 3rd Conv layer
        conv_3 = tf.keras.layers.Conv2D(16, (3, 1), activation="relu", padding="same")(conv_2)
        conv_3 = tf.keras.layers.Conv2D(16, (3, 1), activation="relu", padding="same")(conv_3)
        conv_3 = tf.keras.layers.Conv2D(16, (8, 1), (2, 1), activation="relu", padding="same")(conv_3)
        conv_output = tf.keras.layers.Dropout(drop_out_rate)(conv_3)


        ## Flatten and dense
        conv_flat = tf.keras.layers.Flatten()(conv_output)

        # Branch output
        branch_output = tf.keras.layers.Dense(64, activation="relu")(conv_flat)
        branch_output = tf.keras.layers.Dense(64, activation="relu")(branch_output)
        branch_output = tf.keras.layers.Dense(1)(branch_output)
        
        return branch_output

    # Master input layer
    master_input = tf.keras.Input(shape=input_shape, name="master_input")
    
    # Assign input and get output for each branch
    branch_shape = input_shape[0]
    branch_output_0 = branch((master_input[:, :, 0]), shape=branch_shape, drop_out_rate=drop_out_rate)
    branch_output_1 = branch((master_input[:, :, 1]), shape=branch_shape, drop_out_rate=drop_out_rate)
    branch_output_2 = branch((master_input[:, :, 2]), shape=branch_shape, drop_out_rate=drop_out_rate)
    branch_output_3 = branch((master_input[:, :, 3]), shape=branch_shape, drop_out_rate=drop_out_rate)
    branch_output_4 = branch((master_input[:, :, 4]), shape=branch_shape, drop_out_rate=drop_out_rate)
    branch_output_5 = branch((master_input[:, :, 5]), shape=branch_shape, drop_out_rate=drop_out_rate)
    branch_output_6 = branch((master_input[:, :, 6]), shape=branch_shape, drop_out_rate=drop_out_rate)
    branch_output_7 = branch((master_input[:, :, 7]), shape=branch_shape, drop_out_rate=drop_out_rate)
    branch_output_8 = branch((master_input[:, :, 8]), shape=branch_shape, drop_out_rate=drop_out_rate)


    # Concatenate branch outputs
    concat_output = tf.keras.layers.Concatenate(axis=-1)([branch_output_0, branch_output_1, branch_output_2, branch_output_3, branch_output_4, branch_output_5, branch_output_6, branch_output_7, branch_output_8])


    # Master output layer
    master_flat = tf.keras.layers.Flatten()(concat_output) 
    master_output = tf.keras.layers.Dense(32)(master_flat)
    master_output = tf.keras.layers.Dense(16)(master_output) 
    master_output = tf.keras.layers.Dense(1)(master_output)

    return tf.keras.Model(inputs=master_input,
                           outputs=master_output,
                           name="CNN4DOS",
                           )
