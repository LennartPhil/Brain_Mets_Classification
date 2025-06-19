import tensorflow as tf

kernel_initializer = "he_normal"
activation_func = "mish"

def save_paths_to_txt(paths, type, path_to_callbacks):
    f = open(f"{path_to_callbacks}/{type}.txt", "w")

    # get patient id from path
    paths = [path.split("/")[-1] for path in paths]

    for path in paths:
        f.write(f"{path}\n")

    f.close()

    print(f"Saved {type} paths to txt file")

def __initial_conv_block(input, weight_decay = 5e-4):
    ''' Adds an initial convolution block, with batch normalization and relu activation
    Args:
        input: input tensor
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''

    x = tf.keras.layers.Conv3D(filters = 64,
                               kernel_size = 3,
                               padding = "same",
                               kernel_initializer = kernel_initializer,
                               kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)

    return x

def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay = 5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = tf.keras.layers.Conv3D(filters = grouped_channels,
                                   kernel_size = 3,
                                   padding = "same",
                                   use_bias = False,
                                   strides = (strides, strides, strides),
                                   kernel_initializer = kernel_initializer,
                                   kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation_func)(x)

        return x
    
    # cardinality loop
    for c in range(cardinality):
        x = tf.keras.layers.Lambda(lambda x: x[:, :, :, :, c * grouped_channels : (c + 1) * grouped_channels])(input)

        x = tf.keras.layers.Conv3D(filters = grouped_channels,
                                   kernel_size = 3,
                                   padding = "same",
                                   use_bias = False,
                                   strides = (strides, strides, strides),
                                   kernel_initializer = kernel_initializer,
                                   kernel_regularizer = tf.keras.regularizers.l2(weight_decay))(x)
        
        group_list.append(x)
    
    group_merge = tf.keras.layers.Concatenate(axis=-1)(group_list)
    x = tf.keras.layers.BatchNormalization()(group_merge)
    x = tf.keras.layers.Activation(activation_func)(x)

    return x


def __bottleneck_block(input, filters, cardinality, strides, weight_decay):
    init = input

    # Determine if the shortcut path needs a convolution for matching dimensions
    needs_conv = strides > 1 or input.shape[-1] != filters * 2

    grouped_channels = filters // cardinality
    
    if needs_conv:
        # Apply convolution to shortcut path to match the main path's dimensions
        init = tf.keras.layers.Conv3D(filters * 2, 1, strides=strides, padding="same", use_bias=False,
                                      kernel_initializer=kernel_initializer,
                                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(init)
        init = tf.keras.layers.BatchNormalization()(init)

    # Main path
    x = tf.keras.layers.Conv3D(filters, 1, padding="same", use_bias=False,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation_func)(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = tf.keras.layers.Conv3D(filters * 2, 1, padding="same", use_bias=False,
                               kernel_initializer=kernel_initializer,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Addition - ensuring init and x have compatible shapes
    x = tf.keras.layers.Add()([init, x])
    x = tf.keras.layers.Activation(activation_func)(x)

    return x
 

def create_res_next(img_input, depth = 29, cardinality = 8, width = 4,
                      weight_decay = 5e-4, pooling = None):
    ''' Creates a ResNeXt model with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        depth: Depth of the network. Can be an positive integer or a list
               Compute N = (n - 2) / 9.
               For a depth of 56, n = 56, N = (56 - 2) / 9 = 6
               For a depth of 101, n = 101, N = (101 - 2) / 9 = 11
        cardinality: the size of the set of transformations.
               Increasing cardinality improves classification accuracy,
        width: Width of the network.
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''

    if type(depth) is list or type(depth) is tuple:
        # if a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        # otherwise, default to 3 blocks each of default number of group convolution blocks
        N = [(depth - 2) // 9 for _ in range(3)]
    
    filters = cardinality * width
    filters_list = []

    for _ in range(len(N)):
        filters_list.append(filters)
        filters *= 2
    
    x = __initial_conv_block(img_input, weight_decay)

    # block 1 (no pooling)
    for _ in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)
    
    N = N[1:] # remove the first block from block definition list
    filters_list = filters_list[1:] # remove the first filter from the filter list

    # block 2 to N
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 2, weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides = 1, weight_decay=weight_decay)
        
    if pooling == "avg":
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
    elif pooling == "max":
        x = tf.keras.layers.GlobalMaxPooling3D()(x)
    
    return x
