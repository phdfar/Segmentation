from tensorflow.keras import layers
from tensorflow import keras
from keras.models import load_model


def network(args):
  if args.network=='instance_1':
    return inception_default(args.imagesize,args.num_class,args.channel_input,args.network)
  if args.network=='instance_2':
    return inception_default(args.imagesize,args.num_class,args.channel_input,args.network)
  if args.network=='instance_3':
    return inception_default(args.imagesize,args.num_class,args.channel_input,args.network)

def inception_default(img_size, num_classes,channel_input,network):
    inputs = keras.Input(shape=img_size + (channel_input,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    #x = layers.Conv2D(16, 3, activation="softmax", padding="same")(x)
    #x = layers.Conv2D(8, 3, activation="softmax", padding="same")(x)
    #x = layers.Conv2D(4, 3, activation="softmax", padding="same")(x)

    # Add a per-pixel classification layer
    if network=='instance_1':
        print(network)
        output1 = layers.Conv2D(num_classes, 3, activation="softmax", padding="same",name='output1')(x)
        output2 = layers.Conv2D(num_classes, 3, activation="softmax", padding="same",name='output2')(x)
        output3 = layers.Conv2D(num_classes, 3, activation="softmax", padding="same",name='output3')(x)
    if network=='instance_2':
        print(network)
        output1 = layers.Conv2D(num_classes, 3, activation="softmax", padding="same",name='output1')(x)
        output2 = layers.Conv2D(num_classes, 3, activation="softmax", padding="same",name='output2')(x)
        output3 = layers.add([output1, output2])
        output3 = layers.Conv2D(2, 3, activation="relu", padding="same")(output3)
        output3 = layers.Conv2D(2, 3, activation="softmax", padding="same")(output3)
        
    if network=='instance_3':
        print(network)
        output1 = layers.Conv2D(num_classes, 3, activation="softmax", padding="same",name='output1')(x)
        
        xz = layers.Concatenate(axis=-1)([output1,x])
        xz = layers.Conv2D(2, 3, activation="relu", padding="same")(xz)
        output2 = layers.Conv2D(2, 3, activation="softmax", padding="same",name='output2')(xz)

        output3 = layers.add([output1, output2])
        output3 = layers.Conv2D(2, 3, activation="relu", padding="same")(output3)
        output3 = layers.Conv2D(2, 3, activation="softmax", padding="same")(output3)
            

    # Define the model
    model = keras.Model(inputs, [output1,output2,output3])
    return model
  


