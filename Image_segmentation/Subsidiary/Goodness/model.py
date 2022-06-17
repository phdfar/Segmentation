from tensorflow.keras import layers
from tensorflow import keras
from keras.models import load_model
#from models.Spectral import run

from keras.utils.vis_utils import plot_model
def network(args):
  if args.network=='inception_default':
    return inception_default(args.imagesize,args.channel_input)
 
def inception_default(img_size,channel_input):
    inputs1 = keras.Input(shape=img_size + (channel_input,))
    inputs2 = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    
    y = layers.Conv2D(32, 3, strides=2, padding="same")(inputs2)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    
    y = layers.Conv2D(32, 3, strides=2, padding="same")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)

    z = layers.Concatenate()([x, y])
    
    previous_block_activation = z  # Set aside residual
    x = z

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
    

    # Add a per-pixel classification layer
    x = layers.Conv2D(128, 3, activation="relu", padding="valid")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="valid")(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="valid")(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="valid")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(192,activation="relu")(x)
    x = layers.Dense(48,activation="relu")(x)
    x = layers.Dense(12,activation="relu")(x)
    outputs = layers.Dense(1,activation="sigmoid")(x)

    #outputs = layers.Conv2D(16, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model([inputs1,inputs2],outputs)
    return model
  
# model = inception_default((384,640),3)
# model.summary()

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)