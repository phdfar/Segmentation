
import random
import path
from tensorflow import keras
import model
from keras.models import load_model

def start(args):

  allframe_train,allframe_val,allframe_test = path.getinfo(args)
  random.Random(1337).shuffle(allframe_train)

  # Instantiate data Sequences for each split
  train_gen = path.dataloader(args,allframe_train)
  val_gen = path.dataloader(args,allframe_val)
  test_gen = path.dataloader(args,allframe_test)

  keras.backend.clear_session()
  if args.mode=='train':
    mymodel=model.network(args)
    mymodel.summary()

    
    mymodel.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint(args.model_dir, save_best_only=True)
    ]

    mymodel.fit(train_gen, epochs=args.epoch, validation_data=val_gen, callbacks=callbacks)
  
  if args.mode=='test':
    mymodel = load_model(args.model_dir)
    mymodel.evaluate(test_gen)


  """
  X, y = next(iter(test_gen))
  print(X.shape, y.shape)

  import pickle
  with open('loader.pickle', 'wb') as handle:
    pickle.dump([X,y], handle, protocol=pickle.HIGHEST_PROTOCOL)
  """
