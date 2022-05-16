
import random
import path
from tensorflow import keras
import model
import accuracy
from keras.models import load_model
from keras.callbacks import CSVLogger
import os
import tensorflow as tf
global argss


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
      if argss.upload=='git':
        os.system('. '+argss.basepath+'upload.sh')

def start(args):

  dicid={};
  if args.task=='semantic_seg':
    i=1;
    for x in args.classid:
      dicid.update({x:i});i+=1

  global argss
  argss=args
  allframe_train,allframe_val,allframe_test = path.getinfo(args)
  random.Random(1337).shuffle(allframe_train)

  # Instantiate data Sequences for each split
  train_gen = path.dataloader(args,allframe_train,dicid)
  val_gen = path.dataloader(args,allframe_val,dicid)

  keras.backend.clear_session()
  if args.mode=='train':
    mymodel=model.network(args)
    mymodel.summary()

    
    mymodel.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=[tf.keras.metrics.MeanIoU(args.num_class)])

    callbacks = [
        keras.callbacks.ModelCheckpoint(args.model_dir, save_best_only=True),CSVLogger(args.model_dir+'_log.csv', append=True, separator=','),CustomCallback()
    ]
    if args.restore==True:
      mymodel = load_model(args.model_dir)
      
    mymodel.fit(train_gen, epochs=args.epoch, validation_data=val_gen, callbacks=callbacks)
  
  if args.mode=='test':
    test_gen = path.dataloader(args,allframe_test,dicid)    
    mymodel = load_model(args.model_dir)
    mymodel.evaluate(test_gen);
    accuracy.start(mymodel,allframe_test,args.model_dir,args,dicid)

    
    

  """
  tap=[];vap=[];tep=[]
  
  for pathx in allframe_train:
    frameindex= list(pathx.keys())[0]
    imagepath = pathx[frameindex][0]
    tap.append(imagepath)
    
  for pathx in allframe_val:
    frameindex= list(pathx.keys())[0]
    imagepath = pathx[frameindex][0]
    vap.append(imagepath)
    
  for pathx in allframe_test:
    frameindex= list(pathx.keys())[0]
    imagepath = pathx[frameindex][0]
    tep.append(imagepath)
  
  import pickle
  with open('allpath2.pickle', 'wb') as handle:
    pickle.dump([tap,vap,tep], handle, protocol=pickle.HIGHEST_PROTOCOL)
    

  X, y = next(iter(test_gen))
  print(X.shape, y.shape)

  import pickle
  with open('loader.pickle', 'wb') as handle:
    pickle.dump([X,y], handle, protocol=pickle.HIGHEST_PROTOCOL)
  """

