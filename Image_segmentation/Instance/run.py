
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
import keras.backend as K
import matplotlib.pyplot as plt

def vis(x,y):
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(x[0][:,:,0])
    axarr[0,1].imshow(y[0][:,:,0])
    axarr[1,0].imshow(x[1][:,:,0])
    axarr[1,1].imshow(y[1][:,:,0])

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
      if argss.upload=='git':
        os.system('. '+argss.basepath+'upload.sh')

def instance_loss(y_true, y_pred):
  loss1 = K.sparse_categorical_crossentropy(y_true, y_pred)
  ndf = y_true
  condition = tf.equal(ndf, 1);
  case_true=(ndf*0)+5;case_false=ndf
  ndf1 = tf.where(condition, case_true, case_false)

  condition = tf.equal(ndf1, 2);
  case_true=(ndf*0)+1;case_false=ndf1
  ndf2 = tf.where(condition, case_true, case_false)

  condition = tf.equal(ndf2, 5);
  case_true=(ndf*0)+2;case_false=ndf2
  ndf3 = tf.where(condition, case_true, case_false)

  loss2 = K.sparse_categorical_crossentropy(ndf3, y_pred)
  r1 = tf.reduce_mean(loss1)
  r2 = tf.reduce_mean(loss2)

  if r1>=r2:
    return loss2
  else:
    return loss1

def start(args):

  dicid={};
  #if args.task=='semantic_seg':
  i=1;
  for x in args.classid:
    dicid.update({x:i});i+=1

  global argss
  argss=args
  allframe_train,allframe_val,allframe_test = path.getinfo(args)
  random.Random(1337).shuffle(allframe_train)
    
  dispatcher_loader={1:path.dataloader,2:path.dataloader}

  # Instantiate data Sequences for each split
  train_gen = dispatcher_loader[args.branch_input](args,allframe_train,dicid)
  val_gen = dispatcher_loader[args.branch_input](args,allframe_val,dicid)
  #train_gen = path.dataloader(args,allframe_train,dicid)
  #val_gen = path.dataloader(args,allframe_val,dicid)
  
  #x, y = next(iter(train_gen))
  #vis(x,y)

  keras.backend.clear_session()
  if args.mode=='train':
    mymodel=model.network(args)
    mymodel.summary()

    if args.loss=='default':
        mymodel.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    elif args.loss=='instance_loss':
        mymodel.compile(optimizer="adam", loss=instance_loss)


    callbacks = [
        keras.callbacks.ModelCheckpoint(args.model_dir, save_best_only=True),CSVLogger(args.model_dir+'_log.csv', append=True, separator=','),CustomCallback()
    ]
    if args.restore==True:
      mymodel = load_model(args.model_dir)
      
    mymodel.fit(train_gen, epochs=args.epoch, validation_data=val_gen, callbacks=callbacks)
  
  if args.mode=='test':
    test_gen = dispatcher_loader[args.branch_input](args,allframe_test,dicid)    
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

