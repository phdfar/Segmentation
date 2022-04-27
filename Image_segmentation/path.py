import data
import pickle

  
def getinfo(args):
  base_dir='/content/train/'
  dataset_json = '/content/youtube_vis_train.json'
  meta_plus_path = '/content/stemseg/meta_plus_youtube_vis.pickle'
  dataset,meta_info,seqs =  data.parse_generic_video_dataset(base_dir, dataset_json)
  with open(meta_plus_path, 'rb') as handle:
    meta_plus = pickle.load(handle)
  valid=[]
  for i in meta_plus:
    if i['number_instances']==args.num_instance and i['number_unique_class']==args.unq_class:
      for c in i['unique_class']:
        if c in list(args.classid):
          valid.append(i['id'])
          print(i)