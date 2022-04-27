import data

def path():
  base_dir='/content/train/'
  dataset_json = '/content/youtube_vis_train.json'
  meta_plus_path = '/content/stemseg/meta_plus_youtube_vis.pickle'
  dataset,meta_info,seqs =  data.parse_generic_video_dataset(base_dir, dataset_json)


