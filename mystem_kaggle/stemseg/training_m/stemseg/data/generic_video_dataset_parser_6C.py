from pycocotools import mask as masktools

import cv2
import json
import numpy as np
import os
import pickle

def select_def(dataset_json,base_dir,dataset):

    temp = dataset_json.replace('youtube_vis_train.json','');
    temp = temp.replace('youtube_vis_val.json','');
    print(temp)

    with open(temp+'meta_plus_youtube_vis.pickle', 'rb') as fp:
      target=pickle.load(fp)
      
    idx=0;dic={1:[],2:[],3:[],4:[],20:[]};clip_length=0;train_id=[];valid_id=[]
    for t in target:
        if t['number_instances']<=2:
            flag=1;
            for c in t['unique_class']:
                if c not in [1,2,3,4,20]:
                    flag=0;
            if flag==1:
                for c in t['unique_class']:
                    dic.update({c:dic[c]+[t['id']]})
                    
            if flag==1:
                clip_length=clip_length+t['clip_length']            
                idx+=1;

                             
    for classn in dic.keys():
        a = int(len(dic[classn])*0.85)
        
        train_id = train_id + dic[classn][:a]
        valid_id = valid_id + dic[classn][a:]
        
    seqs=[]
    for seq in dataset:
        if seq['id'] in train_id:
            seqs.append(GenericVideoSequence(seq, base_dir))
    print('limit size data ',len(seqs))
    return seqs

def parse_generic_video_dataset_limit(base_dir, dataset_json):
    
    limit = True;
    with open(dataset_json, 'r') as fh:
        dataset = json.load(fh)

    meta_info = dataset["meta"]

    # convert instance and category IDs from str to int
    meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}

    """
    dic={};new_id=1;new_class={}
    for k, v in meta_info["category_labels"].items():
      if int(k) in [1,2,3,4,20]:
        dic.update({new_id:v})
        new_class.update({int(k):new_id})
        new_id+=1;
    new_class({6:1000})
    meta_info["category_labels"] = dic
    """
    
    if "segmentations" in dataset["sequences"][0]:
        for seq in dataset["sequences"]:
            seq["categories"] = {int(iid): cat_id for iid, cat_id in seq["categories"].items()}
            
            #print('seq["categories"]',seq["categories"])
            #asd
            """
            dic={}
            for iid, cat_id in seq["categories"].items():
              print(iid,cat_id)
              dic.update({iid:new_class[cat_id]})
            seq["categories"]=dic
            """
            seq["segmentations"] = [
                {
                    int(iid): seg
                    for iid, seg in seg_t.items()
                }
                for seg_t in seq["segmentations"]
            ]

            # sanity check: instance IDs in "segmentations" must match those in "categories"
            seg_iids = set(sum([list(seg_t.keys()) for seg_t in seq["segmentations"]], []))
            assert seg_iids == set(seq["categories"].keys()), "Instance ID mismatch: {} vs. {}".format(
                seg_iids, set(seq["categories"].keys())
            )
    
   
    if limit == False:
        seqs = [GenericVideoSequence(seq, base_dir) for seq in dataset["sequences"]]
    else:
        seqs = select_def(dataset_json,base_dir,dataset["sequences"])

    return seqs, meta_info



def parse_generic_video_dataset(base_dir, dataset_json):
    
    with open(dataset_json, 'r') as fh:
        dataset = json.load(fh)

    meta_info = dataset["meta"]

    # convert instance and category IDs from str to int
    meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}

    if "segmentations" in dataset["sequences"][0]:
        for seq in dataset["sequences"]:
            seq["categories"] = {int(iid): cat_id for iid, cat_id in seq["categories"].items()}
            seq["segmentations"] = [
                {
                    int(iid): seg
                    for iid, seg in seg_t.items()
                }
                for seg_t in seq["segmentations"]
            ]

            # sanity check: instance IDs in "segmentations" must match those in "categories"
            seg_iids = set(sum([list(seg_t.keys()) for seg_t in seq["segmentations"]], []))
            assert seg_iids == set(seq["categories"].keys()), "Instance ID mismatch: {} vs. {}".format(
                seg_iids, set(seq["categories"].keys())
            )
    
   

    seqs = [GenericVideoSequence(seq, base_dir) for seq in dataset["sequences"]]  

    return seqs, meta_info


class GenericVideoSequence(object):
    def __init__(self, seq_dict, base_dir):
        self.base_dir = base_dir
        self.image_paths = seq_dict["image_paths"]
        self.image_dims = (seq_dict["height"], seq_dict["width"])
        self.id = seq_dict["id"]

        self.segmentations = seq_dict.get("segmentations", None)
        self.instance_categories = seq_dict.get("categories", None)

    @property
    def instance_ids(self):
        return list(self.instance_categories.keys())

    @property
    def category_labels(self):
        return [self.instance_categories[instance_id] for instance_id in self.instance_ids]

    def __len__(self):
        return len(self.image_paths)

    def load_images(self, frame_idxes=None):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        images = []
        for t in frame_idxes:
            im = cv2.imread(os.path.join(self.base_dir, self.image_paths[t]), cv2.IMREAD_COLOR)
            #print('self.base_dir',self.base_dir)
            #print('self.image_paths[t]',self.image_paths[t])
            pathopt = self.image_paths[t].replace('JPEGImages/','train_optical/')
            pathopt = pathopt.replace('jpg','png')
            opt = cv2.imread(os.path.join('/kaggle/input/train-vis-opt/', pathopt))
            opt = cv2.resize(opt, (im.shape[1],im.shape[0]), interpolation = cv2.INTER_NEAREST)
            #print('im',im.shape)
            #print('optshape',opt.shape)
            #cv2.imwrite('/content/'+str(t)+'.png',opt)
            imo = np.concatenate((im,opt),axis=2)
            #print('imo',imo.shape)
            if imo is None:
                raise ValueError("No image found at path: {}".format(os.path.join(self.base_dir, self.image_paths[t])))
            images.append(imo)
        #print(asd)
        return images

    def load_masks(self, frame_idxes=None):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        masks = []
        for t in frame_idxes:
            masks_t = []

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    rle_mask = {
                        "counts": self.segmentations[t][instance_id].encode('utf-8'),
                        "size": self.image_dims
                    }
                    masks_t.append(np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8)))
                else:
                    masks_t.append(np.zeros(self.image_dims, np.uint8))

            masks.append(masks_t)

        return masks

    def filter_categories(self, cat_ids_to_keep):
        instance_ids_to_keep = sorted([iid for iid, cat_id in self.instance_categories.items() if iid in cat_ids_to_keep])
        for t in range(len(self)):
            self.segmentations[t] = {iid: seg for iid, seg in self.segmentations[t].items() if iid in instance_ids_to_keep}

    def filter_zero_instance_frames(self):
        t_to_keep = [t for t in range(len(self)) if len(self.segmentations[t]) > 0]
        self.image_paths = [self.image_paths[t] for t in t_to_keep]
        self.segmentations = [self.segmentations[t] for t in t_to_keep]

    def apply_category_id_mapping(self, mapping):
        assert set(mapping.keys()) == set(self.instance_categories.keys())
        self.instance_categories = {
            iid: mapping[current_cat_id] for iid, current_cat_id in self.instance_categories.items()
        }

    def extract_subsequence(self, frame_idxes, new_id=""):
        assert all([t in range(len(self)) for t in frame_idxes])
        instance_ids_to_keep = set(sum([list(self.segmentations[t].keys()) for t in frame_idxes], []))

        subseq_dict = {
            "id": new_id if new_id else self.id,
            "height": self.image_dims[0],
            "width": self.image_dims[1],
            "image_paths": [self.image_paths[t] for t in frame_idxes],
            "categories": {iid: self.instance_categories[iid] for iid in instance_ids_to_keep},
            "segmentations": [
                {
                    iid: segmentations_t[iid]
                    for iid in segmentations_t if iid in instance_ids_to_keep
                }
                for t, segmentations_t in enumerate(self.segmentations) if t in frame_idxes
            ]
        }

        return self.__class__(subseq_dict, self.base_dir)


def visualize_generic_dataset(base_dir, dataset_json):
    from stemseg.utils.vis import overlay_mask_on_image, create_color_map

    seqs, meta_info = parse_generic_video_dataset(base_dir, dataset_json)
    category_names = meta_info["category_labels"]

    cmap = create_color_map().tolist()
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    for seq in seqs:
        if len(seq) > 100:
            frame_idxes = list(range(100, 150))
        else:
            frame_idxes = None

        images = seq.load_images(frame_idxes)
        masks = seq.load_masks(frame_idxes)
        category_labels = seq.category_labels

        print("[COLOR NAME] -> [CATEGORY NAME]")
        color_key_printed = False

        for image_t, masks_t in zip(images, masks):
            for i, (mask, cat_label) in enumerate(zip(masks_t, category_labels), 1):
                image_t = overlay_mask_on_image(image_t, mask, mask_color=cmap[i])

                if not color_key_printed:
                    # print("{} -> {}".format(rgb_to_name(cmap[i][::-1]), category_names[cat_label]))
                    print("{} -> {}".format(cmap[i], category_names[cat_label]))

            color_key_printed = True

            cv2.imshow('Image', image_t)
            if cv2.waitKey(0) == 113:
                exit(0)
