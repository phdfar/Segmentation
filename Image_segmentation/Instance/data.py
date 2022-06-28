import cv2
import json
import numpy as np
import os
from pycocotools import mask as masktools
import matplotlib.pyplot as plt

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
    
    """
    for seq in dataset["sequences"]:
        #seqs = GenericVideoSequence(seq, base_dir) 
        break
    """
    return dataset["sequences"], meta_info , seqs


class GenericVideoSequence(object):
    def __init__(self, seq_dict, base_dir):
        self.base_dir = base_dir
        self.image_paths = seq_dict["image_paths"]
        self.image_dims = (seq_dict["height"], seq_dict["width"])
        self.id = seq_dict["id"]
        self.length = seq_dict["length"]
        self.segmentations = seq_dict.get("segmentations", None)
        self.instance_categories = seq_dict.get("categories", None)
        """
        self.mask = self.load_masks()
        self.num_instance = self.instance_ids
        self.RGBimage = self.load_images()
        self.category_labels_instances = self.category_labels
        """
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
            if im is None:
                raise ValueError("No image found at path: {}".format(os.path.join(self.base_dir, self.image_paths[t])))
            images.append(im)

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
    
    def load_masks_for_instance(self, frame_idxes=None):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        masks = {}; masks_total = 0; num_pixel=[];p=0;
        for t in frame_idxes:
            masks_t = []

            for instance_id in self.instance_ids:
                
                if instance_id in self.segmentations[t]:
                    rle_mask = {
                        "counts": self.segmentations[t][instance_id].encode('utf-8'),
                        "size": self.image_dims
                    }
                    data = np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
                    #masks_t.append(data)
                    masks_total = masks_total + data
                    tp = np.where(data==1);
                    num_pixel.append(len(tp[0]))
                else:
                    data = np.zeros(self.image_dims, np.uint8)
                    #masks_t.append(data)
                    masks_total = masks_total + data
                    num_pixel.append(0)
                masks.update({p:data})
                p+=1;
        masks_total[masks_total!=0]=1;
        num_pixel = list(np.argsort(num_pixel));num_pixel.reverse() 
        allmask = []
        for index in num_pixel:
            allmask.append(masks[index])
        return allmask,masks_total
    

    def load_multi_masks(self, frame_idxes=None):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        masks_t = 0;
        for t in frame_idxes:

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    rle_mask = {
                        "counts": self.segmentations[t][instance_id].encode('utf-8'),
                        "size": self.image_dims
                    }
                    masks_t = masks_t + np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
                else:
                    masks_t = masks_t + np.zeros(self.image_dims, np.uint8)
                    
        masks_t[masks_t!=0]=1;

        return masks_t
    def load_one_masks(self, frame_idxes=None):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        for t in frame_idxes:

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    rle_mask = {
                        "counts": self.segmentations[t][instance_id].encode('utf-8'),
                        "size": self.image_dims
                    }
                    masks_t=np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
                else:
                    masks_t=np.zeros(self.image_dims, np.uint8)


        return masks_t
    
    
    def load_multi_masks_semantic(self, frame_idxes,dicid):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        masks_t = 0;
        for t in frame_idxes:

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    lb = dicid[self.instance_categories[instance_id]]
                    rle_mask = {
                        "counts": self.segmentations[t][instance_id].encode('utf-8'),
                        "size": self.image_dims
                    }
                    temp = np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
                    temp[temp!=0]=lb;
                    masks_t = masks_t + temp
                else:
                    masks_t = masks_t + np.zeros(self.image_dims, np.uint8)
                    
        return masks_t
    
    def load_one_masks_semantic(self, frame_idxes,dicid):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        for t in frame_idxes:

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    lb = dicid[self.instance_categories[instance_id]]
                    rle_mask = {
                        "counts": self.segmentations[t][instance_id].encode('utf-8'),
                        "size": self.image_dims
                    }
                    masks_t=np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
                    masks_t[masks_t!=0]=lb;
                else:
                    masks_t=np.zeros(self.image_dims, np.uint8)
                                

        return masks_t
    def load_class(self, frame_idxes,dicid):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))
        category=[]
        for t in frame_idxes:

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    lb = dicid[self.instance_categories[instance_id]]
                else:
                    lb = 0;
                category.append(lb)

        return list(set(category))
    def load_multi_masks_instance(self, frame_idxes):
        if frame_idxes is None:
            frame_idxes = list(range(len(self.image_paths)))

        masks_t = 0;
        for t in frame_idxes:

            for instance_id in self.instance_ids:
                if instance_id in self.segmentations[t]:
                    #lb = dicid[self.instance_categories[instance_id]]
                    rle_mask = {
                        "counts": self.segmentations[t][instance_id].encode('utf-8'),
                        "size": self.image_dims
                    }
                    temp = np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
                    temp[temp!=0]=instance_id;
                    masks_t = masks_t + temp
                else:
                    masks_t = masks_t + np.zeros(self.image_dims, np.uint8)

        return masks_t
    
    
"""
base_dir='/content/train/'
dataset_json = '/content/youtube_vis_train.json'
dataset,meta_info,seqs =  parse_generic_video_dataset(base_dir, dataset_json)
"""
