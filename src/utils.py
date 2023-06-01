# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time

import numpy as np
import param
csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
# FIELDNAMES = ["img_id", "img_h", "img_w", "num_boxes", "boxes",
#               "features"]

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")#50600
        for i, item in enumerate(reader):
            if i%100 ==0:
                print("loading {} images".format(i))
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']

            decode_config = [
                ('objects_id', (boxes,), np.int64),
                ('objects_conf', (boxes,), np.float32),
                ('attrs_id', (boxes,), np.int64),
                ('attrs_conf', (boxes,), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            if param.args.vqa_cp:
                item['img_id'] = int(item['img_id'].split("_")[-1])

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


import tsv_file
import  json


step = 0
def load_obj_tsv_vinvl(fname, topk=None,mode="init",start=0,end=1000000):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """

    file_object = open('/home/myj/3.7Tfile/data/vg/objects_vocab.txt')
    obj_list = []
    for line in file_object.readlines():  # 依次读取每一行
        line = line.strip()  # 去掉每行的头尾空白
        obj_list.append(line)
    file_object.close()
    obj_list.insert(0, 'None')
    obj_dict = {}
    for i in range(len(obj_list)):
        obj_dict[obj_list[i]] = i


    global  step
    split = fname
    data = []
    start_time = time.time()
    print("Start to vinvl detected objects from %s" % fname)
    # get the V_feature of tsv file:
    # inlude:
    dataroot = '/home/myj/3.7Tfile/data/mscoco_imgfeat/vinvl/FRCNN-X152C4/model_0060000/'
    if split == 'train' or 'val' in split:
        imageid_to_index = json.load(open(dataroot+ 'imageid2idx.json'.format(split)))
        image_file = open(dataroot + 'features.tsv'.format(split), 'r')
        step = step + 1
        label_tsv = tsv_file.TSVFile( dataroot+'predictions.tsv'.format(split))
    elif 'test' in split:
        imageid_to_index = json.load(
            open(dataroot+ 'coco2015{}/imageid2idx.json'.format(split)))
        image_file = open(dataroot+'coco2015{}/features.tsv'.format(split), 'r')
        label_tsv = tsv_file.TSVFile( dataroot+'coco2015{}/predictions.tsv'.format(split))
    elif 'vg' in split:
        dataroot = '/home/myj/3.7Tfile/data/gqa/vinvl/model_0060000/'
        imageid_to_index = json.load(
            open(dataroot + 'imageid2idx.json'.format(split)))
        image_file = open(dataroot + 'features.tsv'.format(split), 'r')
        label_tsv = tsv_file.TSVFile(dataroot + 'predictions.tsv'.format(split))
    labels = {}
    data = []
    i=0
    for line_no in range(label_tsv.num_rows()):
        i=i+1
        row = label_tsv.seek(line_no)
        image_id = row[0]
        # if int(image_id) in img_keys:
        arr = [s.strip() for s in image_file.readline().split('\t')]

        num_boxes = int(arr[1])
        feat = np.frombuffer(base64.b64decode(arr[2]), dtype=np.float32).reshape((-1, 2054))
        results = json.loads(row[1])
        objects = results['objects'] if type(
            results) == dict else results
        assert feat.shape[0] == objects.__len__()
        feat = feat[:, :2048]
        if feat.shape[0]>36:
            # feat = feat[:36]
            feat = feat[:36]

            objects = objects[:36]
        else:
            padding_len = 36 - feat.shape[0]
            # feat_padding = np.zeros((padding_len, 2054), dtype=np.float32)
            feat_padding = np.zeros((padding_len, 2048), dtype=np.float32)

            obj_padding_dict = [
                {'class': 'None', 'conf': 0, 'rect': [0, 0, 0, 0],
                 'attributes': 'None', 'attr_scores': [0]}]*padding_len
            feat = np.concatenate((feat,feat_padding),axis=0)
            objects.extend(obj_padding_dict)


        if True:
            labels = {
                "img_id": image_id,
                "img_h": results["image_h"] if type(
                    results) == dict else 600,
                "img_w": results["image_w"] if type(
                    results) == dict else 800,
                "obj_veb": [cur_d['class'] for cur_d in objects],
                "boxes": np.array([cur_d['rect'] for cur_d in objects],
                                  dtype=np.float32),
                "features":feat,
                "objects_conf": np.array([cur_d['conf'] for cur_d in objects],
                                  dtype=np.float32),
            }
            data.append(labels.copy())
            labels.clear()

        assert image_id == arr[0]
        assert feat.shape[0] == objects.__len__()
        if data.__len__() %10000 == 0:
            print("load image nums : {}".format(data.__len__()))
        if topk and data.__len__() > topk:
            elapsed_time = time.time() - start_time
            print("******************************************")
            print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
            print("******************************************")
            return data
    elapsed_time = time.time() - start_time
    print("******************************************")
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    print("******************************************")
    return data
