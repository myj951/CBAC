# coding=utf-8
# Copyleft 2019 project LXRT.

from collections import defaultdict
import json
import random

import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append("/home/myj/code/lxmert_caption/src")
from param import args
from pretrain.qa_answer_table import AnswerTable
from utils import load_obj_tsv,load_obj_tsv_vinvl

TINY_IMG_NUM = 500
# FAST_IMG_NUM = 5000
FAST_IMG_NUM = 40000
DATA_ROOT = '/home/myj/3.7Tfile/data/mscoco_imgfeat/FRCNN/'
Split2ImgFeatPath = {
    'mscoco_train': DATA_ROOT+ 'train2014_obj36.tsv',
    'mscoco_minival':DATA_ROOT+  'val2014_obj36.tsv',
    'mscoco_nominival': DATA_ROOT+ 'val2014_obj36.tsv',
    'vgnococo': '/home/myj/3.7Tfile/data/gqa/frcnn/vg_gqa_imgfeat/' + 'vg_gqa_obj36.tsv',
}

# Split2ImgFeatPath = {
#     'mscoco_train': 'data/mscoco_imgfeat/train2014_obj36.tsv',
#     'mscoco_minival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
#     'mscoco_nominival': 'data/mscoco_imgfeat/val2014_obj36.tsv',
#     'vgnococo': 'data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
# }

if args.caption:
    class InputExample(object):
        """A single training/test example for the language model."""
        def __init__(self, uid, sent, visual_feats=None,
                     obj_labels=None, attr_labels=None,
                     is_matched=None, label=None, final_cap=None):
            self.uid = uid
            self.sent = sent
            self.visual_feats = visual_feats
            self.obj_labels = obj_labels
            self.attr_labels = attr_labels
            self.is_matched = is_matched  # whether the visual and obj matched
            self.label = label
            self.final_cap = final_cap
else:
    class InputExample(object):
        """A single training/test example for the language model."""
        def __init__(self, uid, sent, visual_feats=None,
                     obj_labels=None, attr_labels=None,
                     is_matched=None, label=None):
            self.uid = uid
            self.sent = sent
            self.visual_feats = visual_feats
            self.obj_labels = obj_labels
            self.attr_labels = attr_labels
            self.is_matched = is_matched  # whether the visual and obj matched
            self.label = label



class LXMERTDataset:
    def __init__(self, splits: str, qa_sets=None):
        """
        :param splits: The data sources to be loaded
        :param qa_sets: if None, no action
                        o.w., only takes the answers appearing in these dsets
                              and remove all unlabeled data (MSCOCO captions)
        """
        self.name = splits
        self.sources = splits.split(',')

        # Loading datasets to data
        self.data = []
        for source in self.sources:#56579
            if source == 'vqacp_v2_train':
                import os, pickle
                datas = json.load(open("/home/myj/3.7Tfile/data/vqa/%s_questions.json" % source))

                # msvqa = []
                # for s in ['mscoco_train','mscoco_nominival','mscoco_minival']:
                #     msvqa.extend(json.load(open("/home/myj/3.7Tfile/data/lxmert/%s.json" % s)))



                dataroot = '/home/myj/3.7Tfile/data/vqa/'
                split = source
                answer_path = os.path.join(dataroot, '%s_target.pkl' % split)
                answers = pickle.load(open(answer_path, 'rb'))
                answers = sorted(answers, key=lambda x: x['question_id'])


                self.ans2label = json.load(
                    open("/home/myj/code/lxmert_caption/data/vqa/trainval_ans2label.json"))
                self.label2ans = json.load(
                    open("/home/myj/code/lxmert_caption/data/vqa/trainval_label2ans.json"))
                ans_dict = {}
                for ans in answers:
                    dictAns = {}
                    for i in range(len(ans['scores'])):

                        if ans['labels'][i] != []:
                            dictAns[self.label2ans[ans['labels'][i]]] = ans['scores'][i]
                    ans_dict[ans['question_id']] = dictAns



                for data in datas:
                    data = data
                    img_id = data['image_id']
                    padd = 14- len(str(img_id))
                    data['img_id'] = 'COCO_' +data['coco_split']+ '_' + '0'*padd + str(img_id)

                    question_id = data['question_id']
                    ans = ans_dict[question_id]
                    data['labelf'] = dict()
                    data['labelf']['vqa'] = ans

                    data['sentf'] = dict()
                    # data['sentf']['mscoco'] = []
                    data['sentf']['vqa'] = data['question']


                self.data.extend(datas)




            else:
                self.data.extend(json.load(open("/home/myj/3.7Tfile/data/lxmert/%s.json" % source)))





        print("Load %d data from %s" % (len(self.data), self.name))

        # Create answer table according to the qa_sets
        self.answer_table = AnswerTable(qa_sets)
        print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        # Modify the answers
        for datum in self.data:
            labelf = datum['labelf']
            for cat, labels in labelf.items():
                if labels.__class__ == dict:
                    if len(labels) == 0:
                        continue
                    for ans in list(labels.keys()):
                        new_ans = self.answer_table.convert_ans(ans)
                        if self.answer_table.used(new_ans):
                            if ans != new_ans:
                                labels[new_ans] = labels.pop(ans)
                        else:
                            labels.pop(ans)
                else:
                    for label in labels:
                        for ans in list(label.keys()):
                            new_ans = self.answer_table.convert_ans(ans)
                            if self.answer_table.used(new_ans):
                                if ans != new_ans:
                                    label[new_ans] = label.pop(ans)
                            else:
                                label.pop(ans)
        print("")
    def __len__(self):
        return len(self.data)


def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx),


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class LXMERTTorchDataset(Dataset):
    def __init__(self, dataset: LXMERTDataset, topk=999999999):
        super().__init__()
        self.raw_dataset = dataset
        self.task_matched = args.task_matched

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM

        # Load the dataset
        img_data = []

        if args.vinvl:
            for source in self.raw_dataset.sources[1:]:


                if source == 'mscoco_minival':
                    topk =5000
                    split = 'train'
                elif 'mscoco' in source:
                    split = 'train'
                elif 'vg' in source:
                    split = 'vg'
                img_data.extend(load_obj_tsv_vinvl(split, topk=topk))

        else:
            for source in self.raw_dataset.sources:
                if 'cp' in source:
                    sources =     ['mscoco_train', 'mscoco_minival']
                    for source in sources:
                        img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))
                else:
                    img_data.extend(load_obj_tsv(Split2ImgFeatPath[source], topk))

        if args.caption:
            if args.caption:
                self.imgid2cap = {}
                if args.gener_cap:
                    for split in ['train', 'val', 'test','gqa']:
                        d = json.load(
                            open("/home/myj/code/lxmert_caption/data/caption/generate_{}_captions.json".format(split)))
                        self.imgid2cap = {**self.imgid2cap, **d}
                else:
                    for split in ['train', 'val']:
                        d = json.load(open("/home/myj/3.7Tfile/data/lxmert_caption/%s_captions.json" %split))
                        self.imgid2cap = {**self.imgid2cap, **d}

        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        used_data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)
        # self.data = []
        # for datum in used_data:
        #     sentf = datum['sentf']
        #     for sents_cat, sents in sentf.items():
        #         if sents_cat in datum['labelf']:
        #             labels = datum['labelf'][sents_cat]
        #         else:
        #             labels = None
        #         for sent_idx, sent in enumerate(sents):
        #             new_datum = {
        #                 'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
        #                 'img_id': datum['img_id'],
        #                 'sent': sent
        #             }
        #             if labels is not None:
        #                 new_datum['label'] = labels[sent_idx]
        #             self.data.append(new_datum)
        # print("Use %d data in torch dataset" % (len(self.data)))

        self.data = []
        for datum in used_data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                # if sents_cat =='visual7w':
                #     continue
                if sents_cat in datum['labelf']:
                    labels = datum['labelf'][sents_cat]
                else:
                    continue
                    # labels = None
                for sent_idx, sent in enumerate(sents):
                    new_datum = {
                        'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                        'img_id': datum['img_id'],
                        'sent': sent
                    }
                    if labels is not None:
                        new_datum['label'] = labels[sent_idx]
                    self.data.append(new_datum)
        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data)-1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 35)]
        return feat

    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id']

        if args.caption:
            img_id_cap = str(int(datum['img_id'].split("_")[-1]))

            if img_id_cap not in self.imgid2cap:
                if 'vg' in uid[0] or 'gqa' in uid[0] or 'visual7w'in uid[0] :
                # if True:
                    img_id_cap = 'vg' + img_id_cap
                    if img_id_cap in self.imgid2cap:
                        cap = self.imgid2cap[img_id_cap]
                        cap_num = args.cap_num
                        # assert cap_num == 1
                        captions = cap[:cap_num]
                        final_cap = ''
                        SEP_token = " "  # " [SEP] "
                        for c in captions:
                            final_cap = final_cap + c + SEP_token
                    else:
                        final_cap = ' '
                        if 'vg' in uid[0] :
                            print("vg")
                        if 'gqa' in uid[0] :
                            print("gqa")
                        if 'visual7w' in uid[0]:
                            print("visual7w")
                        print(img_id_cap)

                else:
                    final_cap = ' '
                    print("&&&&&&&&&")
            else:
                cap = self.imgid2cap[img_id_cap]
                cap_num = args.cap_num
                assert cap_num == 1
                captions = cap[:cap_num]
                final_cap = ''
                SEP_token = " "  # " [SEP] "
                for c in captions:
                    final_cap = final_cap + c + SEP_token



            # Get image info
        img_info = self.imgid2img[img_id]
        if args.vinvl:
            obj_num = 36
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
            obj_labels = img_info['obj_veb'].copy()
            obj_confs = img_info['objects_conf'].copy()
            attr_labels = None
            attr_confs = None

        else:
            obj_num = img_info['num_boxes']
            feats = img_info['features'].copy()
            boxes = img_info['boxes'].copy()
            obj_labels = img_info['objects_id'].copy()
            obj_confs = img_info['objects_conf'].copy()
            attr_labels = img_info['attrs_id'].copy()
            attr_confs = img_info['attrs_conf'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        # assert "?" in sent
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data)-1)]
                while other_datum['img_id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]
                sent = other_datum['sent']

        # Label, convert answer to id
        if 'label' in datum:
            label = datum['label'].copy()
            for ans in list(label.keys()):
                label[self.raw_dataset.answer_table.ans2id(ans)] = label.pop(ans)
        else:
            label = None

        # Create target
        if args.caption:
            example = InputExample(
                uid, sent, (feats, boxes),
                (obj_labels, obj_confs), (attr_labels, attr_confs),
                is_matched, label, final_cap
            )
        else:
            example = InputExample(
                uid, sent, (feats, boxes),
                (obj_labels, obj_confs), (attr_labels, attr_confs),
                is_matched, label
            )
        return example


class LXMERTEvaluator:
    def __init__(self, dataset: LXMERTDataset):
        self.raw_dataset = dataset

        # Create QA Eval Data
        self.data = []
        count = 0
        for datum in self.raw_dataset.data:#[-2:-1]
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents.__class__ == str and 'vqa' in  sentf.keys():
                    sents = sentf['vqa']
                    if sents_cat in datum['labelf']:
                        if len(datum['labelf']['vqa']) == 0:
                            count+=1
                            continue  # A labeled dataset
                        labels = datum['labelf']['vqa']
                        sent_idx = 0
                        sent = sents
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels
                        }
                        self.data.append(new_datum)
                    continue

                else:
                    if sents_cat in datum['labelf']:
                        labels = datum['labelf'][sents_cat]
                        for sent_idx, sent in enumerate(sents):
                            new_datum = {
                                'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                                'img_id': datum['img_id'],
                                'sent': sent,
                                'dset': sents_cat,
                                'label': labels[sent_idx]
                            }
                            self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplemented
