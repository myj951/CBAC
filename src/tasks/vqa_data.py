# coding=utf-8
# Copyleft 2019 project LXRT.
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args

from dictionary import Dictionary

import utils
from utils import load_obj_tsv, load_obj_tsv_vinvl

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
if args.test:
    TINY_IMG_NUM = 512
else:
    TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# FILTERWORD  =['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
#  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
#  "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
#  'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
#  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
#  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
#  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
#  'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
#  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
#  'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
#  'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
#  'isn', "isn't", 'ma', 'mightn', "mightn't",'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
#  "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

FILTERWORD = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your',
              'yours', 'yourself', 'yourselves',
              # 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
              'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this',
              'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
              'has', 'had',
              'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
              'until', 'while',
              'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
              'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
              'again', 'further', 'then',
              'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
              'most', 'other',
              'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
              'will',
              'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
              "aren't",
              'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
              "haven't",
              'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
              'shouldn',
              "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",

              ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', "''", "'", '`', '``', '-',
              '--', '|', '\\/',

              'the', ]

# The path to data and image features.
VQA_DATA_ROOT = '/home/myj/code/lxmert-caption/data/vqa'

if args.gqa:
    MSCOCO_IMGFEAT_ROOT = '/home/myj/3.7Tfile/data/gqa/frcnn/vg_gqa_imgfeat/'
    SPLIT2NAME = {
        'train': 'vg_gqa',
        'valid': 'vg_gqa',

        'testdev': 'gqa_testdev', }
else:
    MSCOCO_IMGFEAT_ROOT = '/home/myj/3.7Tfile/data/mscoco_imgfeat/FRCNN/'

    SPLIT2NAME = {
        'train': 'train2014',
        'valid': 'val2014',
        'minival': 'val2014',
        'nominival': 'val2014',
        'test': 'test2015', }


# MSCOCO_IMGFEAT_ROOT = '/home/myj/data/frcnn/lxmert/mscoco_imgfeat/'
# SPLIT2NAME = {
#     'train': 'train2014',
#     'valid': 'val2014',
#     'minival': 'val2014',
#     'nominival': 'val2014',
#     'test': 'test2015',
# }
# SPLIT2NAME = {
#     'train': 'trainval',
#     'valid': 'trainval',
#     'minival': 'trainval',
#     'nominival': 'trainval',
#     'test': 'test2015',
# }


class VQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """

    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')
        # self.dictionary = Dictionary.load_from_file(os.path.join('/home/myj/code/bottom-up-attention-vqa-master/tools', 'dictionary.json'))
        # Loading datasets
        self.data = []
        if args.vqa_cp:
            for split in self.splits:  # '/home/myj/dataset/data/vqa/train.json',
                dataroot = '/home/myj/3.7Tfile/data/vqa/'
                question_path = os.path.join(
                    dataroot, '%s_questions.json' % split)
                self.questions = sorted(json.load(open(question_path)),
                                        key=lambda x: x['question_id'])
                self.data = self.questions
                answer_path = os.path.join(dataroot, '%s_target.pkl' % split)
                answers = pickle.load(open(answer_path, 'rb'))
                self.answers = sorted(answers, key=lambda x: x['question_id'])
                assert (len(self.questions) == len(self.answers))
                entries = []

                count = 0
                for a, b in zip(self.questions, self.answers):
                    q1 = a['question_id']
                    q2 = b['question_id']
                    assert q1 == q2
                    a.update(b)
                self.data = self.questions
        elif args.gqa:
            for split in self.splits:
                self.data.extend(json.load(open("/home/myj/code/lxmert_caption/data/gqa/%s.json" % split)))
        elif args.okvqa:
            for split in self.splits:
                self.data.extend(json.load(open(
                    "/home/myj/code/lxmert_caption/data/vqa/ok-vqa/OpenEnded_mscoco_{}_questions.json".format(SPLIT2NAME[split])))[
                                     'questions'])
                if split == 'valid':
                    split = 'val'
                with open("/home/myj/code/lxmert_caption/data/vqa/ok-vqa/cache/{}_target.json".format(split)) as fd:
                    answers = json.load(fd)
                answers = sorted(answers, key=lambda x: x['question_id'])

            self.id2answer = {
                datum['question_id']: datum  # ???????????
                for datum in answers
            }
        else:
            for split in self.splits:
                self.data.extend(json.load(open("/home/myj/code/lxmert_caption/data/vqa/%s.json" % split)))

        data_foramt = []
        for split in ['train', 'minival', 'nominival']:
            data_foramt.extend(json.load(open("/home/myj/code/lxmert_caption/data/vqa/%s.json" % split)))
        qid_qtype ={}
        for i in data_foramt:
            qid_qtype[i['question_id']] = i['question_type']
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        if args.vqa_cp:
            if 'test' in self.splits:
                vair = False #change the question with data argument
                add = False
            else:
                vair = args.vair #change the question with data argument
                add = args.add
        else:
            vair = False  # change the question with data argument
            add = False
        dataAdd = []
        # Convert list to dict (for evaluation)
        for datas in self.data:
            if 'image_id' in datas.keys():
                datas['img_id'] = datas['image_id']
                if vair:
                    oriSent = datas['question']

                    qusType = qid_qtype[datas['question_id']]
                    qusTypeLen = len(qusType)

                    if qusType == 'none of the above':
                        datas['sent'] = oriSent

                    move_part = oriSent[:qusTypeLen]
                    remaining_part = oriSent[qusTypeLen:-1]

                    datas['sent'] = remaining_part + " " + move_part + "?"

                    # if add:
                else:
                    datas['sent'] = datas['question']

            if not isinstance(datas['img_id'], str):
                datas['img_id'] = str(datas['img_id'])
            elif '_' in datas['img_id']:
                datas['img_id'] = str(int(datas['img_id'].split("_")[-1]))
            sent = datas['sent']
            entities = sent.lower().split('?')[0].split(" ")
            filter_entities = ''
            for en_ in entities:
                if en_ not in FILTERWORD:
                    # entities.remove(en_)
                    filter_entities = filter_entities + en_ + " "
            datas['entity'] = filter_entities
            if args.vqa_cp:
                datas['question_type'] = qid_qtype[datas['question_id']]

            if add:
                datasAdd = datas
                datasAdd['sent'] = oriSent
                dataAdd.append(datasAdd)
        if add:
            self.data +=dataAdd
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }
        # Answers
        if args.gqa:
            self.ans2label = json.load(
                open("/home/myj/code/lxmert_caption/data/gqa/trainval_ans2label.json"))
            self.label2ans = json.load(
                open("/home/myj/code/lxmert_caption/data/gqa/trainval_label2ans.json"))
            for ans, label in self.ans2label.items():
                assert self.label2ans[label] == ans
        elif args.okvqa:
            self.ans2label = json.load(open("/home/myj/code/lxmert_caption/data/vqa/ok-vqa/cache/ans2label.json"))  # exist
            self.label2ans = json.load(open("/home/myj/code/lxmert_caption/data/vqa/ok-vqa/cache/label2ans.json"))
        else:
            # Answers
            self.ans2label = json.load(
                open("/home/myj/code/lxmert_caption/data/vqa/trainval_ans2label.json"))
            self.label2ans = json.load(
                open("/home/myj/code/lxmert_caption/data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)



    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


def load_clss(fname=None, type='object'):
    if fname == None:
        if type == 'object':
            fname = '/home/myj/code/lxmert_caption/1600-400-20/objects_vocab.txt'
        else:
            fname = '/home/myj/code/lxmert_caption/1600-400-20/attributes_vocab.txt'

        with open(fname, 'r') as f:
            list = []
            lines = f.readlines()
            for line in lines:
                line = line.replace("-", " ").replace(",", " ")
                line = line.split("\n")[0]
                list.append(line.split("\n")[0])
        dict_ = {}
        for i, l in zip(range(len(list)), list):
            dict_[i] = l

        return dict_



from collections import defaultdict, Counter
def get_bias2(train_dset):
    # Compute the bias:
    # The bias here is just the expected score for each answer/question type
    answer_voc_size = 3129

    # question_type -> answer -> total score
    question_type_to_probs = defaultdict(Counter)

    # question_type -> num_occurances
    question_type_to_count = Counter()
    for ex in train_dset.data:
        # ans = ex["answer"]
        q_type = ex["question_type"]
        question_type_to_count[q_type] += 1
        if ex["label"] is not None:
            for label, score in ex["label"] :
                label_id = train_dset.ans2label[label]
                question_type_to_probs[q_type][label_id] += score
    question_type_to_prob_array = {}

    for q_type, count in question_type_to_count.items():
        prob_array = np.zeros(answer_voc_size, np.float32)
        for label, total_score in question_type_to_probs[q_type].items():
            prob_array[label] += total_score
        prob_array /= count
        question_type_to_prob_array[q_type] = prob_array

    for ds in [train_dset]:
        for ex in ds.data:
            q_type = ex["question_type"]
            ex["bias"] = question_type_to_prob_array[q_type]

"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""


class VQATorchDataset(Dataset):
    def __init__(self, dataset: VQADataset):
        super().__init__()
        self.raw_dataset = dataset



        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []

        if args.vqa_cp and not args.vinvl:
            for split in ["train", "valid"]:
                load_topk = 5000 if (split == 'minival' and topk is None) else topk
                img_data.extend(load_obj_tsv(
                    os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                    topk=load_topk))
        elif args.vqa_cp and args.vinvl:
            split = 'train'
            load_topk = 5000 if (split == 'minival' and topk is None) else topk
            img_data.extend(load_obj_tsv_vinvl(split, topk=load_topk))
        elif args.vinvl:
            for split in dataset.splits:
                load_topk = 50000 if (split == 'minival' and topk is None) else topk
                img_data.extend(load_obj_tsv_vinvl(split, topk=load_topk))
                break
        else:
            for split in dataset.splits:
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 if (split == 'minival' and topk is None) else topk
                if args.gqa and split=='valid':
                    continue
                img_data.extend(load_obj_tsv(
                    os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                    topk=load_topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            if img_datum['img_id'].__class__ == int:
                img_datum['img_id'] = str(img_datum['img_id'])

            if '_' in img_datum['img_id']:
                img_datum['img_id'] = str(int(img_datum['img_id'].split("_")[-1]))
            self.imgid2img[img_datum['img_id']] = img_datum

        if args.caption:
            self.imgid2cap = {}
            if args.gener_cap:
                if args.gqa:
                    for split in ['gqa']:
                        d = json.load(open(
                            "/home/myj/code/lxmert_caption/data/caption/generate_{}_captions.json".format(
                                split)))
                        self.imgid2cap = {**self.imgid2cap, **d}
                else:
                    for split in ['train', 'val', 'test']:
                        d = json.load(open(
                            "/home/myj/code/lxmert_caption/data/caption/generate_{}_captions.json".format(
                                split)))
                        self.imgid2cap = {**self.imgid2cap, **d}
            # elif args.vqa_cp:
            #     for split in ['train', 'val']:
            #         d = json.load(
            #             open("/home/myj/code/lxmert_caption/data/caption/cp_{}_captions.json".format(split)))
            #         self.imgid2cap = {**self.imgid2cap, **d}
            else:
                for split in dataset.splits:
                    for split in ['train', 'val']:
                        d = json.load(open("/home/myj/code/lxmert_caption/data/caption/{}_captions.json".format(
                                split)))

                        # d = json.load(open(
                        #     "/home/myj/code/lxmert_caption/data/caption/train_captions.json".format(split)))
                        self.imgid2cap = {**self.imgid2cap, **d}

        # Only kept the data with loaded image features

        dataAdd = []
        data_foramt = []
        for split in ['train', 'minival', 'nominival']:
            data_foramt.extend(json.load(open("/home/myj/code/lxmert_caption/data/vqa/%s.json" % split)))
        qid_qtype ={}
        if args.vqa_cp:
            for i in data_foramt:
                qid_qtype[i['question_id']] = i['question_type']
            if 'test' in dataset.splits[0]:
                vair = False  # change the question with data argument
                add = False
            else:
                vair = args.vair   # change the question with data argument
                add = args.add
            for datass in self.raw_dataset.data:
                datass['img_id'] = datass['image_id']
                if vair:
                    oriSent = datass['question']

                    qusType = qid_qtype[datass['question_id']]
                    qusTypeLen = len(qusType)

                    if qusType == 'none of the above':
                        datass['sent'] = oriSent

                    move_part = oriSent[:qusTypeLen]
                    remaining_part = oriSent[qusTypeLen:-1]

                    datass['sent'] = remaining_part + " " + move_part + "?"

                    # if add:
                else:
                    datass['sent'] = datass['question']
                datass['label'] = {}
                for l, s in zip(datass['labels'], datass['scores']):
                    datass['label'][self.raw_dataset.label2ans[l]] = s
                if add:
                    datasAdd = datass.copy()
                    datasAdd['sent'] = oriSent
                    dataAdd.append(datasAdd)
            if add:
                self.raw_dataset.data += dataAdd

        self.data = []
        if args.vinvl and not args.vqa_cp:
            for datum in self.raw_dataset.data:
                img_id = datum['img_id']
                if img_id in self.imgid2img:
                    self.data.append(datum)
            print("Use %d data in torch dataset" % (len(self.data)))
        elif args.vqa_cp:
            for datum in self.raw_dataset.data:
                datum['img_id'] = str(datum['img_id'])
                if datum['img_id'] in self.imgid2img:
                    self.data.append(datum)
            print("Use %d data in torch dataset" % (len(self.data)))
            print()
        else:
            for datum in self.raw_dataset.data:
                if datum['img_id'] in self.imgid2img:
                    self.data.append(datum)
            print("Use %d data in torch dataset" % (len(self.data)))
            print()

        imgid2img2 = {}
        if args.vinvl:
            for datum in self.raw_dataset.data:
                if "_" in str(int(datum['img_id'].split("_")[-1])):
                    img_id = str(int(datum['img_id'].split("_")[-1]))
                else:
                    img_id = datum['img_id']
                if img_id in self.imgid2img:
                    if img_id not in imgid2img2:
                        imgid2img2[img_id] = self.imgid2img[img_id]

            self.imgid2img = imgid2img2

        self.obj_dict = load_clss(type="object")
        self.attr_dict = load_clss(type="attr")
        # self.dictionary = Dictionary.load_from_file(
        #     os.path.join('/home/myj/code/bottom-up-attention-vqa-master/tools', 'dictionary.json'))


    def __len__(self):
        return len(self.data)

    def encode_question(self, question, max_length=4):
        final_tokens =[]
        for q in question:
            tokens = self.dictionary.tokenize(q, False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens +padding
            assert(len(tokens)==max_length)
            final_tokens.append(tokens)
        return torch.LongTensor(final_tokens)

    def __getitem__(self, item: int):

        datum = self.data[item]
        if args.vinvl and not args.vqa_cp:
            img_id = datum['img_id']
        elif args.vqa_cp and args.vinvl:
            img_id = str(datum['img_id'])
        else:
            img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        if args.vinvl:
            obj_num = 36
        else:
            obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        attrs_id = img_info['attrs_id'].copy()
        objects_id = img_info['objects_id'].copy()

        dens_labels = ''
        TOPLK = 20
        count = 1

        dens_labels = []

        for a, o in zip(attrs_id, objects_id):
            attrs_word = self.attr_dict[a]
            objects_word = self.obj_dict[o]
            dens_label = attrs_word + ' ' + objects_word
            dens_labels.append(dens_label)

        assert obj_num == len(boxes) == len(feats)
        dens_labels.append(datum['entity'])


        img_id = datum['img_id']
        if args.gqa:
            if args.caption:
                if args.gqa:
                    img_id_cap = datum['img_id']
                else:
                    img_id_cap = str(int(datum['img_id'].split("_")[-1]))

                if img_id_cap not in self.imgid2cap:
                    if True:
                        # if True:
                        img_id_cap = 'vg' + img_id_cap
                        if img_id_cap in self.imgid2cap:
                            cap = self.imgid2cap[img_id_cap]
                            cap_num = args.cap_num
                            # assert cap_num == 1
                            captions = cap[:cap_num]
                            final_cap = ''
                            if args.sep_token:
                                SEP_token = " [SEP] "
                            else:
                                SEP_token = " "  # " [SEP] "
                            for c in captions:
                                final_cap = final_cap + c + SEP_token
                        else:
                            final_cap = ' '
                            if 'vg' in uid[0]:
                                print("vg")
                            if 'gqa' in uid[0]:
                                print("gqa")
                            if 'visual7w' in uid[0]:
                                print("visual7w")
                            print(img_id_cap)
                            final_cap = ' '
                            print("&&&&&&&&&")

                else:
                    cap = self.imgid2cap[img_id_cap]
                    cap_num = args.cap_num
                    # assert cap_num == 1
                    captions = cap[:cap_num]
                    final_cap = ''
                    SEP_token = " "  # " [SEP] "
                    for c in captions:
                        final_cap = final_cap + c + SEP_token
            else:
                final_cap = None
        else:
            if args.caption:
                if img_id not in self.imgid2cap:
                    final_cap = ' '
                    print("&&&&&&&&&")
                else:
                    cap = self.imgid2cap[img_id]
                    cap_num = args.cap_num
                    captions = cap[:cap_num]
                    final_cap = ''
                    SEP_token = " [SEP] "  # " "
                    for c in captions:
                        final_cap = final_cap + c + SEP_token
            else:
                final_cap = ''

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1 + 1e-5)
        np.testing.assert_array_less(-boxes, 0 + 1e-5)

        if args.denslabel:
            dens_label = ''
            for d in dens_labels:
                dens_label = dens_label + d
            final_caps = [final_cap + dens_label]
        elif args.guide_dense or args.denslabel_cap:
            # final_caps.append(final_cap)
            # final_caps.append(dens_labels)

            final_caps = [final_cap, dens_labels]
        else:
            final_caps = final_cap

        # if not args.caption:
        #     final_caps = ques

        # tokens = self.encode_question(final_caps[1])
        # final_caps[1] = tokens

        if  args.gqa:
            if img_id[0]== 'n':
                img_id = '987' + img_id[1:]
            img_id = int(img_id)
        else:
            img_id = int(img_id)
        if args.vqa_cp:
            labels = datum['labels']
            scores = datum['scores']
            target = torch.zeros(3129)
            for ans, score in zip(labels, scores):
                target[ans] = score
            return ques_id, feats, boxes, ques, final_caps, target, img_id
        if args.gqa:
            if 'label' in datum:
                label = datum['label']
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    if ans in self.raw_dataset.ans2label:
                        target[self.raw_dataset.ans2label[ans]] = score
                return ques_id, feats, boxes, ques, final_cap, target, img_id
            else:
                return ques_id, feats, boxes, ques, final_cap, img_id
        if  args.okvqa:
            target = torch.zeros(self.raw_dataset.num_answers)
            label = self.raw_dataset.id2answer[ques_id]['labels']
            scores = self.raw_dataset.id2answer[ques_id]['scores']
            for ans, score in zip (label,scores):
                target[label] = score
            datum['label'] = label
            datum['scores'] = scores
            return ques_id, feats, boxes, ques, final_cap, target
        else:
            if 'label' in datum:
                label = datum['label']
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    target[self.raw_dataset.ans2label[ans]] = score
                return ques_id, feats, boxes, ques, final_caps, target, img_id
            else:
                return ques_id, feats, boxes, ques, final_caps, img_id


class VQAEvaluator:
    def __init__(self, dataset: VQADataset):
        self.dataset = dataset
        if args.okvqa:
            self.ans2label = json.load(open("/home/myj/code/lxmert_caption/data/vqa/ok-vqa/cache/ans2label.json"))  # exist
            self.label2ans = json.load(open("/home/myj/code/lxmert_caption/data/vqa/ok-vqa/cache/label2ans.json"))

    def evaluate(self, quesid2ans: dict):
        score = 0.
        if args.okvqa:
            for quesid, ans in quesid2ans.items():
                datum = self.dataset.id2datum[quesid]

                label = datum['label']
                scores = datum['scores']
                ans = self.ans2label[ans]

                if ans in label:
                    idx = label.index(ans)
                    score += scores[idx]
            return score / len(quesid2ans)


        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]

            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


