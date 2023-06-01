import json

capt_val =  json.load(open("/home/myj/3.7Tfile/data/ofa/OFA-main/results/caption/val.json"))
capt_train =  json.load(open("/home/myj/3.7Tfile/data/ofa/OFA-main/results/caption/train.json"))
capt_test =  json.load(open("/home/myj/3.7Tfile/data/ofa/OFA-main/results/caption/test.json"))
vg_capt_train =  json.load(open("/home/myj/3.7Tfile/data/ofa/OFA-main/results/caption/vg_train.json"))
vg_capt_val =  json.load(open("/home/myj/3.7Tfile/data/ofa/OFA-main/results/caption/vg_val.json"))
gqa_capt1 =  json.load(open("/home/myj/3.7Tfile/data/ofa/OFA-main/results/caption/generate_gqa1_captions.json"))
gqa_capt2 =  json.load(open("/home/myj/3.7Tfile/data/ofa/OFA-main/results/caption/generate_gqa2_captions.json"))
gqa_cap_all = []

gqa_cap_all.extend(gqa_capt1)
gqa_cap_all.extend(gqa_capt2)

captions = gqa_cap_all
coco = 'COCO_train2014_' #COCO_val2014_ COCO_train2014_


cap_dict = {}
for caption in captions:
    anas = caption

    image_id = anas['image_id'][:-2]
    #coco
    # image_id = str(int(image_id.split("_")[-1]))
    #vg
    image_id = image_id

    # image_id = str(anas['image_id'])
    # padding_len = 12-len(image_id)
    # padding = '0' * padding_len
    # image_id = coco + padding + image_id

    if image_id not in cap_dict:
        # if image_id == 'COCO_val2014_000000179765':
        #     print('test')

        if anas['caption'] != None:
            cap_dict[image_id] = [anas['caption']]
        else:
            print('no caption')
            continue
    else:
        cap_dict[image_id].append(anas['caption'])

with open('/home/myj/3.7Tfile/data/lxmert_caption/generate_gqa_captions.json','w',encoding='utf-8') as file:
    file.write(json.dumps(cap_dict, indent=2, ensure_ascii=False))

# with open('/home/myj/data/frcnn/COCO/annotations/pre-process/cp_val_captions.json','w',encoding='utf-8') as file:
#     file.write(json.dumps(cap_dict, indent=2, ensure_ascii=False))

# test = json.load(open('/home/myj/3.7Tfile/data/lxmert_caption/generate_test_captions.json'))
