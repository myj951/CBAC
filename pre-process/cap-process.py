import json


capt_val =  json.load(open("/home/myj/data/frcnn/COCO/annotations/captions_val2014.json"))
capt_train =  json.load(open("/home/myj/data/frcnn/COCO/annotations/captions_train2014.json"))
# json.load(open("/home/myj/data/frcnn/COCO/annotations/stuff_val2017.json"))

captions = capt_train
coco = 'COCO_train2014_' #COCO_val2014_ COCO_train2014_

cap_dict = {}
for caption in captions['annotations']:
    anas = caption

    image_id = anas['image_id']

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

with open('/home/myj/data/frcnn/COCO/annotations/pre-process/cp_train_captions.json','w',encoding='utf-8') as file:
    file.write(json.dumps(cap_dict, indent=2, ensure_ascii=False))

# with open('/home/myj/data/frcnn/COCO/annotations/pre-process/cp_val_captions.json','w',encoding='utf-8') as file:
#     file.write(json.dumps(cap_dict, indent=2, ensure_ascii=False))

# test = json.load(open('/home/myj/data/frcnn/COCO/annotations/pre-process/val_captions.json'))
