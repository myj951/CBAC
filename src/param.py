# coding=utf-8
# Copyleft 2019 project LXRT.

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'  # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--caption", default=False)
    # parser.add_argument("--caption_model", default=False)
    test = False
    cp_test = False
    test_split = False
    # parser.add_argument("--test", default='test')
    # parser.add_argument('--load', type=str, default='/home/myj/code/lxmert_caption/snap/vqa/vqa_lxr5222_96_caption301_num5_pool1_accu1_shardl_vinvl_nonorm/BEST',

    parser.add_argument("--caption",type=bool, default=test)
    parser.add_argument("--caption_model",type=bool, default=test)
    parser.add_argument("--cap_num", type=int, default=1)
    parser.add_argument("--cap_max_len", type=int, default=15)
    parser.add_argument("--sep_token", type=bool, default=test)


    parser.add_argument("--mul_qv",type=bool, default=False)
    parser.add_argument("--mul_class",type=bool, default=False)
    parser.add_argument("--adapW",type=bool, default=False)
    parser.add_argument("--cls_dropout", type=float, default=0.)
    parser.add_argument("--singal_test", type=bool, default=False)


    parser.add_argument("--denslabel", type=bool, default=False)
    parser.add_argument("--guide_dense", type=bool, default=False)
    parser.add_argument("--denslabel_cap", type=bool, default=False)

    parser.add_argument("--loss0", type=bool, default=False)

    parser.add_argument("--box_key", default=True)

    parser.add_argument("--gener_cap", default=False)

    parser.add_argument("--vinvl", default=False)

    parser.add_argument("--caption_sem_type", default=test)
    parser.add_argument("--type_vocab_size", type=int, default=2)
    parser.add_argument("--accumulation_step", type=int, default=1)

    parser.add_argument("--vqa_cp", type=bool, default=cp_test)
    parser.add_argument("--gqa", type=bool, default=cp_test)
    parser.add_argument("--okvqa", type=bool, default=cp_test)

    parser.add_argument("--v_gloabl", default=False)
    parser.add_argument("--v_gloabl_pool", default=False)
    parser.add_argument("--v_cls", default=False)
    parser.add_argument("--q_global", default=False)
    parser.add_argument("--qc_global", default=False)
    parser.add_argument("--qv_global", default=False)
    parser.add_argument("--cv_global", default=True)




    parser.add_argument("--con_epoch", type=int, default=999)
    parser.add_argument("--con_Loss", default=False)
    parser.add_argument("--con_alpha", type=float, default=1.0)
    parser.add_argument("--l2_alpha", type=float, default=1.0)

    parser.add_argument("--a_cls", type=int, default=3)
    parser.add_argument("--pool_type", type=int, default=3)

    parser.add_argument("--opt2", default=False)

    parser.add_argument("--shared_l", default=test)
    cross = 2
    parser.add_argument("--q_num", type=int, default=5)
    parser.add_argument("--v_num", type=int, default=5)
    parser.add_argument("--c_num", type=int, default=5)
    parser.add_argument("--cq_num", type=int, default=cross)
    parser.add_argument("--cv_num", type=int, default=cross)
    parser.add_argument("--cc_num", type=int, default=cross)
    parser.add_argument("--qv_num", type=int, default=cross)

    # Data Splits

    if cp_test:
        parser.add_argument("--train", default='vqacp_v2_train')  # vqacp_v2_train
        parser.add_argument("--valid", default='vqacp_v2_test')  # vqacp_v2_test
    elif test_split:
        parser.add_argument("--train", default='train')  # vqacp_v2_train
        parser.add_argument("--valid", default='')  # vqacp_v2_test
    else:
        # parser.add_argument("--train", default='train')  # vqacp_v2_train
        # parser.add_argument("--valid", default='nominival,minival')  # vqacp_v2_test
        # parser.add_argument("--train", default='train,nominival')  # vqacp_v2_train
        # parser.add_argument("--valid", default='minival')  # vqacp_v2_test
        parser.add_argument("--train", default='train,nominival,minival')  # vqacp_v2_train
        parser.add_argument("--valid", default='')  # vqacp_v2_test
    if not test_split:
        parser.add_argument("--test", default=None)
    else:
        parser.add_argument("--test", default='test')
    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=96)
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Debugging
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=test, const=True)
    parser.add_argument("--tqdm", action='store_const', default=False, const=True)

    # Model Loading

    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--loadLXMERT', dest='load_lxmert', type=str, default=None,
                        help='Load the pre-trained LXMERT model.')
    parser.add_argument('--loadLXMERTQA', dest='load_lxmert_qa', type=str, default=None,
                        # '/home/myj/code/lxmert_caption/snap/pretrained/model',
                        help='Load the pre-trained LXMERT model with QA answer head.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load, --loadLXMERT, --loadLXMERTQA is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)

    # LXRT Model Config
    # Note: LXRT = L, X, R (three encoders), Transformer
    parser.add_argument("--llayers", default=9, type=int, help='Number of Language layers')
    parser.add_argument("--xlayers", default=5, type=int, help='Number of CROSS-modality layers.')
    parser.add_argument("--rlayers", default=5, type=int, help='Number of object Relationship layers.')

    # LXMERT Pre-training Config
    parser.add_argument("--taskMatched", dest='task_matched', action='store_const', default=False, const=True)
    parser.add_argument("--taskMaskLM", dest='task_mask_lm', action='store_const', default=False, const=True)
    parser.add_argument("--taskObjPredict", dest='task_obj_predict', action='store_const', default=False, const=True)
    parser.add_argument("--taskQA", dest='task_qa', action='store_const', default=False, const=True)
    parser.add_argument("--visualLosses", dest='visual_losses', default='obj,attr,feat', type=str)
    parser.add_argument("--qaSets", dest='qa_sets', default=None, type=str)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate', default=0.15, type=float)

    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Parse the arguments.8
    args = parser.parse_args()

    args.batch_size = 96# 256r
    # args.load_lxmert = '/home/myj/code/lxmert_caption/snap/pre_gqa_gener_num2_pool1_shardl_cls1_vcls_8_5e-5/BEST_EVAL_LOSS'
    args.load_lxmert = '/home/myj/code/lxmert_caption/snap/pre_gqa_gener_8_1e-4/BEST_EVAL_LOSS'
    # args.load_lxmert = '/home/myj/code/lxmert_caption/snap/pre_vqa_gener_8_1e-4/BEST_EVAL_LOSS'

    args.epochs = 4
    args.lr = 5e-5#5e-5

    #
    args.vqa_cp = True
    args.train = 'vqacp_v2_train'
    args.valid = 'vqacp_v2_test'

    # args.gqa = True
    # args.train = 'train,valid'
    # args.valid = 'testdev'

    # args.okvqa = True
    # args.train = 'train'
    # args.valid = 'valid'

    args.tiny= False  #True, False


    args.denslabel = False
    args.guide_dense = False
    args.denslabel_cap = False
    args.sep_token = False
    #
    args.llayers = 9
    args.xlayers = 5
    args.rlayers = 2

    args.q_num = 9
    args.v_num = 1
    args.qv_num = 5
    args.cq_num = 2
    args.cc_num = 2
    args.cv_num = 2

    args.fromScratch= False

    args.caption_model = True
    args.gener_cap = True
    args.caption = True
    args.shared_l= True
    args.caption_sem_type= False
    args.cap_num = 3 #1,5, 2
    # args.output = '/home/myj/code/lxmert_caption/snap/vqa_gener_num2_pool1_shardl_cls1_vcls_denselabel'
    # args.output = '/home/myj/code/lxmert_caption/snap/vqa_gener_num2_pool1_shardl_cls2_adapW_vcls_loss0_conlossC1.0'
    args.output = '/home/myj/code/lxmert_caption/snap/vqa_gener_num2_pool1_shardl_cls1_vcls_conloss0.3C1.0'

    args.vair = True
    args.add = False

    args.v_cls = True
    args.con_Loss = True

    args.caption_fusion = True

    args.con_alpha = 0.5 #5-0.25
    args.l2_alpha = 0.45 #010.2
    args.q_global =  True

    args.cv_global = False
    args.qv_global = False
    args.qc_global = False
    #
    #########################################
    args.pool_type = -1 #-1--no mul_cls , 1 mul_cls
    args.a_cls = 1
    args.adapW = False
    args.mul_class = False
    args.loss0 = False

    # args.pool_type = 1 #-1--no mul_cls , 1 mul_cls
    # args.a_cls = 2
    # args.adapW = True
    # args.mul_class = True
    # args.loss0 = True



    #########################################
    # args.test = 'test'  #test
    # args.train = 'minival'
    # args.valid = ''
    #
    # # args.test = 'testdev'
    # # args.valid = 'testdev'
    # # args.valid = ''
    #
    # args.tiny = True
    # # args.load = '/home/myj/code/lxmert_caption/snap/vqa_gener_num2_pool1_shardl_cls2_adapW_vcls_loss0/BEST'
    # # args.load = '/home/myj/code/lxmert_caption/snap/vqa_gener_num2_pool1_shardl_cls1_vcls_conloss0.3C1.0/LAST'
    # args.load = '/home/myj/code/lxmert_caption/snap/vqa_gener_num2_pool1_shardl_cls1_vcls_conloss0.3C1.0/test2'


    #################################################
    # args.task_matched= True
    # args.task_mask_lm = True
    # args.taskObjPredict = True
    # args.task_qa = True
    # args.load_lxmert_qa = '/home/myj/code/lxmert_caption/snap/pretrain/TEST'
    # args.tiny = True
    # args.caption = True
    # args.caption_model = True
    # args.fromScratch= False
    # args.pool_type =1
    # args.v_cls =True
    # args.shared_l=True
    # args.caption_sem_type=True
    # if args.vinvl:
    #     args.visual_losses = 'obj,feat'
    # else:
    #     args.visual_losses = 'obj,attr,feat'
    # assert  args.vinvl !=( 'attr' in args.visual_losses)
    # args.a_cls = 2
    # args.con_Loss = True
    # args.cap_num =2
    # args.gener_cap = True

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
