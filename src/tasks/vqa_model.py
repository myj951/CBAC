# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder,LXRTEncoderCaption
from lxrt.modeling import BertLayerNorm, GeLU
import param
# Max length including <bos> and <eos>
MAX_VQA_LENGTH = 20
import  torch


if args.caption_model:
    class VQAModel(nn.Module):
        def __init__(self, num_answers):
            super().__init__()

            # Build LXRT encoder
            self.lxrt_encoder = LXRTEncoder(
                args,
                max_seq_length=MAX_VQA_LENGTH
            )
            hid_dim = self.lxrt_encoder.dim

            # VQA Answer heads
            if param.args.mul_class:
                self.logit_fc1 = nn.Sequential(
                    nn.Linear(hid_dim, hid_dim * 2),
                    GeLU(),
                    BertLayerNorm(hid_dim * 2, eps=1e-12),
                    nn.Linear(hid_dim * 2, num_answers)
                )
                self.logit_fc1.apply(self.lxrt_encoder.model.init_bert_weights)
                self.logit_fc2 = nn.Sequential(
                    nn.Linear(hid_dim, hid_dim * 2),
                    GeLU(),
                    BertLayerNorm(hid_dim * 2, eps=1e-12),
                    nn.Dropout(args.cls_dropout, inplace=True),
                    nn.Linear(hid_dim * 2, num_answers)
                )
                self.logit_fc2.apply(self.lxrt_encoder.model.init_bert_weights)
                if args.a_cls==3:
                    self.logit_fc3 = nn.Sequential(
                        nn.Linear(hid_dim, hid_dim * 2),
                        GeLU(),
                        BertLayerNorm(hid_dim * 2, eps=1e-12),
                        nn.Dropout(args.cls_dropout, inplace=True),
                        nn.Linear(hid_dim * 2, num_answers)
                    )
                    self.logit_fc3.apply(self.lxrt_encoder.model.init_bert_weights)
                if args.adapW:
                    self.adapted_w = nn.Parameter(torch.ones(args.a_cls, num_answers))
                    self.key =True

            else:
                self.logit_fc = nn.Sequential(
                    nn.Linear(hid_dim, hid_dim * 2),
                    GeLU(),
                    BertLayerNorm(hid_dim * 2, eps=1e-12),
                    nn.Dropout(args.cls_dropout, inplace=True),
                    nn.Linear(hid_dim * 2, num_answers)
                )
                self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)


        def forward(self, feat, pos, sent, caption = None):
        # def forward(self, inuput):
        #     sent = ['Does the woman have any underwear on?']
        #     caption =['a woman in a white dress holding a tennis racquet [SEP] a woman in a white dress holding a tennis racket [SEP] ']
        #     pos = torch.randn((1, 36, 4)).to('cuda')
        #     feat= inuput
            """
            b -- batch_size, o -- object_number, f -- visual_feature_size

            :param feat: (b, o, f)
            :param pos:  (b, o, 4)
            :param sent: (b,) Type -- list of string
            :param leng: (b,) Type -- int numpy array
            :return: (b, num_answer) The logit of each answers.
            """
            if not param.args.caption_model:
                x = self.lxrt_encoder(sent, (feat, pos), caption)
                v_global = None
                c_gloabl = None
                # print(v_global)
                # print(c_gloabl)
                logit = self.logit_fc(x)

                return logit

            else:
                if param.args.mul_class:
                    if args.a_cls == 3:
                        x,xl,xv,v_global, c_gloabl, q_gloabl = self.lxrt_encoder(sent, (feat, pos), caption)
                    else:
                        x, xl, v_global, c_gloabl, q_gloabl = self.lxrt_encoder(sent, (feat, pos), caption)
                    logit1 = self.logit_fc1(x)
                    logit2 = self.logit_fc2(xl)
                    if args.a_cls==3:
                        logit3 = self.logit_fc3(xv)
                        if args.adapW:
                            adapted_w = torch.softmax(self.adapted_w, 0)
                            logits = torch.cat([logit1.unsqueeze(1), logit2.unsqueeze(1), logit3.unsqueeze(1)], 1)
                            logit = torch.mul(logits, adapted_w.unsqueeze(0)).sum(1)
                        else:
                            logit = logit1 + logit2 + logit3

                        return logit, logit1, logit2, logit3, v_global, c_gloabl, q_gloabl
                    else:

                        if args.adapW:
                            if self.key:
                                print(self.adapted_w)
                                self.key =False
                            adapted_w = torch.softmax(self.adapted_w, 0)
                            logits = torch.cat([logit1.unsqueeze(1), logit2.unsqueeze(1)], 1)
                            logit = torch.mul(logits, adapted_w.unsqueeze(0)).sum(1)
                        else:
                            logit = logit1 + logit2

                        return logit, logit1, logit2, v_global, c_gloabl, q_gloabl
                        # index = torch.arange(1)
                        # return logit[index].squeeze()
                        # return logit
                else:
                    x, v_global, c_gloabl, q_gloabl = self.lxrt_encoder(sent, (feat, pos), caption)
                    logit = self.logit_fc(x)



                    return logit,v_global, c_gloabl, q_gloabl

elif args.caption:
    class VQAModel(nn.Module):
        def __init__(self, num_answers):
            super().__init__()
            # Build LXRT encoder
            self.lxrt_encoder = LXRTEncoder(  # can get load in the vqa.py
                args,
                max_seq_length=MAX_VQA_LENGTH
            )
            hid_dim = self.lxrt_encoder.dim

            # VQA Answer heads
            self.logit_fc = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_answers)
            )
            # initial weight when starting
            # the CE loss here for obj_label == obj_label*obj_conf-- to make the model get more close to the right obj_information
            self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        def forward(self, feat, pos, sent, caption=None):
            """
            b -- batch_size, o -- object_number, f -- visual_feature_size

            :param feat: (b, o, f)
            :param pos:  (b, o, 4)#4--box
            :param sent: (b,) Type -- list of string
            :param leng: (b,) Type -- int numpy array
            :return: (b, num_answer) The logit of each answers.
            """
            x = self.lxrt_encoder(sent, (feat, pos),caption
                                )  # the final feature x-[batch_size:256, dim:768]; x has three model;
            logit = self.logit_fc(x)  # 2 layer FC with GeLu_action

            return logit
else:
    class VQAModel(nn.Module):
        def __init__(self, num_answers):
            super().__init__()
            # Build LXRT encoder
            self.lxrt_encoder = LXRTEncoder(  # can get load in the vqa.py
                args,
                max_seq_length=MAX_VQA_LENGTH
            )
            hid_dim = self.lxrt_encoder.dim

            # VQA Answer heads
            self.logit_fc = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_answers)
            )
            # initial weight when starting
            # the CE loss here for obj_label == obj_label*obj_conf-- to make the model get more close to the right obj_information
            self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        def forward(self, feat, pos, sent):
            """
            b -- batch_size, o -- object_number, f -- visual_feature_size

            :param feat: (b, o, f)
            :param pos:  (b, o, 4)#4--box
            :param sent: (b,) Type -- list of string
            :param leng: (b,) Type -- int numpy array
            :return: (b, num_answer) The logit of each answers.
            """
            x = self.lxrt_encoder(sent, (feat, pos),
                                )  # the final feature x-[batch_size:256, dim:768]; x has three model;
            logit = self.logit_fc(x)  # 2 layer FC with GeLu_action

            return logit


class VQAModelCaption(nn.Module):
        def __init__(self, num_answers):
            super().__init__()
            # Build LXRT encoder
            self.lxmer = VQAModel(
                num_answers
            )
            self.tokenCaption = LXRTEncoderCaption()

        def forward(self, feat, pos, sent, caption):
            """
            b -- batch_size, o -- object_number, f -- visual_feature_size

            :param feat: (b, o, f)
            :param pos:  (b, o, 4)#4--box
            :param sent: (b,) Type -- list of string
            :param leng: (b,) Type -- int numpy array
            :return: (b, num_answer) The logit of each answers.
            """
            logit = self.lxmert(feat, pos, sent)  # the final feature x-[batch_size:256, dim:768]; x has three model;


            return logit