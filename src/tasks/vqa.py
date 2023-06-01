# coding=utf-8
# Copyleft 2019 project LXRT.

import sys

sys.path.append("/home/myj/code/lxmert_caption/src")

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args

import numpy as np


#################################################


from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator

from Loss import ContrastiveLoss
from Loss import ContrastiveLoss3

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
import torch.optim as optim


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets

        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=256,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        # self.con_loss1 = ContrastiveLossELI5(args.batch_size)
        self.con_loss2 = ContrastiveLoss(measure='l2', margin=args.l2_alpha)  # 0.2,#0.3
        # self.con_loss2 = ContrastiveLoss(measure='dot', margin=args.l2_alpha)
        # self.con_loss2 = ContrastiveLoss3(args.batch_size,temperature =args.l2_alpha)
        self.bce_loss = nn.BCEWithLogitsLoss()

        if 'bert' in args.optim:
            if args.caption_model:
                batch_per_epoch = int(len(self.train_tuple.loader) / args.accumulation_step) + 1
            else:
                batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        if args.opt2:
            self.optim2 = optim.Adamax(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)

            # self.optim2 = optim.Adamax(self.model.parameters() ,lr = args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.02)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        print("args.mul_class:{}".format(args.mul_class))
        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}



            for i, (ques_id, feats, boxes, sent, caption, target,img_id) in iter_wrapper(enumerate(loader)):

                array = img_id.numpy()
                matrix = (array[:, np.newaxis] == array).astype(int)
                np.fill_diagonal(matrix, False)

                self.model.train()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()

                # import numpy as np
                # # for i in zip(feats[0][0], boxes[0][0], target[0][0],caption[0][0]):
                # #     print(np.array(i).shape)
                # #     # print(i)
                # for i,j,k,l in zip(feats.cpu()[0], boxes.cpu()[0], sent[0],caption[0]):
                #     print(np.array(i).shape)
                #     print(np.array(j).shape)
                #     print(np.array(k).shape)
                #     print(np.array(l).shape)
                #     break
                #     # print(i)
                # return 0

                if args.caption_model:
                    if args.mul_class:
                        if args.a_cls == 3:
                            logit, logit1, logit2, logit3, v_global, c_gloabl, q_gloabl = self.model(feats, boxes, sent,
                                                                                                     caption)
                            # loss0 = self.bce_loss(logit, target)* logit.size(1)
                            loss0 = 0.1 * self.bce_loss(logit, target) * logit.size(1)
                            loss1 = self.bce_loss(logit1, target) * logit.size(1)
                            loss2 = self.bce_loss(logit2, target) * logit.size(1)
                            loss3 = self.bce_loss(logit3, target) * logit.size(1)
                            loss = loss0 + loss1 + loss2 + loss3
                        else:
                            logit, logit1, logit2, v_global, c_gloabl, q_gloabl = self.model(feats, boxes, sent,
                                                                                             caption)
                            # loss0 = self.bce_loss(logit, target) * logit.size(1)
                            loss0 = 0.1 * self.bce_loss(logit, target) * logit.size(1)
                            loss1 = self.bce_loss(logit1, target) * logit.size(1)
                            loss2 = self.bce_loss(logit2, target) * logit.size(1)
                            loss3 = 0
                            loss = loss0 + loss1 + loss2
                            if args.loss0:
                                loss = loss0 * 10
                            else:
                                loss = loss
                    else:
                        logit, v_global, c_gloabl, q_gloabl = self.model(feats, boxes, sent, caption)
                        # logit = self.model(feats, boxes, sent, )
                        loss = self.bce_loss(logit, target)
                        loss = loss * logit.size(1)
                else:
                    logit = self.model(feats, boxes, sent, )
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)
                assert logit.dim() == target.dim() == 2


                    # if args.con_Loss and epoch>0:
                if args.con_Loss and epoch <= args.con_epoch:


                    # l = self.con_loss1(v_global, c_gloabl)
                    # l_c = self.con_loss2(v_global, c_gloabl)
                    # l_c = self.con_loss3(v_global, c_gloabl)

                    if args.cv_global:
                        l_c = self.con_loss2(v_global, c_gloabl,matrix)
                        l = l_c
                    else:
                        l_c = 0
                        l = 0

                    if args.qv_global:
                        l_q = self.con_loss2(v_global, q_gloabl,matrix)
                        l = l + l_q
                    else:
                        l_q = 0

                    if args.qc_global:
                        l_qc = self.con_loss2(c_gloabl, q_gloabl,matrix)
                        l = l + l_qc
                        if i < 2:
                            print("loss1:{}   loss2:{} ".format(loss, l_qc))

                    if args.loss0:
                        loss = loss0*10 + l * args.con_alpha
                    else:
                        loss = loss + l * args.con_alpha

                    assert args.v_gloabl_pool or args.v_gloabl or args.v_cls
                    if i % 100 ==0:
                        print("loss1:{}   loss2:{}   loss3:{}".format(loss, l_c, l_q))

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)

                if args.caption_model:
                    if i % args.accumulation_step == 0:
                        if args.opt2:
                            self.optim2.step()
                            self.optim2.zero_grad()
                        else:
                            self.optim.step()
                            self.optim.zero_grad()
                else:
                    if args.opt2:
                        self.optim2.step()
                        self.optim2.zero_grad()
                    else:
                        self.optim.step()
                        self.optim.zero_grad()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    if type(qid) == str:
                        quesid2ans[qid] = ans
                    else:
                        quesid2ans[qid.item()] = ans

            if args.mul_class:
                log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.) + \
                          "loss: %0.2f ; %0.2f ; %0.2f; %0.2f %0.2f;\n" % (loss, loss0, loss1, loss2, loss3,)
            else:
                log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.) + \
                          "loss: %0.2f  ;\n" % (loss)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")
                if args.mul_class:
                    log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                               "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.) + \
                               "loss: %0.2f ; %0.2f ; %0.2f; %0.2f %0.2f;\n" % (loss, loss0, loss1, loss2, loss3,)
                else:
                    log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                               "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.) + \
                               "loss: %0.2f ;\n" % (loss,)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple

        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            if args.caption_model:
                ques_id, feats, boxes, sent, caption, img_id = datum_tuple[:6]
            else:  # Avoid seeing ground truth
                ques_id, feats, boxes, sent, img_id = datum_tuple[:5]
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()

                if args.caption_model:
                    if args.mul_class:
                        if args.a_cls == 3:
                            logit, logit1, logit2, logit3, v_global, c_gloabl, q_gloabl = self.model(feats, boxes, sent,
                                                                                                     caption)
                        else:
                            logit, logit1, logit2, v_global, c_gloabl, q_gloabl = self.model(feats, boxes, sent,
                                                                                             caption)
                    else:
                        logit, v_global, c_gloabl, q_gloabl = self.model(feats, boxes, sent, caption)
                else:
                    logit = self.model(feats, boxes, sent)
                if args.singal_test and args.mul_class:
                    score, label = logit1.max(1)
                else:
                    score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    if type(qid) == str:
                        quesid2ans[qid] = ans
                    else:
                        quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, caption, target, img_id) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                if qid == '201175144':
                    print("")
                ans = dset.label2ans[l]
                if type(qid) == str:
                    quesid2ans[qid] = ans
                else:
                    quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict,strict=False)


if __name__ == "__main__":
    # Build Class


    print("args.output:{}".format(args.output))
    print("args.v_num:{}".format(args.mul_class))
    print("args.tiny:{}".format(args.tiny))
    print("args.adapW:{}".format(args.adapW))
    print("args.con_Loss:{}".format(args.con_Loss))
    print("args.con_alpha:{}".format(args.con_alpha))
    print("args.guide_dense:{}".format(args.guide_dense))
    print("args.mul_class:{}".format(args.mul_class))
    print("args.loss0:{}".format(args.loss0))
    vqa = VQA()
    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.gqa:
        if args.test is not None:
            args.fast = args.tiny = False  # Always loading all data in test
            if 'submit' in args.test:
                vqa.predict(
                    get_data_tuple(args.test, bs=256,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'test_predict.json')
                )
            if 'testdev' in args.test:
                # Since part of valididation data are used in pre-training/fine-tuning,
                # only validate on the minival set.
                result = vqa.evaluate(
                    get_data_tuple('testdev', bs=512,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'testdev_predict.json')
                )
                print(result)
            elif 'valid' not in args.test:
                # Since part of valididation data are used in pre-training/fine-tuning,
                # only validate on the minival set.
                result = vqa.evaluate(
                    get_data_tuple('valid', bs=256,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'valid_predict.json')
                )
                print(result)
            else:
                assert False, "No such test option for %s" % args.test
        else:
            print('Splits in Train data:', vqa.train_tuple.dataset.splits)
            if vqa.valid_tuple is not None:
                # if args.gqa:
                #     print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
                #     print("Valid Oracle: %0.2f" % (gqa.oracle_score(vqa.valid_tuple) * 100))
                # else:
                #     print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
                #     print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
                print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
                print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
            else:
                print("DO NOT USE VALIDATION")
            vqa.train(vqa.train_tuple, vqa.valid_tuple)
    else:
        if args.test is not None:
            args.fast = args.tiny = False  # Always loading all data in test
            if 'test' in args.test:
                vqa.predict(
                    get_data_tuple(args.test, bs=512,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'test_predict.json')
                )
            elif 'nominival' in args.test:
                # Since part of valididation data are used in pre-training/fine-tuning,
                # only validate on the minival set.
                result = vqa.evaluate(
                    get_data_tuple('minival,nominival', bs=512,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'minival_predict.json')
                )
                print(result)
            elif 'minival' not in args.test:
                # Since part of valididation data are used in pre-training/fine-tuning,
                # only validate on the minival set.
                result = vqa.evaluate(
                    get_data_tuple('minival', bs=512,
                                   shuffle=False, drop_last=False),
                    dump=os.path.join(args.output, 'minival_predict.json')
                )
                print(result)
            else:
                assert False, "No such test option for %s" % args.test
        else:
            print('Splits in Train data:', vqa.train_tuple.dataset.splits)
            if vqa.valid_tuple is not None:
                # if args.gqa:
                #     print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
                #     print("Valid Oracle: %0.2f" % (gqa.oracle_score(vqa.valid_tuple) * 100))
                # else:
                #     print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
                #     print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
                print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
                print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
            else:
                print("DO NOT USE VALIDATION")
            vqa.train(vqa.train_tuple, vqa.valid_tuple)


