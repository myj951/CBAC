# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os

import torch
import torch.nn as nn

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG
from lxrt.modeling import BertEmbeddings
from param import args


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features_cap(sents, max_seq_length=None, tokenizer=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


def convert_sents_to_features_concat(sents, max_seq_length=None, tokenizer=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        idx1 = tokens.index('[SEP]') + 1
        # idx2 = tokens[idx1:].index('[SEP]') +idx1 +1
        # idx3 = tokens[idx2:].index('[SEP]')+idx2+1

        # idx4 = tokens[idx3:].index('[SEP]') + idx3 +1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if args.caption_sem_type:
            segment_ids = [1] * len(input_ids)
            semgment_padding = [1] * (max_seq_length - len(input_ids))
        else:
            segment_ids = [0] * len(input_ids)
            semgment_padding = [0] * (max_seq_length - len(input_ids))
        # segment_ids = [1] * idx1 + [2] * (idx2-idx1) + [3] * (idx3-idx2) + [4] * (len(tokens)-idx3)

        # segment_ids = [0] * idx1 + [1] * (idx2 - idx1) + [2] * (idx3 - idx2) + [3] * (len(tokens) - idx3)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += semgment_padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


class BertEmbeddings_dense(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings_dense, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        BertLayerNorm = torch.nn.LayerNorm
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        # position_embeddings = self.position_embeddings(position_ids)
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertAttention_denselabel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048

        ctx_dim = 3840
        self.query = nn.Linear(3840, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        batch = hidden_states.shape[0]
        try:
            mixed_query_layer = self.query(hidden_states[:, :-1, :])
        except:
            print(" ")

        mixed_key_layer = self.key(hidden_states[:, -1, :]).flatten().reshape(batch, 1, -1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        attention_scores = attention_scores.permute(0, 2, 1, 3).flatten().reshape(batch, 36, 12)
        attention_scores = attention_scores.sum(-1)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        return attention_probs


def convert_denselabel_to_features(sents, max_seq_length=None, tokenizer=None):
    """Loads a data file into a list of `InputBatch`s."""
    max_seq_length = 5
    features = []
    for (i, sentss) in enumerate(sents):
        input_idss = []
        input_maskss = []
        segment_idss = []
        for sent in sentss:
            tokens_a = tokenizer.tokenize(sent.strip())

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            # Keep segment id which allows loading BERT-weights.
            tokens = tokens_a
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.

            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            input_idss.append(input_ids)
            input_maskss.append(input_mask)
            segment_idss.append(segment_ids)

        features.append(
            InputFeatures(input_ids=input_idss,
                          input_mask=input_maskss,
                          segment_ids=segment_idss))
    return features


def convert_sents_to_features(sents, max_seq_length=None, tokenizer=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))
    return features


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers


class LXRTEncoderCaption(nn.Module):
    def __init__(self, ):
        super().__init__()
        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            # '/home/myj/code/lxmert-caption/snap/bert-base-uncased-vocab.txt',
            do_lower_case=True
        )

    @property
    def dim(self):
        return 768

    def forward(self, caption, ):
        train_features2 = convert_sents_to_features_cap(
            caption, args.cap_max_len * args.cap_num, self.tokenizer)

        cap_input_ids = torch.tensor([f.input_ids for f in train_features2], dtype=torch.long).cuda()
        cap_input_mask = torch.tensor([f.input_mask for f in train_features2], dtype=torch.long).cuda()
        cap_segment_ids = torch.tensor([f.segment_ids for f in train_features2], dtype=torch.long).cuda()

        return cap_input_ids, cap_segment_ids, cap_input_mask,

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, init_emb, freeze=False, dropout=0.0):
        super(WordEmbedding, self).__init__()
        weights = torch.from_numpy(init_emb)
        self.emb = nn.Embedding.from_pretrained(weights, freeze=freeze) # padding_idx= ntoken
        ntoken, emb_dim = weights.shape
        self.emb = nn.Embedding(ntoken, emb_dim, padding_idx=-1)
        self.emb.weight.data = weights
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.emb(self.drop(x))

import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from src.lxrt.fc import FCNet

class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, glimpses=1, dropout=0.2):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        self.v_proj = FCNet([v_dim, hid_dim], dropout)
        self.q_proj = FCNet([q_dim, hid_dim], dropout)
        self.drop = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hid_dim, glimpses), dim=None)
    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_proj = self.v_proj(v)  # [batch, k, vdim]
        q_proj = self.q_proj(q).unsqueeze(1)# [batch, 1, qdim]

        # logits = self.linear(self.drop(v_proj * q_proj))
        # return nn.functional.softmax(logits, 1), logits

        attention_scores = torch.matmul(v_proj, q_proj.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hid_dim)
        attention_scores = attention_scores.view(-1,36)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        return attention_probs, attention_scores

class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x'):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(args)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            # "bert-base-uncased",
            '/home/myj/code/lxmert_caption/snap/bert-base-uncased-vocab.txt',
            do_lower_case=True
        )

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            # '/home/myj/code/lxmert-caption/snap/bert-base-uncased-vocab.txt',
            "bert-base-uncased",
            mode=mode
        )

        self.embeddings_dense = BertEmbeddings_dense(self.model.config)

        self.atten_dense = BertAttention_denselabel(self.model.config)
        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

        # import numpy as np
        # embeddings  = np.load(os.path.join('/home/myj/code/bottom-up-attention-vqa-master/tools', 'glove6b_init_300d.npy'))
        # self.w_emb = WordEmbedding(
        # embeddings,
        # freeze=False,
        # dropout=0.0
        #     )
        # self.Attention = Attention(1200, 1200, 768, glimpses=1, dropout=0.2)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, caption=None, visual_attention_mask=None):

        if args.guide_dense or args.denslabel_cap:
            captions = caption
            if args.caption:
                caption = captions[0]
            if captions[1][0][0][0].__class__ !=torch.Tensor:
                dense_label = captions[1]
                dense_label = list(map(list, zip(*dense_label)))

                batch = dense_label.__len__()
                label_features = convert_denselabel_to_features(
                    dense_label, self.max_seq_length, self.tokenizer)

                label_input_ids = torch.tensor([f.input_ids for f in label_features], dtype=torch.long).cuda()
                label_input_mask = torch.tensor([f.input_mask for f in label_features], dtype=torch.long).cuda()
                label_segment_ids = torch.tensor([f.segment_ids for f in label_features], dtype=torch.long).cuda()

                if label_input_mask is None:
                    label_input_mask = torch.ones_like(label_input_ids)
                if label_segment_ids is None:
                    label_segment_ids = torch.zeros_like(label_input_ids)

                label_input_ids = label_input_ids.flatten().reshape(-1, 5)

                embedding_output = self.embeddings_dense(label_input_ids, label_segment_ids)

                embedding_output = embedding_output.flatten().reshape(batch, 37, -1)
                try:
                    assert embedding_output.shape[-1] == 3840
                except:
                    print(" ")
                attention_score = self.atten_dense(embedding_output)
                embedding_output = embedding_output[:, :-1, :]

                topk_dense = attention_score.topk(k=10, dim=1).indices
                filter_dense = []
                # if args.denslabel_cap:
                #     for topk,dens in zip(topk_dense,dense_label):
                #         filter_dense_ = dens[topk]
                #         filter_dense.append(filter_dense_)
            else:
                dense_label = captions[1].cuda()
                dense_label = self.w_emb(dense_label)

                # dense_label = dense_label.view(dense_label.shape[0], 37, -1)
                # attention_score,_ = self.Attention(dense_label[:,:-1,:],dense_label[:,-1,:])

                embedding_output = dense_label

        else:
            embedding_output = None,
            attention_score = None
            # caption = caption[0]

        if not args.caption:
            train_features = convert_sents_to_features(
                sents, self.max_seq_length, self.tokenizer)
            input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
            input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
            segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

            output = self.model(input_ids, segment_ids, input_mask,
                                visual_feats=feats,
                                visual_attention_mask=visual_attention_mask,
                                entitle=embedding_output,
                                guide_atten=attention_score
                                )
            return output
        else:
            if not args.caption_model:
                # type1
                list_tmp = []
                for s, c in zip(sents, caption):
                    list_tmp.append(s + " " + "[SEP]" + " " + c)
                sents = list_tmp
                train_features = convert_sents_to_features_concat(
                    sents, self.max_seq_length + args.cap_max_len, self.tokenizer)
                input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
                input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
                segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

                output = self.model(input_ids, segment_ids, input_mask,
                                    visual_feats=feats,
                                    visual_attention_mask=visual_attention_mask)
                return output
            else:
                # type2
                train_features1 = convert_sents_to_features(
                    sents, self.max_seq_length, self.tokenizer)
                if args.denslabel:
                    cap_max_len = args.cap_max_len * args.cap_num + 70
                    caption = caption[0]
                else:
                    cap_max_len = args.cap_max_len * args.cap_num
                train_features2 = convert_sents_to_features_cap(
                    caption, cap_max_len, self.tokenizer)
                input_ids = torch.tensor([f.input_ids for f in train_features1], dtype=torch.long).cuda()
                input_mask = torch.tensor([f.input_mask for f in train_features1], dtype=torch.long).cuda()
                segment_ids = torch.tensor([f.segment_ids for f in train_features1], dtype=torch.long).cuda()

                cap_input_ids = torch.tensor([f.input_ids for f in train_features2], dtype=torch.long).cuda()
                cap_input_mask = torch.tensor([f.input_mask for f in train_features2], dtype=torch.long).cuda()
                cap_segment_ids = torch.tensor([f.segment_ids for f in train_features2], dtype=torch.long).cuda()

                if args.mul_class:
                    if args.a_cls == 3:
                        pooled_output, pooled_outputl, pooled_outputv, v_global, c_gloabl, q_gloabl = self.model(
                            input_ids, segment_ids, input_mask,
                            cap_input_ids, cap_segment_ids, cap_input_mask,
                            visual_feats=feats,
                            visual_attention_mask=visual_attention_mask,
                            entitle=embedding_output,
                            guide_atten=attention_score
                            )
                        return pooled_output, pooled_outputl, pooled_outputv, v_global, c_gloabl, q_gloabl
                    else:
                        pooled_output, pooled_outputl, v_global, c_gloabl, q_gloabl = self.model(input_ids, segment_ids,
                                                                                                 input_mask,
                                                                                                 cap_input_ids,
                                                                                                 cap_segment_ids,
                                                                                                 cap_input_mask,
                                                                                                 visual_feats=feats,
                                                                                                 visual_attention_mask=visual_attention_mask,
                                                                                                 entitle=embedding_output,
                                                                                                 guide_atten=attention_score
                                                                                                 )
                        return pooled_output, pooled_outputl, v_global, c_gloabl, q_gloabl
                else:
                    output, v_global, c_gloabl, q_gloabl = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                                                      cap_input_ids=cap_input_ids, cap_token_type_ids=cap_segment_ids, cap_attention_mask = cap_input_mask,
                                                                      visual_feats=feats,
                                                                      visual_attention_mask=visual_attention_mask,
                                                                      entitle=embedding_output,
                                                                      guide_atten=attention_score
                                                                      )
                    return output, v_global, c_gloabl, q_gloabl

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)




