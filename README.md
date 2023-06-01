# CBAC



## 目录
- [简介](#简介)
- [论文摘要](#论文摘要)
- [环境配置](#环境配置)
- [使用说明](#使用说明)
- [示例代码](#示例代码)
- [贡献](#贡献)
- [许可证](#许可证)

## 简介
PyTorch code for ["Bridging the Cross-Modality Semantic Gap in Visual Question Answering"].

## 论文摘要
The objective of visual question answering (VQA) is to adequately comprehend a question and identify relevant contents in an image that can provide an answer.Existing approaches in VQA often combine visual and question features directly to create a unified cross-modality representation for answer inference. However, this kind of approaches fail to bridge the semantic gap between visual and text modalities, resulting in a lack of alignment in cross-modality semantics and the inability to match key visual content accurately. In this paper, we propose a model called Caption Bridge-based cross-modality Alignment and Contrastive learning model (CBAC) to address the issue. The CBAC model aims to reduce the semantic gap between different modalities. It consists of a caption-based cross-modality alignment module and a vision-caption contrastive learning module. By utilizing an auxiliary caption that shares the same modality as the question and has closer semantic associations with visuals, we are able to effectively reduce the semantic gap by separately matching the caption with both the question and the visual to generate pre-alignment features for each, which are then used in the subsequent fusion process. We also leverage the fact that visual-caption pairs exhibit stronger semantic connections compared to question-visual pairs to employ a contrastive learning mechanism on visual and caption pairs to further enhance the semantic alignment capabilities of single-modality encoders.


## 环境配置
```bash
conda create -n CBAC python=3.9
pip install -r requirement.txt
```

## 致谢
This repository is partly based on LXMERT[https://github.com/airsplay/lxmert#google-drive] and OFA[https://github.com/ofa-sys/ofa] repositories.


