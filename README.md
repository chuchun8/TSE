# TSE

ACL 2023 main conference paper: A New Direction in Stance Detection: Target-Stance Extraction in the Wild.

## Abstract

Stance detection aims to detect the stance toward a corresponding target. Existing works use the assumption that the target is known in advance, which is often not the case in the wild. Given a text from social media platforms, the target information is often unknown due to implicit mentions in the source text and it is infeasible to have manual target annotations at a large scale. Therefore, in this paper, we propose a new task Target-Stance Extraction (TSE) that aims to extract the (target, stance) pair from the text. We benchmark the task by proposing a two-stage framework that first identifies the relevant target in the text and then detects the stance given the predicted target and text. Specifically, we first propose two different settings: Target Classification and Target Generation, to identify the potential target from a given text. Then we propose a multi-task approach that takes target prediction as the auxiliary task to detect the stance toward the predicted target. We evaluate the proposed framework on both in-target stance detection in which the test target is always seen in the training stage and zero-shot stance detection that needs to detect the stance for the targets that are unseen during the training phase. The new TSE task can facilitate future research in the field of stance detection.

## Setup

You can download the project and install required packages using following commands:

```bash
git clone https://github.com/chuchun8/TSE.git
cd src
mkdir ./trained_models
pip install -r requirements.txt
```

## Running

A multi-task framework that uses target prediction as the auxiliary task is proposed in this work. You can train the stance classifier in the multi-task setting using the following command:

```
bash ./train.sh > train.log
```
Results of in-target stance detection (Table 8) is shown in `train.log`. Specifically, in train.sh,

`-m` indicates the model we use to train or evaluate on stance datasets, which includes `bertweet`, `crossnet`, `bice`, `tan`, `bilstm`. Trained models are saved in the `trained_models` folder.

`-mul` indicates that we train stance classifiers with our proposed multi-task setting (Figure 2). Removing `-mul` in the script will degrade the training setting to only stance detection. Some examples of these settings are given in `train.sh`.

In this work, a two-stage framework is proposed to involve both target identification and stance detection given a text. After training the stance classifier, you can run the following command to evaluate the model on the outputs of "Target Classification" and "Target Generation" (Section 4.1):
```
bash ./eval.sh > eval.log
```
Results of two-stage stance detection (Tables 5-7) is shown in `eval.log`. Some examples of "Target Classification", "Target Generation" and "Zero-Shot" settings are given in `eval.sh`.

## Data

The folder is organized as follows:

- `*_target_classification.csv`: output files from "Target Classification" stage;
- `*_target_generation.csv`: output files from "Target Generation" stage;
- `*_target_generation_zero_shot.csv`: output files from "Target Generation" stage for zero-shot evaluation;
- `train/dev/test.csv`: dataset files used to train stance classifiers.

Please contact the co-author regarding more details of files of "Target Classification" and "Target Generation".

## Checkpoints

We make trained models publicly available at [here](https://drive.google.com/drive/folders/1jN7n0U2-6A3xMmCLIvPfsZMLM0zzHpuJ?usp=sharing). Please feel free to train models by yourself or directly download well-trained models for evaluation.

## Contact Info

Please contact Yingjie Li at liyingjie@westlake.edu.cn or yli300@uic.edu with any questions.

## Citation

```bibtex
@inproceedings{li-etal-2023-new,
    title = "A New Direction in Stance Detection: Target-Stance Extraction in the Wild",
    author = "Li, Yingjie  and
      Garg, Krishna  and
      Caragea, Cornelia",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.560",
    pages = "10071--10085",
}
```
