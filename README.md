<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/logo1.jpg" width="150px">
</div>


## INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning</h2>
<p>
<a href="https://github.com/DaoD/INTERS/blob/main/LICENSE">
<img src="https://img.shields.io/badge/MIT-License-blue" alt="license">
</a>
</p>

**Authors**: Yutao Zhu, Peitian Zhang, Chenghao Zhang, Yifei Chen, Binyu Xie, Zhicheng Dou, Zheng Liu, and Ji-Rong Wen

<p>
📃 <a href="https://arxiv.org/abs/2401.06532">ArXiv Paper</a>
  •
📚 <a href="https://huggingface.co/datasets/yutaozhu94/INTERS">Dataset</a>
</p>
<p>
🤗 HuggingFace Model List
</p>

| Model                                                                            | Backbone Model                                                          |
|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------|
| [INTERS-LLaMA-7b-Chat](https://huggingface.co/yutaozhu94/INTERS-LLaMA-7b-chat)   | [LLaMA-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)           |
| [INTERS-LLaMA-7b-Base](https://huggingface.co/yutaozhu94/INTERS-LLaMA-7b-base) | [LLaMA-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| [INTERS-Mistral-7b](https://huggingface.co/yutaozhu94/INTERS-Mistral-7b)         | [Mistral-7b](https://huggingface.co/mistralai/Mistral-7B-v0.1)          |
| [INTERS-Minima-3b](https://huggingface.co/yutaozhu94/INTERS-Minima-3b)           | [Minima-2-3b](https://huggingface.co/GeneZC/MiniMA-2-3B)                |
| [INTERS-Falcon-1b](https://huggingface.co/yutaozhu94/INTERS-Falcon-1b)           | [Falcon-rw-1b](https://huggingface.co/tiiuae/falcon-rw-1b)              |

## News
- Feb, 2024: We release the training set, validation set, part of the test set, instruction templates, and fine-tuned models. Other resources are still in preparation.

## Introduction
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/intro.jpg"  width="600px">
</div>

Large language models (LLMs) have demonstrated impressive capabilities in various natural language processing tasks. Despite this, their application to information retrieval (IR) tasks is still challenging due to the infrequent occurrence of many IR-specific concepts in natural language. While prompt-based methods can provide task descriptions to LLMs, they often fall short in facilitating a comprehensive understanding and execution of IR tasks, thereby limiting LLMs' applicability. To address this gap, in this work, we explore the potential of instruction tuning to enhance LLMs' proficiency in IR tasks. We introduce a novel instruction tuning dataset, INTERS, encompassing 20 tasks across three fundamental IR categories: query understanding, document understanding, and query-document relationship understanding. The data are derived from 43 distinct datasets with manually written templates. Our empirical results reveal that INTERS significantly boosts the performance of various publicly available LLMs, such as LLaMA, Mistral, and Phi, in IR tasks. Furthermore, we conduct extensive experiments to analyze the effects of instruction design, template diversity, few-shot demonstrations, and the volume of instructions on performance. 

Our dataset and the models fine-tuned on it will be released soon!

## Tasks & Datasets
We consider tasks under the categories of query understanding, document understanding, and query-document understanding. Our dataset consists of 20 tasks derived from 43 datasets. All tasks and datasets we used are shown in the figure below.
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/dataset.png">
</div>

## Dataset Construction
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/process.jpg">
</div>

## General Performance
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/in-domain-google.png">
</div>

## Citation
Please kindly cite our paper if it helps your research:
```BibTex
@article{INTERS,
  author       = {Yutao Zhu and
                  Peitian Zhang and
                  Chenghao Zhang and
                  Yifei Chen and
                  Binyu Xie and
                  Zhicheng Dou and
                  Zheng Liu and
                  Ji{-}Rong Wen},
  title        = {{INTERS:} Unlocking the Power of Large Language Models in Search with
                  Instruction Tuning},
  journal      = {CoRR},
  volume       = {abs/2401.06532},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2401.06532},
  doi          = {10.48550/ARXIV.2401.06532},
  eprinttype    = {arXiv},
  eprint       = {2401.06532}
}
```
