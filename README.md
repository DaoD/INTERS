<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/logo1.jpg" width="150px">
</div>


## INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning</h2>
<p>
<a href="https://github.com/DaoD/INTERS/blob/main/LICENSE">
<img src="https://img.shields.io/badge/MIT-License-blue" alt="license">
</a>
</p>
<p>
📃 <a href="">ArXiv Paper</a>
  •
🤗 <a href="">HuggingFace Model</a> 
  •
📚 <a href="">Dataset</a>
</p>

**Authors**: Yutao Zhu, Peitian Zhang, Chenghao Zhang, Yifei Chen, Binyu Xie, Zhicheng Dou, Zheng Liu, and Ji-Rong Wen

⭐ **We will release the datasets, models, templates, and codes within a month (before Feb. 15th). Thanks for your attention!**

## Introduction
<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/img/intro.jpg"  width="600px">
</div>

Large language models (LLMs) have demonstrated impressive capabilities in various natural language processing tasks. Despite this, their application to information retrieval (IR) tasks is still challenging due to the infrequent occurrence of many IR-specific concepts in natural language. While prompt-based methods can provide task descriptions to LLMs, they often fall short in facilitating a comprehensive understanding and execution of IR tasks, thereby limiting LLMs' applicability. To address this gap, in this work, we explore the potential of instruction tuning to enhance LLMs' proficiency in IR tasks. We introduce a novel instruction tuning dataset, INTERS, encompassing 20 tasks across three fundamental IR categories: query understanding, document understanding, and query-document relationship understanding. The data are derived from 43 distinct datasets with manually written templates. Our empirical results reveal that INTERS significantly boosts the performance of various publicly available LLMs, such as LLaMA, Mistral, and Phi, in IR tasks. Furthermore, we conduct extensive experiments to analyze the effects of instruction design, template diversity, few-shot demonstrations, and the volume of instructions on performance. 

Our dataset and the models fine-tuned on it will be released soon!

## Tasks & Datasets

## Dataset Construction

## General Performance

## Citation
Please kindly cite our paper if it helps your research:
```BibTex
@article{Inters,
    author={Yutao Zhu and
            Peitian Zhang and
            Chenghao Zhang and
            Yifei Chen and
            Binyu Xie and
            Zhicheng Dou and
            Zheng Liu and
            Ji-Rong Wen},
    title={INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning},
    journal={CoRR},
    volume={abs/2401.06532},
    year={2024},
    url={https://arxiv.org/abs/2401.06532},
    eprinttype={arXiv},
    eprint={}
}
```
