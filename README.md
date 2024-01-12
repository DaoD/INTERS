<div align=center>
<img src="https://github.com/DaoD/INTERS/blob/main/logo1.jpg" width="150px">
</div>


## INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning</h2>
<p>
<a href="https://github.com/DaoD/INTERS/blob/main/LICENSE">
<img src="https://img.shields.io/badge/MIT-License-blue" alt="license">
</a>
</p>
<p>
<a href="">📃 ArXiv</a>
  •
<a href="">🤗 HuggingFace</a>
</p>

**Authors**: Yutao Zhu, Peitian Zhang, Chenghao Zhang, Yifei Chen, Binyu Xie, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen

## Introduction
Large language models (LLMs) have demonstrated impressive capabilities in various natural language processing tasks. Despite this, their application to information retrieval (IR) tasks is still challenging due to the infrequent occurrence of many IR-specific concepts in natural language. While prompt-based methods can provide task descriptions to LLMs, they often fall short in facilitating comprehensive understanding and execution of IR tasks, thereby limiting LLMs' applicability. To address this gap, in this work, we explore the potential of instruction tuning to enhance LLMs' proficiency in IR tasks. We introduce a novel instruction tuning dataset, \ourdata{}, encompassing 21 tasks across three fundamental IR categories: query understanding, document understanding, and query-document relationship understanding. The data are derived from 43 distinct datasets with manually written templates. Our empirical results reveal that \ourdata{} significantly boosts the performance of various publicly available LLMs, such as LLaMA, Mistral, and Phi, in search-related tasks. Furthermore, we conduct a comprehensive analysis to ascertain the effects of base model selection, instruction design, volume of instructions, and task variety on performance. 

Our dataset and the models fine-tuned on it will be released soon!
