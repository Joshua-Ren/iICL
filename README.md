# Language Model Evolution: An Iterated Learning Perspective (iICL)
This repository contains all the LLM experiments in [Language Model Evolution: An Iterated Learning Perspective]()

### Abstract
With the widespread adoption of Large Language Models (LLMs), the prevalence of iterative interactions among these models is anticipated to increase. Notably, recent advancements in multi-round self-improving methods allow LLMs to generate new examples for training subsequent models. At the same time, multi-agent LLM systems, involving automated interactions among agents, are also increasing in prominence. Thus, in both short and long terms, LLMs may actively engage in an evolutionary process. We draw parallels between the behavior of LLMs and the evolution of human culture, as the latter has been extensively studied by cognitive scientists for decades. Our approach involves leveraging Iterated Learning (IL), a Bayesian framework that elucidates how subtle biases are magnified during human cultural evolution, to explain some behaviors of LLMs. This paper outlines key characteristics of agents' behavior in the Bayesian-IL framework, including predictions that are supported by experimental verification with various LLMs. This theoretical framework could help to more effectively predict and guide the evolution of LLMs in desired directions.



### Interesting findings

- Letting LLMs learn from the data samples generated by another LLM (can be itself) attracts more attentions these days.

- Such a procedure could be depicted by the following Bayesian iterated learning framework (with interaction filter included): bias in prior would be amplified generation by generation and a good interaction phase can mitigate this.

<div align=center><img src="https://github.com/Joshua-Ren/better_supervisory_signal/blob/main/gifs/settings.gif" width="480"/><img src="https://github.com/Joshua-Ren/better_supervisory_signal/blob/main/gifs/hardsamples.gif" width="220"/></div>

# About this repo
The experiments we considered here is simple but quite helpful: we can directly observe many details during evolution, like the logits of all hypothesis, stats of the transferred data, etc.
And the experiments only takes less than $5.

### Requirements
- API of GPT, Claude, Mistral

# Reference
For technical details and full experimental results, please check [our paper]( ).
```
@inproceedings{ren:iicl,
    author = {Yi Ren and Shangmin Guo and Linlu Qiu and Bailin Wang and Danica J. Sutherland},
    title = {Language Model Evolution: An Iterated Learning Perspective},
    year = {2024},
    booktitle = {arxiv},
}
```

# Contact
Please contact renyi.joshua@gmail.com if you have any question on the codes.
