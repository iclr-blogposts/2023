---
layout: distill
title: Practical Applications of Bsuite For Reinforcement Learning
description: In 2019, researchers at DeepMind published a suite of reinforcement 
  learning environments called Behavior Suite for Reinforcement Learning, or bsuite. 
  Each environment is designed to directly test a core capability of a general 
  reinforcement learning agent, such as its ability to generalize from past experience 
  or handle delayed rewards. The authors claim that bsuite can be used to benchmark 
  agents and bridge the gap between theoretical and applied reinforcement learning 
  understanding. In this blog post, we extend their work by providing specific examples 
  of how bsuite can address common challenges faced by reinforcement learning practitioners 
  during the development process. Our work offers pragmatic guidance to researchers and 
  highlights future research directions in reproducible reinforcement learning.
date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2022-12-01-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
    subsections:
      - name: Background
      - name: BSuite Summary
      - name: Motivation
      - name: Contribution
  - name: Initial Model Choice 
    subsections:
      - name: Comparing Baseline Algorithms
      - name: Comparing Off-the-Shelf Implementations
      - name: Gauging Diminishing Returns of Agent Complexity
      - name: Summary and Future Work
  - name: Preprocessing Selection
    subsections:
      - name: Choosing a Better Model vs Preprocessing
      - name: Verification of Preprocessing
      - name: Other
      - name: Summary and Future Work
  - name: Hyperparameter Tuning
    subsections:
      - name: Unintuitive Hyperparameters
      - name: Promising Ranges of Hyperparameters
      - name: Pace of Annealing Hyperparameters
      - name: Summary and Future Work
  - name: Testing and Debugging
    subsections:
      - name: Missing Add-on
      - name: Incorrect Constant
      - name: OTS Algorithm Testing
      - name: Summary and Future Work
  - name: Model Improvement
    subsections:
      - name: Increasing Network Complexity
      - name: Decoupling or Adding Confidence
      - name: Determining Necessary Improvements
      - name: Summary and Future Work
  - name: Conclusion
    subsections:
    - name: Summary
    - name: Green Computing Statement
    - name: Inclusive Computing Statement

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

For the past decade, the field of AI has appeared similar to the Wild West, with rapid achievements (mnist to imagenet), daring investments (chat gp3?), and showdowns (go vs human) happening in new and uncharted territory. Our topic of study, reinforcement learning, has been no exception, where progress in the applied realm (akin to the frontier) is past that of theory (civilization?). In this subfield, as in many other AI subfields, prevailing questions without provable answers remain such as *Which model should I use?*, *How can I tune hyperparameters?*, and *What is the best way to improve my model?*. In this blog post, we try to help tame the (frontier?) Wild West landscape of reinforcement learning by providing insights and quantitative answers to such questions through diagnostic and methodical reinforcement learning techniques. In particular, we focus on DeepMind's *Behaviour Suite for Reinforcement Learning*, or bsuite, codebase and showcase explicit examples of how it can aid reinforcement learning researchers in the development cycle.

This introduction section provides the necessary background and motivation to understand the importance of our contribution. The background describes how deep learning provides a blueprint for bridging theory to practice, and then discusses traditional reinforcement learning benchmarks. The bsuite summary section provides a high-level overview of the core capabilities tested by bsuite, an example output (radar plot), an example environment, and a comparison against traditional benchmark environments. The information from these first two sections was primarily distilled from the original bsuite publication. In the motivation section presents arguments for increasing the wealth of documented bsuite examples, with references to the paper and the reviews. Finally, the contribution section showcases four distinct contributions of our work and provides our rationale for the experiment setups and the content of the remainder of the paper.

### Background
The current state of reinforcement learning (RL) theory notably lags progress in practice, especially in challenging problems. There are examples of deep reinforcement learning (DRL) agents learning to play Go from scratch at the professional level ([Silver et al., 2016](https://www.nature.com/articles/nature16961)), learning to navigate diverse video games from raw pixels ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)), and learning to manipulate objects with robotic hands ([Andrychowicz et al., 2020](https://journals.sagepub.com/doi/10.1177/0278364919887447)). While these algorithms have some foundational roots in theory, including gradient descent ([Bottou, 2010](https://link.springer.com/chapter/10.1007/978-3-7908-2604-3_16)), TD learning ([Sutton, 1988](https://link.springer.com/article/10.1007/BF00115009)), and Q-learning ([Watkins, 1992](https://link.springer.com/article/10.1007/BF00992698)), the authors of bsuite acknowledge that, "The current theory of deep reinforcement learning is still in its infancy" ([Osband et al., 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).  A strong theory is prized since it can help provide insight and direction for improving known algorithms, while hinting at future research directions.

Fortunately, deep learning (DL) provides a blueprint of the interaction between theoretical and practical improvements. During the 'neural network winter', deep learning techniques were disregarded in favor of more theoretically sound convex loss methods ([Cortes & Vapnik, 1995](https://link.springer.com/article/10.1007/BF00994018)), even though the main ideas and successful demonstrations existed many years previously ([Rosenblatt, 1958](https://psycnet.apa.org/record/1959-09865-001); [Ivakhenko, 1968](https://en.wikipedia.org/wiki/Alexey_Ivakhnenko); [Fukushima, 1979](https://en.wikipedia.org/wiki/Kunihiko_Fukushima)). It was only until the creation of benchmark problems, mainly for image recognition ([Krizhevsky et al., 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)), deep learning earned the research spotlight due to better scores on the relevant benchmarks. Consequently, a renewed interested in deep learning theory followed shortly after ([Kawaguchi, 2016](https://proceedings.neurips.cc/paper/2016/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html); [Bartlett et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html); [Belkin et al., 2019](https://www.pnas.org/doi/abs/10.1073/pnas.1903070116)), bolstered by the considerable wealth of applied research. Due to the lack of theory in DRL and the proximity of the DL and DRL research fields, <span style="color: red;">one enticing avenue to accelerate progress in reinforcement learning research is to follow the blueprint laid out by deep learning research and create well-defined and vetted benchmarks for the understanding of deep reinforcement learning algorithms</span>.

To this end, RL benchmark environments have been strong drivers of progress in the field, and they have seen an increase in overall complexity and perhaps the publicity potential. The earliest such benchmarks were simple MDPs that served as basic testbeds with fairly obvious solutions, such as *Cartpole* ([Barto et al., 1983](https://ieeexplore.ieee.org/abstract/document/6313077)) and *MountainCar* ([Moore, 1990](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.html)). Other benchmarks proved to be more diagnostic by targeting certain capabilities such as *RiverSwim* ([Strehl & Littman, 2008](https://www.sciencedirect.com/science/article/pii/S0022000008000767)) for exploration and *Taxi* ([Dietterich, 2000](https://www.jair.org/index.php/jair/article/view/10266)) for temporal abstraction. Modern benchmarks such as the *ATARI Learning Environment* ([Bellemare et al., 2013](https://www.jair.org/index.php/jair/article/view/10819)) and board games such as *Chess*, *Go*, and *Shogi* are more complex and prove difficult for humans, with even the best humans unable to achieve perfect play. The corresponding achievements were highly publicized ([Silver et al., 2016](https://www.nature.com/articles/nature16961); [Mnih et al., 2015](https://www.nature.com/articles/nature14236)) due to the superhuman performance of the agents, with the agents taking actions that were not even considered by their human counterparts. Consequently, this surge in publicity has vaulted the notion of superhuman performance to be the de facto prize on numerous benchmarks ([Vinyals et al., 2019](https://www.nature.com/articles/s41586-019-1724-z); [Silver et al., 2019](https://www.science.org/doi/abs/10.1126/science.aar6404); [Perolat et al., 2022](https://www.science.org/doi/abs/10.1126/science.add4679); [Ecoffet et al., 2021](https://www.nature.com/articles/s41586-020-03157-9); [Bakhtin et al., 2022](https://www.science.org/doi/abs/10.1126/science.ade9097)).

### Summary of bsuite

The opensource bsuite benchmark ([Osband et al. , 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)) goes against the grain of the current benchmark trend of increasing complexity and publicity. Instead of chasing superhuman performance, it acts as a complement to existing benchmarks by creating 23 environments with minimial confounding factors to test 7 behavioral core capabilities of RL agents, as follows: **basic competency**, **exploration**, **memory**, **generalization**, **noise**, **scale**, and **credit assignment**.  Current benchmarks often contain most of these capabilities within a single environment, whereas bsuite tailors its environments to target one or a few of these capabilities. Each bsuite environment is scalable and has 16 to 22 levels of difficulty, allowing for a more precise analysis of the corresponding capabilities than a binary(?) ranking of algorithm performance. Furthermore, algorithms have fixed evaluation regimes (seeds and timesteps), which targets the assessment of the capability rather than focus on excessive compute. Finally, bsuite is very flexible with respect to algorithm input, allowing for a more uniform benchmark across many domains of RL research. For example, DQN can be evaluated for \$6 With respect to the benchmarks described in the preceding paragraph, bsuite is most akin to the diagnostic benchmarks of *RiverSwim* ([Strehl & Littman, 2008](https://www.sciencedirect.com/science/article/pii/S0022000008000767)) for and *Taxi* ([Dietterich, 2000](https://www.jair.org/index.php/jair/article/view/10266)) as its use of a stepping stone for tackling more challenging benchmarks.

As a quick illustration, the bsuite evaluation of an agent yields a radar chart that displays the agent's score from 0 to 1 on all seven capabilities (Figure X), allowing for a quick glance of quantitative comparisons. Due to the more general learning nature of RL agents, testing on 7 qualities ('makes sense') Scores near 0 indicate poor performance, often akin to an agent acting randomly, while scores near 1 indicate mastery of all environment difficulties. A central premise of bsuite is that <span style="color: red;">if an agent achieves high scores on certain environments, then it is much more likely to exhibit the associated core capabilities due to the targeted nature of the environments. Therefore, the agent will more likely perform better on a challenging environment that contains many of the capabilities than one with lower scores on bsuite</span>. The targeted and scalable nature of bsuite can provide insights such as eliciting bottlenecks and revealing scaling properties that are opaque in traditional benchmarks. By removing confounding factors, bsuite aims to provide more concrete insights of conceptual ideas. This premise is corroborated by current research that shows how insights on small-scale environments can still hold true on large-scale environments ([Ceron et al., 2021](https://proceedings.mlr.press/v139/ceron21a.html)) and how aggregate scores can be misleading and inhibit progress in the field ([Agarwal et al. 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)). Due to its high-quality codebase and clear experiments, bsuite provides a high-quality benchmark that aids research in RL reproducibility.

<div style="text-align: center;">

![](/images/radar01.png)

*Figure 1. Radar chart of DQN with 7 core capabilities of bsuite.*

</div>

An example environment is *deep sea* that is targeted towards assessing exploration power. As shown in the picture, *deep sea* is and $N \times N$ grid with starting state at cell $(1, 1)$ and treasure at $(N, N)$, with $N$ from 10 to 100. The agent has two actions, left and right, and the goal is to reach the treasure and receive a reward of $1$ by always moving to the right. A reward of $0$ is given to the agent for moving left at a timestep; a reward $-0.01/N$ is given for moving to the right. Note that superhuman performance is not used as a benchmark in bsuite since a human can spot an optimal policy (always move right) nearly instantaneously, while we will show in ([1.1](#11-comparing-baseline-algorithms)) that standard DRL agents fail miserably at this task. The evaluation protocol of *deep sea* only allows for 10K episodes across any environment, which prevents an algorithm with unlimited time from lazily (instead of directed) exploring the entire state space.

<div style="text-align: center;">

![](/images/radar02.png)

*Figure 2. Illustration of Deep Sea environment taken from [Osband et al. , 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html).*

</div>

The **challenge** of *deep sea* is the necessity of exploration in an environment that has a suboptimal greedy action (moving left) at every time step. This environment **targets** the exploration power of the agent by ensuring that a successful agent will explore the state space and not quickly conceded to always choosing the greedy action. The **simplistic** implementation distills the observations to high-level features that remove conflicting goals of an agent, such as learning to see (cite atari). Furthermore, this environment provides a non-binary exploration score by **scaling** the environment size $N$ and determining when the agent starts to fail. Finally, the implementation of the environment yields **fast** computation, allowing multiple, quick runs with minimal overhead and compute cost. These 5 aforementioned key qualities are encompassed by all bsuite environments, and we contrast such environments against traditional benchmark environments in the below table.

| Key Quality     |   Traditional Benchmark Environment  | bsuite Environment                                                                               |
|-----------------|-------------------|----------------------------------------------------------------------------------------------------|
| **Targeted**    | Performance on environment subtly related to many or all core capabilities. | Performance on environment is directly related with one or few core capabilities.                  |
| **Simple**      | Exhibits many confounding factors related to performance. | Removes confounding factors related to performance.                                                |
| **Challenging** | Requires competency in many core capabilities but not necessarily past normal range in any capability. | Pushes agents beyond normal range in one or few core capabilities.                                 |
| **Scalable**    | Discerns agent's power through comparing against other algorithms and human performance. | Discerns agent's competency of core capabilities through increasingly more difficult environments. |
| **Fast**        | Long episodes with computationally-intensive observations. | Relatively small episode and experiment lengths with low observation complexity.                   |


### Motivation

The authors of bsuite stated, "Our aim is that these experiments can help provide a bridge between theory and practice, with benefits to both sides" ([Osband et al. , 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).  As discussed in ([0.1](#01-background)), establishing clear benchmarks can yield applied progress, which in turn can accelerate theoretical progress. An example is MIPLIB (cite) that evaluates key proprties of problems in the mixed integer programming domain, where applied progress has as well surpass theoretical progress. The use of bsuite in this manner seems highly prospective since its environments are targeted, which allows for hypothesis testing and eventual formalization into provable guarantees. As such, <span style="color: red;">it is instrumental that the applied aspect of bsuite is emphasized through the adoption and diverse application of deep reinforcement learning practitioners</span>.

The applied examples in the published paper are rather meagre: there are two examples of algorithm comparison on two specific environments and three example comparisons of algorithms, optimizers, and ensemble sizes across the entire bsuite gamut in the appendix. The examples on the specific environments showcase how bsuite can be used for directed algorithm improvement, but the experiments in the appendices only discuss the general notion of algorithm comparison using bsuite scores. In addition to the examples, the authors supply some comments throughout the paper that provide hints regarding the applied usage of bsuite, as well as mentioning how the plots 'give a flavour' (cite?) of the diagnostic nature of bsuite. As much as BSuite has directed experiments, we feel like there should be directed examples.

Looking at the [paper reviews](https://openreview.net/forum?id=rygf-kSYwH), [reviewer #1](https://openreview.net/forum?id=rygf-kSYwH&noteId=rkxk2BR3YH) mentioned how there was no explicit conclusion from the evaluation, and [reviewer #3](https://openreview.net/forum?id=rygf-kSYwH&noteId=rJxjmH6otS) mentioned that examples of diagnostic use and concrete examples would help support the paper. Furthermore, [reviewer #2](https://openreview.net/forum?id=rygf-kSYwH&noteId=SJgEVpbAFr) encouraged publication of bsuite at a top venue to see traction within with the RL research community, and the [program chairs](https://openreview.net/forum?id=rygf-kSYwH&noteId=7x_6G9OVWG) mentioned how success or failure can rely on community acceptance. Considering that bsuite received a spotlight presentation at ICLR 2020 and has amassed over 100 citations in the relatively small field of RL reproducibility throughout the past few years, bsuite has all intellectual merit and some community momentum to reach the level of a top-tier and timeless benchmark in RL research. <span style="color: red;">To elevate bsuite to the status of a top-tier reinforcement learning benchmark and to help bridge the theoretical and applied sides of reinforcement learning, we believe that it is necessary to develop and document concrete bsuite examples that help answer difficult and prevailing questions during the reinforcement learning development cycle</span>.

### Contribution

**Contribution Statement**: This blog post extends the work of bsuite by showcasing 15 explicit use cases with experimental illustration that directly address specific questions in the RL development cycle to (i) help bridge the gap between theory and practice, (ii) promote community acceptance, (iii) aid applied practitioners, and (iv) highlight potential research directions for reproducible RL.

### Experiment Summary

We separate our examples into 5 categories of **model choice**, **preprocessing selection**, **hyperparameter tuning**, **debugging**, and **model improvement**. This blog post follows a similar structure to the paper *Deep Reinforcement Learning that Matters* ([Henderson et al, 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11694)) by posing and answering a question regarding each category, and then providing 3 illustrative examples with conclusions. Most examples use Stable-Baselines3 (SB3) ([Raffin et al., 2021](https://dl.acm.org/doi/abs/10.5555/3546258.3546526)) for training the DRL agents due to its clarity and simplicity. We provide code and instructions for each experiment in our GitHub codebase (cite). Since the focus of this blog is the discussion of diverse example use cases, not architectural considerations or implementation details, we refer the reader to the [paper appendix](https://openreview.net/pdf?id=rygf-kSYwH#page=13) and the [colab analysis tutorial](https://colab.research.google.com/github/deepmind/bsuite/blob/master/bsuite/analysis/results.ipynb) for more information about the environments and to the [colab intro tutorial](https://colab.research.google.com/drive/1rU20zJ281sZuMD1DHbsODFr1DbASL0RH) and our own codebase (cite) for instructions and examples of the implementation of bsuite. Our examples focus primarily DRL, as that is the hub of more RL research these days, but they extend to general RL. Our baselines algorithms of DQN, A2C, and PPO are compared in (1.1), which is a good place to start reviewing examples since we discuss choices of initial hyperparameters there. Most similar to our examples would be the deep sea and memory length experiments in the paper, as they draw conclusions/insights from the radar charts.

Due to computational necessity, we created a subset of bsuite, which we will refer to as *mini-bsuite* or *msuite* in this work that reduced the number of experiments from X to Y and reduced the number of environments per core capability from W to Z. We designed *msuite* to mirror the general scaling pattern of each bsuite environment and diversity of core capabilities among all bsuite environments; a complete description of *msuite* can be found in our GitHub codebase (cite). Since bsuite was a single `bsuite2019` release and meant to evolve over time, the selection for number and diversity of environments seemed to have an arbitrary threshold; therefore, we don't hesitate to create our own arbitrary threshold resulting in *msuite*, and we feel that running experiments on a subset of bsuite highlights the strength and flexibility of using a targeted diagnostic benchmark to elicit insights. One reason our charts may vary from those of bsuite paper is that they reduce the number of experiments and generally keep the harder experiments. While we always use bsuite, we again mention how methodical and diagnostic RL with emphasis on reproducibility is the general goal.

We stress that the below examples are not meant to amaze the reader or exhibit state-of-the-art research. <span style="color: red;">The main products of this work are the practicality and diversity of ideas in the examples</span>, while the examples are primarily for basic validation and illustrative purposes. Moreover, these experiments use modest compute power and showcase the effectiveness of bsuite in the low-compute regime. Each example has a benefit such as saving development time, shorten compute time, increase performance, and lessen frustration of the practitioner, among other benefits; we don't directly acknowledge these benefits since they are mainly due to the domain choice and implementation of a researcher. Discussion of these savings are relegated to the individual categories, and to maintain any sense of brevity, we now begin discussion of the examples. These examples need to be brief to maintain any sort of brevity. Some examples are more conclusive than others.

(MAKE NEW EXAMPLES SECTION!!!)

## Initial Model Selection
The reinforcement learining development cycle typically begins with selecting or being given an underlying environment. Perhaps the first question in the cycle is as follows, "*Which underlying RL model should I choose to best tackle this environment, given my resources*?" Resources can range from the hardware (e.g model size on the GPU), to temporal constraints, to availability of off-the-shelf algorithms ([Liang et al., 2018](https://proceedings.mlr.press/v80/liang18b); [Raffin et al., 2021](https://dl.acm.org/doi/abs/10.5555/3546258.3546526)), to maximimum difficulty of agent implementation. In this section, we illustrate that, while optimally answering the above question may remain out of reach, bsuite can be used to provide quantitative answers to those questions.

### Comparing Baseline Agents

Perhaps the first task of reinforcement learning, given an environment, is choosing an agent. There are many architectures to choose from, and it would be nice to know which is better beforehand. Unfortunately, the No Free Lunch Theorem of data science tailored to reinforcement learning states that no algorithm will prove better than any other unless the characteristics of the underlying environment are known. Using bsuite provides a quantitative assessment of agent performance on capabilities that are prevalent in many or even most reinforcement learning environments of interest.

*Example*: This example runs the Stable-Baselines3 (SB3) implementations of DQN, A2C, and PPO on *mbsuite* and displays results in Figure 3. Traditional knowledge states that PPO is more powerful and a natural successor to A2C, and DQN is possibly the most basic useful DRL algorithm, and the results corroborate that fact (cite on-policy paper?). In most categories, especially the credit assignment category, PPO scores higher than the other agents, prompting its use as the premiere baseline algorithm.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/1.1/radar.png" class="img-fluid" %}

*Figure 1.1. Comparison of default DQN, A2C, and PPO baselines.*

</div>

### Comparing Off-the-Shelf Implementations

Due to the non-modular nature of reinforcement learning (model-based, heirarchical), there have been many libraries, each with different purposes. Often, resources or coding capabilities do not allow for self-implementation, and OTS algorithms have been thoroughly tested. Fortunately, bsuite can provide a glance of an OTS's strengths. This is important because many OTS libraries may have hidden hyperparameters, etc.

*Example*: In figure 2, we compare the DQN implementation from SB3 to the example DQN given by bsuite. It is clear that the DQN from bsuite is superior in each tested capability. This shows that bsuite's implementation may have been tuned for these tasks or may have had a different architecture. (check to see hyperparams).

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/1.2/radar.png" class="img-fluid" %}

*Figure 1.2. Comparison of default DQN, A2C, and PPO baselines.*

</div>

### Gauging Hardware Necessities

Even when a RL agent is selected, hardware limitations can prevent the agent from being deployed. These limitations can range from network size to data storage. Testing agents quickly on bsuite can provide a fast way of determining diminishing returns of a model size, especially when the size isn't known well in advance. (discuss how papers may have giant sizes).

*Example*: We show how different replay buffer sizes has an impact on *msuite*. The original buffer in DQN paper is 1M which doesn't always fit into computers; other implementations for consumers use 10K steps (cite lapan). Our results show that there are significant returns when the model size is at least 10K. Since some of the experiments have rather short episode lengths, it makes sense that the larger buffer doesn't push out those experiences.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/1.3/radar.png" class="img-fluid" %}

*Figure 1.3. Comparison of default DQN, A2C, and PPO baselines.*

</div>

### Summary and Future Work

This section provided a quick glance of possible comparisons during the model selection phase. Due to the diversity of OTS libraries, one possible research direction in reproducible RL is to test similar algorithms with the same hyperparameters on bsuite and create a directory of bsuite radar charts. Also, one could test the prowess of various hardware (e.g. NN model) sizes and perhaps show a sweet spot on bsuite and other benchmarks.

## Preprocessing Choice
Many environments come with various complexities, such as high-dimensional, unscaled observations, unscaled rewards, unnecessary actions, and partially-observable Markov Decision Process (POMDP) dynamics. A natural question to ask is, "*What environment preprocessing techniques will best help my agent attain its goal in this environment*?" While environments sometimes come proprocessed 'out-of-the-box', the classic benchmarking and evaluation paper on *ATARI* ([Machado et al., 2018](https://www.jair.org/index.php/jair/article/view/11182)) states that preprocessing is considered part of the underlying algorithm and is indeed a choice of the practitioner. In this section, we show how bsuite can provide insight when selecting the preprocessing methods.

### Choosing a Better Model vs. Preprocessing

Instead of proprocessing the environment better, it could be the case that a more sophisticated agent is required. For example, many improvements on DQN have been in aiding stability, overestimation, and noise. Comparing the algorithms on *msuite* provides a way to determine *the extent* to which an improvement is better, which can also help with development time considerations.

*Example*: Framestacking was introduced by (atari paper) to transform the atari environments into MDPs. An addition was shown ... which instead used an RNN to have memory and use the memory to change the POMDP to an MDP. Using *msuite* shows that the improvement with RNN does indeed score higher without sacrificing anything else.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/2.1/radar.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### Verification of Preprocessing

Making modifications to the environment is often directed at some feature of the environment. While a new preprocessing technique can help, there is always the chance that it doesn't (cite improvement paper) or even harms other capabilities. Invoking bsuite can quickly assure that there is no harm done and that the preprocessing occurs as planned.

*Example*: Environments can come with varying scales of rewards out of the box, and sometimes it is unknown what the range is. Here, we create a reward normalization wrapper that normalizes the rewards (how?). The results show that the algorithm improves on the reward scale capability and doesn't suffer much at all in any other capability, corroborating its use.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/2.2/radar.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### Other

### Summary and Future Work
This section showed how bsuite can effectively and efficiently gauge the power and capabilities of certain preprocessing techniques. (cite mainly for improving performance and verification). Of course, one research direction is to document possible preprocessing techniques and determine their scores on bsuite for quick comparisons. Another avenue is to critique the literature to determine the extent to which preprocessing techniques aided results.

## Hyperparameter Tuning
After selecting a model and determining any preprocessing of the environment, the next step is to train the agent on the environment and gauge its competency. During the training process, initial choices of hyperparameters can play a large role in the agent performance ([Andrychowicz et al., 2021](https://arxiv.org/abs/2006.05990)), ranging from how to explore, how quickly the model should learn from experience, and the length of time that actions are considered to influence rewards. Due to their importance, a question is, "*How can I choose hyperparameters to yield the best performance, given a model?*" In this section, we show how bsuite can be used for validation and efficiency of tuning hyperparameters.

### Unintuitive Hyperparameters
Some hyperparameters such as exploration percentage and batch size are more concrete, while others such as gamma and learning rate are a little less intuitive. Determining a starting value of an unintuitive hyperparameter (or one made up by the experimenter) can be challenging. Instead of trying many runs of a difficult environment, running bsuite can give a rough estimate of an acceptable hyperparameter value.

*Example*: The entropy bonus coefficient of PPO is unintuitive, with few comparisons, in contrast to learning rate with DL. Here, we try many different hyperparameters of PPO and show that when the entropy coefficient is 0.01, the agent has highest performance on *msuite*.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/3.1/radar.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### Promising Ranges of Hyperparameters
While choosing a 'best' hyperparameter with *msuite* is enticing, a range of valid hyperparameters may be what is required. For example, an increase of computational power may allow a slightly lower learning rate across more episodes for added stability. Running a range of hyperparameters can help determine the most promising regions and certain soft boundaries for hyperparameters, especially without changing other hyperparameters.

*Example*: Learning rate in RL is typically lower than in DL due to the non-stationary training dataset, required for stability. Here, we test learning rates of various scales with the default DQN implementation. The learning rates near 1e-2 to 1e-4 are certainly better for these tasks (mention regret). Note how the original DQN paper has learning rate 1e-4 (mention mainly looking at top bound because of regret).

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/3.2/radar.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### Pace of Annealing Hyperparameters
While some hyperparameters stay fixed, some must change throughout the course of training. Typically, these include hyperparameters that control the exploration vs. exploitation dilemma, including entropy bonus and epsilon-greedy exploration. One can use bsuite to provide a quick confirmation that the annealing of these parameters isn't too fast or slow on basic tasks.

*Example*: We anneal possibly the most well-known parameter on DQN: epsilon. Running *msuite* on possible annealing fractions yields the following figure. As can be seen, the value of 0.1 does well, and in general exploring less on these tasks is better due to the regret score. These results corroborate the exploration fraction in the DQN paper (cite).

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/3.3/radar.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### Summary and Future Work
Determining hyperparameters can directly save experimentation time and improve performance of an algorithm. Having fast runs on bsuite can reduce computational burden of a run on another environment, although care needs to be taken so results are transferrable (e.g. making sure capabilities are same). The three experiments above can be extended by documenting the change in hyperparameters, and the possible annealing. Furthermore, since bsuite is quantitative, it would be interesting to see an automatic hyperparameter tuner with bsuite and compare against one for a specific environment or integrate it with another tuner.

## Testing and Debugging
Known to every practitioner, testing and debugging a program is neraly unavoidable. A common question in the RL development cycle is, "*What tests can I perform to verify that my agent is running as intended?*" Due to the prevalence of silent bugs in RL code and long runtimes, quick unit tests can be invaluable for the practitioner, as shown in successor work to bsuite ([Rajan & Hutter, 2019](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/19-NeurIPS-Workshop-MDP_Playground.pdf)). In this section, we show how bsuite can be used as a sanity check the expectations and assumptions of the implementation, which was mentioned as a use case of bsuite in the paper.

### Missing Add-on

### Incorrect Constant

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/4.2/radar.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### OTS Algorithm Testing

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/4.3/radar.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### Summary and Future Work
Debugging surely saves development time and lessens frustration of the practitioner. Future research directions are two sides to the same coin - using bsuite and logging bugs from poor performance, and creating directed unit tests to squash bugs. (Anything else?). For testing purposes, perhaps create a suite of increasing benchmarks to determine where the difficulty is. (intermediate between completely diagnostic and complex benchmarks) - does that fit here? A catalogue of specific algorithms and hyperparameters would help with testing (discussed previously).

## Model Improvement
A natural milestone in the RL development cycle is getting an algorithm running bug-free with notable signs of learning. A common follow-up question to ask is "*How can I improve my model to yield better performance?*" The practitioner may consider choosing an entirely new model and repeating some of the above steps; usually, a more enticing option is directly improving the existing model by reusing its core structure and only making minor additions or modifications, an approach taken in the state-of-the-art RAINBOW DQN algorithm ([Hessel et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11796)). In this section, we discuss ideas regarding the improvement of pre-existing somewhat competent models.

### Increasing Network Complexity

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/5.1/radar.png" class="img-fluid" %}

*Figure X. Comparison of PPO with FNN (default) and PPO with RNN (recurrent).*

</div>

### Decoupling or Adding Confidence

### Determining Necessary Improvements

### Summary and Future Work

Recommender system - document improvements of bells and whistles.

## Conclusion
The above sections complete the main ideas of this paper. We now provide a hindsight summary of our work. Afterwards, we supply statements on green and inclusive computing regarding our contribution.

### Summary

Traditional RL benchmarks contain many confounding variables, which makes post-analysis of agent performance somewhat opaque. In contrast, bsuite  provides targeted environments that are meant to gauge agent prowess in one or few core capabilities. The goal of bsuite is meant to bridge the gap between practical theory and practical algorithms, yet there currently is no database or list of example use cases for the practitioner. Furthermore, bsuite is poised to be a standard RL benchmark for years to come due to its acceptance in a top-tier venue, well-structured codebase, multiple tutorials, and over 100 citations in the past few years in a relatively small field.

Our work extends bsuite by providing 15 concrete examples of its use, with 3 examples in 5 categories. Each category section provides at least one possible avenue of related future work or research. We aim to help propel bsuite, and more generally methodical and reproducible RL research, into the mainstream through our explicit examples with simple code. With a diverse set of examples to choose from, we intend applied practitioners to understand more use cases, apply and document the use of bsuite in their experiments, and ultimately help bridge the gap between practical theory and practical algorithms.

### Green Computing Statement

The use of bsuite can help find directed improvements in algorithms, from high-level model selection and improvement to lower-level debugging, testing, and hyperparameter tuning. Due to the current climate crisis, we feel that thoroughly-tested and accessible ideas that can greatly reduce computational cost should be promoted to a wide audience of researchers.

### Inclusive Computing Statement

Many of the ideas in bsuite and this post are most helpful in the areas of low compute resources, due to more directed areas of improvment and selection. Due to the seemingly-increasing gap between compute power of various research teams, we feel that thoroughly-tested and accessible ideas that can greatly benefit teams with meagre compute power should be promoted to a wide audience of researchers.

## Acknowledgements
{Redacted for peer-review}