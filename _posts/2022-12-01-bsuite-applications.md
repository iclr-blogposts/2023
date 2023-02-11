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
  - name: 0. Introduction
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

## 0. Introduction
For the past few decades, the field of AI has appeared similar to the Wild West. There have been rapid achievements ([Krizhevsky et al. 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), [Hessel et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11796)), uncertain regulations ([Ramesh et al. 2022](https://arxiv.org/abs/2204.06125), [Schulman et al. 2022](https://openai.com/blog/chatgpt/)), and epic showdowns ([Brown & Sandholm 2019](https://www.science.org/doi/abs/10.1126/science.aay2400), [Silver et al. 2016](https://www.nature.com/articles/nature16961), [Vinyals et al. 2019](https://www.nature.com/articles/s41586-019-1724-z)) happening in the frontier of AI research. The subfield of reinforcement learning has been no exception, where progress in the frontier has generated sensational applied feats while leaving theoretical understanding in the dust ([Osband et al., 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)). As in many other AI subfields, there remain prevailing questions such as, *"Which model should I initially select for the given task?"*, *"How can I tune hyperparameters to increase performance?"*, and *"What is the best way to improve my already working model?"*. In this blog post, we help tame the frontier of reinforcement learning research by providing insights and quantitative answers to such questions through diagnostic, methodical, and reproducible reinforcement learning techniques. In particular, we focus on DeepMind's *Behaviour Suite for Reinforcement Learning* (bsuite) codebase and showcase explicit examples of how it can aid reinforcement learning researchers in the development process and help provide a bridge between theoretical and applied reinforcement learning understanding. 

This introduction section provides the necessary background and motivation to understand the importance of our contribution. The background section describes how deep learning provides a blueprint for bridging theory to practice, and then discusses traditional reinforcement learning benchmarks. The bsuite summary section provides a high-level overview of the core capabilities tested by bsuite, its motivation, an example environment, and a comparison against traditional benchmark environments.  In the motivation section, we present arguments for increasing the wealth and diversity of documented *bsuite* examples, with references to the paper and reviewer comments. The contribution statement presents the four distinct contributions of our work that help extend the bsuite publication. Finally, the experiment summary section describes our setup and rationale for the experimental illustrations in sections 1-5. The information in this introduction section is primarily distilled from the original *bsuite* publication ([Osband et al., 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).

### 0.1 Background
The current state of reinforcement learning (RL) theory notably lags progress in practice, especially in challenging problems. There are examples of deep reinforcement learning (DRL) agents learning to play Go from scratch at the professional level ([Silver et al., 2016](https://www.nature.com/articles/nature16961)), learning to navigate diverse video games from raw pixels ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)), and learning to manipulate objects with robotic hands ([Andrychowicz et al., 2020](https://journals.sagepub.com/doi/10.1177/0278364919887447)). While these algorithms have some foundational roots in theory, including gradient descent ([Bottou, 2010](https://link.springer.com/chapter/10.1007/978-3-7908-2604-3_16)), TD learning ([Sutton, 1988](https://link.springer.com/article/10.1007/BF00115009)), and Q-learning ([Watkins, 1992](https://link.springer.com/article/10.1007/BF00992698)), the authors of *bsuite* acknowledge that, "The current theory of deep reinforcement learning is still in its infancy" ([Osband et al., 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).  A strong theory is prized since it can help provide insight and direction for improving known algorithms, while hinting at future research directions.

Fortunately, deep learning (DL) provides a blueprint of the interaction between theoretical and practical improvements. During the 'neural network winter', deep learning techniques were disregarded in favor of more theoretically sound convex loss methods ([Cortes & Vapnik, 1995](https://link.springer.com/article/10.1007/BF00994018)), even though the main ideas and successful demonstrations existed many years previously ([Rosenblatt, 1958](https://psycnet.apa.org/record/1959-09865-001); [Ivakhenko, 1968](https://en.wikipedia.org/wiki/Alexey_Ivakhnenko); [Fukushima, 1979](https://en.wikipedia.org/wiki/Kunihiko_Fukushima)). It was only until the creation of benchmark problems, mainly for image recognition ([Krizhevsky et al., 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)), that deep learning earned the research spotlight due to better scores on the relevant benchmarks. Consequently, a renewed interested in deep learning theory followed shortly after ([Kawaguchi, 2016](https://proceedings.neurips.cc/paper/2016/hash/f2fc990265c712c49d51a18a32b39f0c-Abstract.html); [Bartlett et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html); [Belkin et al., 2019](https://www.pnas.org/doi/abs/10.1073/pnas.1903070116)), bolstered by the considerable wealth of applied research. Due to the lack of theory in DRL and the proximity of the DL and DRL research fields, <span style="color: red;">one enticing avenue to accelerate progress in reinforcement learning research is to follow the blueprint laid out by deep learning research and create well-defined and vetted benchmarks for the understanding of reinforcement learning algorithms</span>.

To this end, the trend of RL benchmarks has seen an increase in overall complexity and perhaps the publicity potential. The earliest such benchmarks were simple MDPs that served as basic testbeds with fairly obvious solutions, such as *Cartpole* ([Barto et al., 1983](https://ieeexplore.ieee.org/abstract/document/6313077)) and *MountainCar* ([Moore, 1990](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.html)). Other benchmarks proved to be more diagnostic by targeting certain capabilities such as *RiverSwim* ([Strehl & Littman, 2008](https://www.sciencedirect.com/science/article/pii/S0022000008000767)) for exploration and *Taxi* ([Dietterich, 2000](https://www.jair.org/index.php/jair/article/view/10266)) for temporal abstraction. Modern benchmarks such as the *ATARI Learning Environment* ([Bellemare et al., 2013](https://www.jair.org/index.php/jair/article/view/10819)) and board games such as *Chess*, *Go*, and *Shogi* are more complex and prove difficult for humans, with even the best humans unable to achieve perfect play. The corresponding achievements were highly publicized ([Silver et al., 2016](https://www.nature.com/articles/nature16961); [Mnih et al., 2015](https://www.nature.com/articles/nature14236)) due to the superhuman performance of the agents, with the agents taking actions that were not even considered by their human counterparts. Consequently, this surge in publicity has been a strong driver of progress in the field and has vaulted the notion of superhuman performance to be the most coveted prize on numerous benchmarks ([Vinyals et al., 2019](https://www.nature.com/articles/s41586-019-1724-z); [Silver et al., 2019](https://www.science.org/doi/abs/10.1126/science.aar6404); [Perolat et al., 2022](https://www.science.org/doi/abs/10.1126/science.add4679); [Ecoffet et al., 2021](https://www.nature.com/articles/s41586-020-03157-9); [Bakhtin et al., 2022](https://www.science.org/doi/abs/10.1126/science.ade9097)).

### 0.2 Summary of *bsuite*

The open-source *Behaviour Suite for Reinforcement Learning* (bsuite) benchmark ([Osband et al. , 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)) goes against the grain of the current benchmark trend of increasing complexity and publicity. Instead of chasing superhuman performance, it acts as a complement to existing benchmarks by creating 23 environments with minimal confounding factors to test 7 behavioral core capabilities of RL agents, as follows: **basic**, **exploration**, **memory**, **generalization**, **noise**, **scale**, and **credit assignment**. Current benchmarks often contain most of these capabilities within a single environment, whereas bsuite tailors its environments to target one or a few of these capabilities. Each bsuite environment is scalable and has 16 to 22 levels of difficulty, providing a more precise analysis of the corresponding capabilities than a simple, and possibly misleading ([Agarwal et al. 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)), ranking of algorithm performance. Furthermore, algorithms have fixed evaluation regimes based on the number of seeds and episodes allowed during training, which rewards algorithms that exhibit the capabilities rather than those that focus on sheer compute power. The targeted and scalable nature of bsuite can provide insights such as eliciting bottlenecks and revealing scaling properties that are opaque in traditional benchmarks. With respect to the benchmarks described in the preceding paragraph, bsuite is most similar to the diagnostic benchmarks of *RiverSwim* ([Strehl & Littman, 2008](https://www.sciencedirect.com/science/article/pii/S0022000008000767)) for and *Taxi* ([Dietterich, 2000](https://www.jair.org/index.php/jair/article/view/10266)) due to its purpose as a stepping stone for tackling more challenging benchmarks.

The bsuite evaluation of an agent yields a radar chart (Fig. 1) that displays the agent's score from 0 to 1 on all seven capabilities, usually based on regret, that yields a quick quantitative comparison between agents. Scores near 0 indicate poor performance, often akin to an agent acting randomly, while scores near 1 indicate mastery of all environment difficulties. A central premise of bsuite is that <span style="color: red;">if an agent achieves high scores on certain environments, then it is much more likely to exhibit the associated core capabilities due to the targeted nature of the environments. Therefore, the agent will more likely perform better on a challenging environment that contains many of the capabilities than one with lower scores on bsuite</span>.  This premise is corroborated by recent research that shows how insights on small-scale environments can still hold true on large-scale environments ([Ceron et al., 2021](https://proceedings.mlr.press/v139/ceron21a.html)).



<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar01.png" class="img-fluid" %}

</div>
<div class="caption">
    Figure 1. Example radar chart of DQN on all 7 bsuite core capabilities.
</div>

An example environment is *deep sea* that targets exploration power. As shown in Figure 2, *deep sea* is an $N \times N$ grid with starting state at cell $(1, 1)$ and treasure at $(N, N)$, with $N$ ranging from 10 to 100. The agent has two actions, move downward left and downward right; the goal is to reach the treasure and receive a reward of $1$ by always moving downward right. A reward of $0$ is given to the agent for moving downward left at a timestep, while a penalizing reward of $-0.01/N$ is given for moving downward right. The evaluation protocol of *deep sea* only allows for $10K$ episodes of $N-1$ time steps each, which prevents an algorithm with unlimited time from casually exploring the entire state space and stumbling upon the treasure. Note that superhuman performance is nonexistent in *deep sea* (and more precisely in the entire bsuite gamut) since a human can spot the optimal policy nearly instantaneously. Surprisingly, we will show later that baseline DRL agents fail miserably at this task. 

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/diagram02.png" class="img-fluid" %}

<div class="caption">
    Figure 2. Illustration of deep sea environment taken from ([Osband et al. , 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).
</div>

</div>

The **challenge** of *deep sea* is the necessity of exploration in an environment that presents an irreversible, suboptimal greedy action (moving downward left) at every time step. This environment **targets** exploration power by ensuring that a successful agent must deliberately choose to explore the state space by neglecting the greedy action. The **simplistic** implementation removes confounding goals, such as learning to see from pixels while completing other tasks ([Mnih et al. 2015](https://www.nature.com/articles/nature14236)). Furthermore, this environment provides a granular exploration score through **scaling** the environment size by $N$ and determining when an agent starts to fail. Finally, the implementation of the environment yields **fast** computation, allowing multiple, quick runs with minimal overhead and compute cost. These 5 aforementioned key qualities are encompassed by all bsuite environments, and we contrast such environments against traditional benchmark environments in the below table.

| Key Quality     | Traditional Benchmark Environment                                                                      | *bsuite* Environment                                                                            |
|-----------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| **Targeted**    | Performance on environment subtly related to many or all core capabilities.                            | Performance on environment directly related with one or few core capabilities.                  |
| **Simple**      | Exhibits many confounding factors related to performance.                                              | Removes confounding factors related to performance.                                             |
| **Challenging** | Requires competency in many core capabilities but not necessarily past normal range in any capability. | Pushes agents beyond normal range in one or few core capabilities.                              |
| **Scalable**    | Discerns agent's power through comparing against other agents and human performance.                   | Discerns agent's competency of core capabilities through increasingly more difficult environments. |
| **Fast**        | Long episodes with computationally-intensive observations.                                             | Relatively small episode and experiment lengths with low observation complexity.                |


### 0.3 Motivation

The authors of bsuite stated, "Our aim is that these experiments can help provide a bridge between theory and practice, with benefits to both sides" ([Osband et al. , 2020](https://iclr.cc/virtual_2020/poster_rygf-kSYwH.html)).  As discussed in the background section, establishing clear benchmarks can yield applied progress, which in turn can accelerate theoretical progress. The use of bsuite in this manner seems highly fruitful since its environments are targeted, which allows for hypothesis testing and eventual formalization into provable guarantees. As such, <span style="color: red;">it is instrumental that the applied aspect of bsuite is emphasized through the adoption and diverse application of reinforcement learning practitioners</span>. 

The applied examples in the published paper are rather meagre: there are two examples of algorithm comparison on two specific environments and three example comparisons of algorithms, optimizers, and ensemble sizes across the entire bsuite gamut in the appendix. The two examples on the specific environments showcase how bsuite can be used for directed algorithm improvement, but the experiments in the appendices only discuss the general notion of algorithm comparison using bsuite scores. In addition to the examples, the authors supply some comments throughout the paper that provide hints regarding the applied usage of bsuite. Looking at the [paper reviews](https://openreview.net/forum?id=rygf-kSYwH), [reviewer #1](https://openreview.net/forum?id=rygf-kSYwH&noteId=rkxk2BR3YH) mentioned how there was no explicit conclusion from the evaluation, and [reviewer #3](https://openreview.net/forum?id=rygf-kSYwH&noteId=rJxjmH6otS) mentioned that examples of diagnostic use and concrete examples would help support the paper. Furthermore, [reviewer #2](https://openreview.net/forum?id=rygf-kSYwH&noteId=SJgEVpbAFr) encouraged publication of bsuite at a top venue to see traction within with the RL research community, and the [program chairs](https://openreview.net/forum?id=rygf-kSYwH&noteId=7x_6G9OVWG) mentioned how success or failure can rely on community acceptance. Considering that bsuite received a spotlight presentation at ICLR 2020 and has amassed over 100 citations in the relatively small field of RL reproducibility during the past few years, bsuite has all intellectual merit and some community momentum to reach the level of a timeless benchmark in RL research. <span style="color: red;">To elevate bsuite to the status of a timeless reinforcement learning benchmark and to help bridge the theoretical and applied sides of reinforcement learning, we believe that it is necessary to develop and document concrete bsuite examples that help answer difficult and prevailing questions throughout the reinforcement learning development process</span>.   

### 0.4 Contribution Statement

This blog post extends the work of bsuite by showcasing 12 example use cases with experimental illustration that directly address specific questions in the reinforcement learning development process to (i) help bridge the gap between theory and practice, (ii) promote community acceptance, (iii) aid applied practitioners, and (iv) highlight potential research directions in reproducible reinforcement learning. 

### 0.5 Experiment Summary

We separate our examples into 5 categories of **initial model selection**, **preprocessing choice**, **hyperparameter tuning**, **testing and debugging**, and **model improvement**. This blog post follows a similar structure to the paper *Deep Reinforcement Learning that Matters* ([Henderson et al, 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11694)) by posing and answering a question in each category, and then providing a few illustrative examples with conclusions. Most examples use Stable-Baselines3 (SB3) ([Raffin et al., 2021](https://dl.acm.org/doi/abs/10.5555/3546258.3546526)) for training DRL agents due to its clarity and simplicity, and the examples focus on DRL due to its pervasiveness in the applied RL community. We provide code and instructions for each experiment in our GitHub codebase (cite), along with hyperparameters and implementation details. Since the focus of this blog post is the discussion of diverse example use cases, not architectural considerations or implementation details, we refer the reader to the [paper appendix](https://openreview.net/pdf?id=rygf-kSYwH#page=13) and the [colab analysis tutorial](https://colab.research.google.com/github/deepmind/bsuite/blob/master/bsuite/analysis/results.ipynb) for more information about the environments and to the [colab intro tutorial](https://colab.research.google.com/drive/1rU20zJ281sZuMD1DHbsODFr1DbASL0RH) and our own codebase (cite) for instructions and examples regarding the implementation of bsuite.

Although running a bsuite environment is orders of magnitude faster than most benchmark environments, the wealth of our examples required us to create a subset of bsuite, which we will refer to as *mini-bsuite* or *msuite* in this work. Using msuite reduced the number of experiments in the gamut from X to Y and reduced the number of environments per core capability from W to Z. We designed msuite to mirror the general scaling pattern of each bsuite environment and the diversity of core capabilities in bsuite; a complete description of msuite can be found in our GitHub codebase (cite). Running experiments on a subset of bsuite highlights its flexibility, and we will show, still elicits quality insights. Since we use a subset of bsuite for our experiments, our radar charts will look different from those in the original bsuite paper. We generally keep the more challenging environments and consequently produce lower scores, especially in the generalization category. 

We stress that the below examples are not meant to amaze the reader or exhibit state-of-the-art research. <span style="color: red;">The main products of this work are the practicality and diversity of ideas in the examples</span>, while the experiments are primarily for basic validation and illustrative purposes. Moreover, these experiments use modest compute power and showcase the effectiveness of bsuite in the low-compute regime. Each example has tangible benefits such as saving development time, shortening compute time, increasing performance, and lessening frustration of the practitioner, among others. To maintain any sense of brevity in this post, we now begin discussion of the examples.

## 1. Initial Model Selection
The reinforcement learning development cycle typically begins with an environment to solve. A natural question usually follows: "*Which underlying RL model should I choose to best tackle this environment, given my resources*?". Resources can range from the hardware (e.g. model size on the GPU), to temporal constraints, to availability of off-the-shelf algorithms ([Liang et al., 2018](https://proceedings.mlr.press/v80/liang18b); [Raffin et al., 2021](https://dl.acm.org/doi/abs/10.5555/3546258.3546526)), to programming efficiency of the practitioner. Initially selecting an effective model can save a great amount of development time due to the potentially greater performance baseline of the agent. In this section, we illustrate how bsuite can be used to effectively answer the question of initial model selection.

### 1.1 Comparing Baseline Algorithms

Perhaps the first choice in the RL development cycle is choosing the algorithm. A considerable amount of RL research is focused on the corresponding algorithms, which presents many possibilities for the researcher. The No Free Lunch Theorem ([Wolpert & Macready](https://ieeexplore.ieee.org/abstract/document/585893/)) tailored to reinforcement learning would state that no algorithm will prove better than any other unless the characteristics of the underlying environment are known. Using bsuite provides a quantitative assessment of algorithm performance on capabilities that are prevalent in many or even most reinforcement learning environments of interest.

Example: Figure 3 shows the performance of the Stable-Baselines3 (SB3) implementations of DQN, A2C, and PPO on msuite with our default hyperparameters. Recent research ([Andrychowicz et al. 2020](https://arxiv.org/abs/2006.05990)) suggests that PPO is the most commonly used RL algorithm, and it was a successor to DQN and A2C. The results indeed show that PPO is superior on msuite in most categories, providing credibility for its use as the premiere baseline DRL algorithm.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar11.png" class="img-fluid" %}

<div class="caption">
    Figure 3. Comparison of SB3 default DQN, A2C, and PPO baseline algorithms.
</div>
</div>

### 1.2 Comparing Off-the-Shelf Implementations

Due to the vast number of reinforcement learning paradigms (e.g. model-based, hierarchical), there are many off-the-shelf (OTS) libraries that provide a select number of thoroughly tested reinforcement learning algorithms. Often, temporal resources or coding capabilities do not allow for practitioners to implement every algorithm by hand. Fortunately, running an algorithm on bsuite can provide a quick glance of an OTS algorithm's abilities at low cost to the practitioner.

Example: Figure 4 compares our default DQN implementation against the example DQN implementation in the bsuite codebase (Nathan check). There is a significant difference between the performance of each implementation on msuite, with the bsuite implementation displaying its superiority. Note that the hyperparameters of bsuite DQN were most likely chosen with the evaluation on bsuite in mind, which could explain its increased performance.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar12.png" class="img-fluid" %}

<div class="caption">
    Figure 4. Comparison of SB3 DQN and bsuite DQN.
</div>

</div>

### 1.3 Gauging Hardware Necessities

Even after an initial algorithm is selected, hardware limitations such as network size and data storage can prevent the agent from being deployed. Using bsuite provides a low-cost comparison among possible hardware choices that can be used to argue for their necessity. This is especially important for small development teams since there can likely be a major disparity between their own hardware resources and those discussed in corresponding research publications. 

Example: Figure 5 compares the default DQN implementation when varying replay buffer sizes, from $1e2$ to $1e5$, with the default having size $1e4$. The original DQN implementation (cite) used a replay buffer of size $1e6$, which is too large for the RAM constraints of many personal computers. The results show that increasing the buffer size to at least $1e4$ yields significant returns on msuite. Note that since the experiment lengths (total time steps for all episodes) of msuite were sometimes less than $1e5$, the larger buffer size of $1e5$ did not always push out experiences from very old episodes, which most likely worsened performance.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar13.png" class="img-fluid" %}

<div class="caption">
    Figure 5. Comparison of DQN with varying buffer sizes.
</div>

</div>

### 1.4 Future Work

Due to the diversity of OTS libraries, one possible research direction in reproducible RL is to test algorithms from different OTS libraries using the same hyperparameters on bsuite and create a directory of bsuite radar charts. This provides practitioners a comparison with their own implementation or a starting point when selecting an OTS library and algorithm. Another direction is to test various aspects related to hardware constraints and attempt to show the tradeoff between constraints and performance on bsuite and other benchmarks. This would especially help practitioners with low compute resources to budget resource use on multiple projects.

## 2. Preprocessing Choice
Most benchmark environments present complexities such as high-dimensional observations, unscaled rewards, unnecessary actions, and partially-observable Markov Decision Process (POMDP) dynamics. Some of these difficulties can be curbed using environment preprocessing techniques. While certain environments such as *ATARI* have formalized standards for preprocessing, there are some aspects such as frame skipping that are considered part of the underlying algorithm, and therefore, a choice of the practitioner ([Machado et al., 2018](https://www.jair.org/index.php/jair/article/view/11182)). A natural question to ask is, "*What environment preprocessing techniques will best help my agent attain its goal in this environment*?".  In this section, we show how bsuite can provide insight to the choice of preprocessing, with benefits of increased performance and shortened training time.

### 2.1 Verification of Preprocessing
Preprocessing techniques usually targeted to ease some aspect of the agent's training. For example, removing unnecessary actions (e.g. in a joystick action space) prevents the agent from having to learn which actions are useless. While a new preprocessing technique can provide improvements, there is always the chance that it fails to make a substantial improvement, or worse yet, generally decreases performance. Invoking bsuite can help provide verification that the preprocessing provided the planned improvement.

Example: Figure 5 shows the performance of the default DQN agent versus an agent that received normalized rewards from the environment. Normalizing the rewards increases the speed of training a neural network, since the parameters are usually initialized to expect target values in a range from $-1$ to $1$. Our results show that the normalization preprocessing indeed increases the capability of navigating varying reward scales while not suffering drastically in any other capability.


<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar21.png" class="img-fluid" %}

<div class="caption">
    Figure 5. Comparison of DQN with and without reward normalization.
</div>

</div>

### 2.2 Choosing a Better Model versus Preprocessing

Instead of choosing to preprocess the environment, a more sophisticated algorithm may better achieve the preprocessing goals. For example, many improvements on the original DQN algorithm have been directed towards accomplishing goals such as improving stability, reducing overestimation, and bolstering exploration. Comparing preprocessing against an algorithmic improvement provides a quantitative reason for deciding between the two options, especially since development time of many common preprocessing wrappers is quite short.

Example: Figure 6 shows the results of PPO with a recurrent network versus PPO having its observation as the last 4 stacked frames from the environment. Frame stacking is common on *ATARI* since it converts the POMDP dynamics to an MDP, which is necessary to determine velocity of any element on the screen. An improvement to DQN, Deep Recurrent Q-networks ([Hausknecht & Stone 2015](https://arxiv.org/abs/1507.06527)), uses a recurrent LSTM to aid in memory and achieve the same effects of frame stacking. The  msuite results show that memory is considerably improved with PPO RNN and therefore may be worth the extra development time.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar22.png" class="img-fluid" %}

<div class="caption">
    Figure 6. Comparison of PPO with framestacking and PPO with RNN.
</div>

</div>


### 2.3 Future Work
One research direction is to document common preprocessing techniques and determine their scores on bsuite. This would provide practitioners a summary of directed strengths for each preprocessing technique while possibly uncovering unexpected behavior. Another direction is to determine the extent to which preprocessing techniques aided previous results in the literature, which could illuminate strengths or weaknesses in the corresponding algorithms.

## 3. Hyperparameter Tuning
After selecting a model and determining any preprocessing of the environment, an agent must eventually be trained on the environment to gauge its performance. During the training process, initial choices of hyperparameters can heavily influence the agent's performance ([Andrychowicz et al., 2021](https://arxiv.org/abs/2006.05990)), including how to explore and how quickly the model should learn from past experience. The corresponding question to ask is, "*How can I choose hyperparameters to yield the best performance, given a model?*" In this section, we show how *bsuite* can be used to tune hyperparameters, thereby increasing performance and shortening compute time.

### 3.1 Unintuitive Hyperparameters
Some hyperparameters such as exploration percentage and batch size are more concrete, while others such as discounting factor and learning rate are a little less intuitive. Determining a starting value of an unintuitive hyperparameter can be challenging and require a few trials before honing in on a successful value. Instead of having to run experiments on a costly environment, using *bsuite* can provide a thoughtful initial guess of the value at a low compute cost.

*Example*: Figure X shows the results of running PPO with various entropy bonus coefficients across *msuite*. The entropy bonus affects the action distribution of the agent, and the value of 1e-2 presented in the original paper ([Schulman et al. 2017](https://arxiv.org/pdf/1707.06347.pdf)) is fairly unintuitive. The results show that the value of 1e-2 is indeed superior on *msuite* by a small margin. Since SB3 has the entropy bonus initialized to 0, this example also shows how hyperparameter tuning with *msuite* can improve performance even with OTS implementations.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar31.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### 3.2 Promising Ranges of Hyperparameters
Instead of determining a single value of a hyperparameter, gauging an acceptable range may be required. Since hyperparameters can have confounding effects, knowing approximate soft boundaries of hyperparameters at which agents start to fail at basic tasks can provide useful information during a more general hyperparameter tuning process. For example, smaller learning rates generally take longer for algorithm convergence, and a practitioner may want to know a promising range of learning rates if the computing budget is flexible. The scaling nature of *bsuite* presents knowledge of the extent to which different hyperparameter choices effect performance, greatly aiding in ascertaining a promising hyperparameter range.

*Example*: Figure X shows the results of testing the aforementioned experiment with various learning rates using DQN as the underlying algorithm on *msuite*. The results show that leraning rates above 1e-2 start to yield diminishing returns. Since some experiment lengths in *msuite* only run for 10K episodes, the lowest learning rate of 1e-6 may never converge in time even with high-quality training data, necessitating a modification to *msuite* to learn a lower bound.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar32.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### 3.3 Pace of Annealing Hyperparameters
While some hyperparameters stay fixed, others must change throughout the course of training. Typically, these include hyperparameters that control the exploration vs. exploitation dilemma, including entropy bonus and epsilon-greedy exploration. These hyperparameters are often dependent on the entire experiment; for example, SB3 anneals epsilon-greedy exploration for a fixed fraction of the experiment. Therefore, entire experiments, some consisting of millions of episodes, need to be run to determine successful values of these hyperparameters. Using *bsuite* can provide a quick confirmation that the annealing of these parameters happens at an acceptable rate.

*Example*: Figure X shows the performance of DQN with various epsilon-greedy exploration annealing lengths, based on a fixed fraction of the entire experiment. The annealing fraction of 0.1 performs best on *msuite*, which is the same choice of parameter in the original DQN paper (cite). Furthermore, performance decreases with greater annealing lengths. Since *bsuite* environments are generally scored with regret, we acknowledge that the longer annealing lengths may have better relative performance if *bsuite* were scored with a training versus testing split.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar33.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### 3.4 Future Work
The three experiments above can be extended by documenting the affect of varying hyperparameters on performance, especially in OTS implementations. This would help practitioners understand the effects of certain hyperparameters on the *bsuite* core  capabilities, allowing for a better initial hyperparameter choice when certain capabilities are necessary for the environment at hand. Another research direction is to determine if integrating a fast hyperparameter tuner on general environments such as *bsuite* into a hyperparameter tuner for single, complex environments would increase the speed of tuning on the fixed environment. Since the *bsuite* core capabilities are necessary in many complex environments, and initial pass to determine competency on *bsuite* would act as a first pass of the tuning algorithm.

## 4. Testing and Debugging
Known to every RL practitioner, testing and debugging during the development cycle is nearly unavoidable. It is common to encounter silent bugs in RL code, where the program runs but the agent fails to learn because of an implementation error. Examples include incorrect preprocessing, incorrect hyperparameters, or missing algorithm additions. Quick unit tests can be invaluable for the RL practitioner, as shown in successor work to *bsuite* ([Rajan & Hutter, 2019](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/19-NeurIPS-Workshop-MDP_Playground.pdf)). A corresponding question to ask during the testing and debugging phase is, "*What tests can I perform to verify that my agent is running as intended?*" In this section, we show how *bsuite* can be used as a sanity check the expectations and assumptions of the implementation, saving compute time and lessening the frustration of the practitioner (a very existential and limited quantity). In an effort to refrain from contrived examples, the two examples below highlight real-life scenarios where using *bsuite* could have saved the authors of this blog post hours of frustration in their own work.

### 4.1 Incorrect Hyperparameter
As discussed in the previous section, hyperparameters are of major importance to the performance of a RL algorithm. A missing or incorrect hyperparameter will not necessarily prevent a program from running, but most such bugs will severely degrade performance. Using *bsuite* can quickly expose poor performance of an algorithm at a low cost to the practitioner.

*Example*: Figure X shows the default PPO implementation against a PPO implementation with an erroneous learning rate of 1e-3. Many hyperparameters such as total training steps, minimum buffer size before training, steps until epsilon is fully annealed, and maximum buffer size are usually coded using scientific notation since they are so large; consequently, it is easy to forget the 'minus sign' in the coding of the learning rate and instead code the learning rate as 1e3. The results on *msuite* show that performance has degraded severely from an OTS implementation, and more investigation into the code is required. One of the authors of this blog post would have saved roughly a day of training a PPO agent in their own work had they realized this mistake.  

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar41.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### 4.2 OTS Algorithm Testing
While the previous example used an OTS algorithm for comparison to illuminate silent bugs, it may be the case that the OTS algorithm could have a silent bug. Whether due to an incorrect library being used or a misunderstanding of the OTS algorithm, any silent bug in an OTS algorithm can be difficult to detect due to the codebase being written by another practitioner. Again, *bsuite* can be used to diagnose poor performance and elucidate a coding problem.

*Example*: Figure X shows the results of the SB3 DQN with our default experimental hyperparameters and with the default SB3 hyperparameters on *msuite*. A core difference between the hyperparameters is the burn rate: the default SB3 hyperparameters perform 10K steps before learning takes place (e.g. backprop), while our hyperparameters start the learning much more quickly (Nathan). Since many of the 'easier' *msuite* environments only last 10K time steps, failure to learn anything during that time severely degrades performance, as shown. Noticing the default value of this hyperparameter in SB3 would have saved the authors roughly 10 hours of training time.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar42.png" class="img-fluid" %}

*Figure X. Comparison of DQN and DQN with Reward Scaling.*

</div>

### 4.3 Future Work
The training time for a complete run of *bsuite* can take an hour on even the most basic algorithms. Considering that a few of the easiest *bsuite* environments could have shown poor performance in the above examples within mere minutes, one research avenue is to create a fast debugging system for reinforcement learning algorithms. In the spirit of *bsuite*, it should implement targeted experiments to provide actionable solutions for eliminating silent bugs. Such work would primarily act as a public good, but it could also help bridge the gap between RL theory and practice if it could embody the targeted nature of *bsuite*.

## 5. Model Improvement
A natural milestone in the RL development cycle is getting an algorithm running bug-free with notable signs of learning. A common follow-up question to ask is "*How can I improve my model to yield better performance?*" The practitioner may consider choosing an entirely new model and repeating some of the above steps; a more enticing option is usually to improve the existing model by reusing its core structure and only making minor additions or modifications, an approach taken in the baseline RAINBOW DQN algorithm ([Hessel et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11796)). In this section, discuss how *bsuite* can be used to provide targeted improvements of  existing models and increase performance while mitigating compute time.

### 5.1 Increasing Network Complexity
In DRL, the neural network usually encodes the policy, and its architecture directly affects the agent's learning capacity. The more complicated CNN architecture was a driver for the first superhuman performance of a DRL algorithm on the ATARI suite due to its ability to distill image data into higher-level features. Using *bsuite* can provide a quick verification if an architectural improvement produces its intended effect.

*Example*: Figure X shows the results of PPO against PPO with a recurrent neural network. As mentioned in a previous example, RNNs aid memory and were originally incorporated into DRL as a way to deal with POMDP dynamics. The results on *msuite* display the substantial increase in memory capability while sacrificing on credit assignment. This example highlights how *bsuite* can provide warnings of possible unexpected decreases in certain capabilities, which must be monitored closely by the practitioner. 


<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar51.png" class="img-fluid" %}

*Figure X. Comparison of PPO with FNN (default) and PPO with RNN (recurrent).*

</div>

### 5.2 OTS Improvements
While previous examples discussed comparison, verification, and debugging OTS implementations, many OTS libraries provide support for well-known algorithm improvements. For example, some DQN implementations have boolean values to signify the use of noisy networks, double Q-learning, and more. Using *bsuite* provides the necessary targeted analysis to help determine if certain improvements are fruitful for the environment at hand.

*Example*: Figure X shows the results of our default DQN compared against the SB3 QRDQN algorithm with default hyperparameters and the SBE QRDQN algorithm with hyperparameters matching our default DQN implementation. The QRDQN algorithm is an improvement over DQN that aims to capture the distribution over returns instead of a point estimate of the expected return. This implementation is more complex but allows for a precise estimate that aids in stability. The results show that this improvement was rather negligible on *msuite*, and unless credit assignment is the major concern in the environment at hand, a different improvement may prove more useful.

<div style="text-align: center;">

{% include figure.html path="assets/img/2022-12-01-bsuite-applications/radar52.png" class="img-fluid" %}

*Figure X. Comparison of PPO with FNN (default) and PPO with RNN (recurrent).*

</div>

### 5.3 Future Work
 Since *bsuite* provides quantitative results, one avenue of research is to create a recommender system that uses information from previous *bsuite* experiments to recommend improvements in DRL algorithms. The practitioner would need to provide as input the most important capabilities that the environment is believed to exhibit, and *bsuite* would tailor recommendations towards those capabilities. Such a recommender system could save compute time, increase performance, and ultimately expose the practitioner to new and exciting algorithmic possibilities.

## 6. Conclusion

Traditional RL benchmarks contain many confounding variables, which makes analysis of agent performance rather opaque. In contrast, *bsuite*  provides targeted environments that help gauge agent prowess in one or few core capabilities. The goal of *bsuite* is to help bridge the gap between practical theory and practical algorithms, yet there currently is no database or list of example use cases for the practitioner. Our work extends *bsuite* by providing concrete examples of its use, with a few examples in each of five categories. We supply at least one possible avenue of related future work or research for each category. In its current state, *bsuite* is poised to be a standard RL benchmark for years to come due to its acceptance in a top-tier venue, well-structured codebase, multiple tutorials, and over 100 citations in the past few years in a relatively small field. We aim to help propel *bsuite*, and more generally methodical and reproducible RL research, into the mainstream through our explicit use cases and examples. With a diverse set of examples to choose from, we intend for applied RL practitioners to understand more use cases of *bsuite*, apply and document the use of *bsuite* in their experiments, and ultimately help bridge the gap between practical theory and practical algorithms. 

### 6.1 Green Computing Statement

The use of *bsuite* can provide directed improvements in algorithms, from high-level model selection and improvement to lower-level debugging, testing, and hyperparameter tuning. Due to the current climate crisis, we feel that thoroughly-tested and accessible ideas that can reduce computational cost should be promoted to a wide audience of researchers.

### 6.2 Inclusive Computing Statement

Many of the ideas in *bsuite* and this blog post are most helpful in regimes with low compute resources because of the targeted nature of these works. Due to the increasing gap between compute power of various research teams, we feel that thoroughly-tested and accessible ideas that can benefit teams with meagre compute power should be promoted to a wide audience of researchers.

## Acknowledgements
{Redacted for peer-review}