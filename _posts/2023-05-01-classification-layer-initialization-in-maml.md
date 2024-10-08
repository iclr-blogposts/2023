---
layout: distill
title: Strategies for Classification Layer Initialization in Model-Agnostic Meta-Learning
description: [This blog post discusses different strategies for initializing the classification layers parameters before fine-tuning on a new task in Model-Agnostic Meta-Learning. Each of the strategies in question has emerged from a different problem, and it will be analyzed whether one approach can solve the problems addressed by the other approaches.]
date: 2023-05-01
htmlwidgets: true

# anonymize when submitting 
# authors:
#  - name: Anonymous 

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
authors:
   - name: Nys Tjade Siegel
     url: "https://www.linkedin.com/in/nys-tjade-siegel-b06a1718a?originalSubdomain=de"
     affiliations:
       name: ALU Freiburg
   - name: Thomas Goerttler
     url: "https://scholar.google.de/citations?user=ppQIwpIAAAAJ&hl=de"
     affiliations:
       name: TU Berlin
   - name: Klaus Obermayer
     url: "https://www.tu.berlin/ni/"
     affiliations:
       name: TU Berlin

# must be the exact same name as your blogpost
bibliography: 2023-05-01-classification-layer-initialization-in-maml.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction
  - name: What is Meta-Learning?
  - name: Quick recap on MAML
  - name: Learning a single initialization vector
  - name: Zero initialization
    subsections:
      - name: MAML's SCL Intuition
  - name: Initialization using prototypes
  - name: What else is there?
  - name: Conclusion and Discussion
---


## Introduction

In a previous study, Raghu et al. [2020] <d-cite key="DBLP:conf/iclr/RaghuRBV20"></d-cite> found that in model-agnostic meta-learning (MAML) for few-shot classification, the majority of changes observed in the network during the inner loop fine-tuning process occurred in the linear classification head. It is commonly believed that during this phase, the linear head remaps encoded features to the classes of the new task. In
traditional MAML, the weights of the final linear layer are meta-learned in the usual way. However, there are some issues with this approach:

First, it is difficult to imagine that a single set of optimal classification head weights can be learned. This becomes apparent when considering class label permutations: two different tasks may have the same classes but in a different order. As a result, the weights that perform well for the first task will likely not be effective for the second task. This is reflected in the fact that MAML's performance can vary by up to 15% depending on the class label ordering during testing <d-cite key="DBLP:conf/iclr/YeC22"></d-cite>.

Second, more challenging datasets are being proposed as few-shot learning benchmarks, such as Meta-Dataset <d-cite key="DBLP:conf/iclr/TriantafillouZD20"></d-cite>. These datasets have varying numbers of classes per task, making it impossible to learn a single set of weights for the classification layer.

Therefore, it seems logical to consider how to initialize the final classification layer before fine-tuning on a new task. Random initialization may not be optimal, as it can introduce unnecessary noise <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite>. 

This blog post will discuss different approaches to the last layer initialization that claim to outperform the original MAML method.

## What is Meta-Learning?


Before diving into the topic, let's look at the general idea of meta-learning. In supervised machine learning, tasks are learned using a large number of labeled examples.
However, acquiring a sufficient amount of labeled data can be labor extensive. Also, this approach to machine learning evidently deviates from the human learning process; a child is certainly
able to learn what a specific object is, using only a few examples, and not hundreds or thousands.
This is where meta-learning comes in. Its goal can be described as acquiring the ability to learn new tasks from only a few examples <d-cite key="9428530"></d-cite>.

There is not one fixed framework for meta-learning; however, a common approach is based on the principle that the conditions in which a model is trained and evaluated must match <d-cite key="vinyals2016matching"></d-cite>.\\
Let's look at this in more detail for the case of few-shot classification, which can be solved with meta-learning. Here, the meta-learning goal
can be verbalized as "learning to learn new classes from few examples" <d-cite key="DBLP:conf/iclr/TriantafillouZD20"></d-cite>. When evaluating a meta-learner, one needs a training set $$
\mathcal{D^{tr}} = ((\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), (\mathbf{x}_3, y_3), ...)$$, consisting of labeled examples for unseen classes.
Those are used by the meta-learner to adapt to the novel task. How well the meta-learner performs can then be evaluated on labeled examples from the same classes: $$
\mathcal{D^{test}} = ((\mathbf{x}_{1}^{\ast}, y_{1}^{\ast}), (\mathbf{x}_{2}^{\ast}, y_{2}^{\ast}), (\mathbf{x}_{3}^{\ast}, y_{3}^{\ast}), ...)$$. The combination of such a training and test set is referred to
as an episode or a task: $\mathcal{T} = (\mathcal{D^{tr}}, \mathcal{D^{test}})$.

To match the conditions for training and evaluation, one would split all available classes with their examples into a dataset for meta-training $$\mathcal{C}_{train}$$ and a dataset for 
meta-testing $$\mathcal{C}_{test}$$. Tasks are then drawn from those datasets for either training or testing purposes.\\
A possible approach for using a task in the training phase could be: Fine-tune the meta-learner
using $$\mathcal{D^{tr}}$$, evaluate its performance on $$\mathcal{D^{test}}$$, and finally update the model based on this evaluation error. 

## Quick recap on MAML
Model-Agnostic Meta-Learning (MAML) <d-cite key="DBLP:conf/icml/FinnAL17"></d-cite> is a well-established algorithm in the field of optimization-based meta-learning. Its goal is to find parameters $\theta$ for a parametric model $f_{\theta}$ that can be efficiently adapted to perform an unseen task from the same task distribution, using only a few training examples. The pre-training of $\theta$ is done using two nested loops (bi-level optimization), with meta-training occurring in the outer loop and task-specific fine-tuning in the inner loop. The task-specific fine-tuning is typically done using a few steps of gradient descent:

$$
\theta_{i}' = \theta - \alpha\nabla_{\theta}\mathcal{L_{\mathcal{T_{i}}}}(\theta, \mathcal{D^{tr}})
$$

where $\alpha$ is the inner loop learning rate, $\mathcal{L_{\mathcal{T_{i}}}}$ is a task's loss function, and $\mathcal{D^{tr}}$ is a task's training set. The task includes a test set as well: $\mathcal{T_{i}} = (\mathcal{D_{i}^{tr}}, \mathcal{D_{i}^{test}})$.

In the outer loop, the meta parameter $\theta$ is updated by backpropagating through the inner loop to reduce errors made on the tasks' test set using the fine-tuned parameters:

$$
\theta' = \theta - \eta\nabla_{\theta} \sum_{\mathcal{T_{i}} \sim p(\mathcal{T})}^{} \mathcal{L_{\mathcal{T_{i}}}}(\theta_{i}', \mathcal{D^{test}}).
$$

Here, $\eta$ is the meta-learning rate. The differentiation through the inner loop involves calculating second-order derivatives, which mainly distinguishes MAML from simply optimizing for a $\theta$ that minimizes the average task loss.

It is worth noting that in practical scenarios, this second-order differentiation is computationally expensive, and approximation methods such as first-order MAML (FOMAML) <d-cite key="DBLP:conf/icml/FinnAL17"></d-cite> or Reptile <d-cite key="DBLP:journals/corr/abs-1803-02999"></d-cite> are often used. In FOMAML, the outer loop update is simply: $$ \theta' = \theta - \eta\nabla_{\theta'} \sum_{\mathcal{T_{i}} \sim p(\mathcal{T})}^{}\mathcal{L_{\mathcal{T_{i}}}}(\theta_{i}', \mathcal{D^{test}}) $$, which avoids differentiating through the inner loop.

Before proceeding, let's prepare ourselves for the next sections by looking at the notation we can use when discussing MAML in the few-shot classification regime: The model's output prediction can be described as $\hat{y} = f_{\theta}(\mathbf{x}) = \underset{c\in[N]}{\mathrm{argmax}} h_{\mathbf{w}} (g_{\phi}(\mathbf{x}), c)$, where we divide our model $f_{\theta}(\mathbf{x})$ (which takes an input $\mathbf{x}$) into a feature extractor $g_{\phi}(\mathbf{x})$ and a classifier $h_\mathbf{w}(\mathbf{r}, c)$, which is parameterized by classification head weight vectors ${\mathbf{w}}_{c=1}^N$. $\mathbf{r}$ denotes an input's representation, and $c$ is the index of the class we want the output prediction for.

Finally, $\theta = {\mathbf{w_1}, \mathbf{w_1}, ..., \mathbf{w_N}, \phi}$, and we are consistent with our previous notation.

## Learning a single initialization vector
The first two variants of MAML - we look at - approach the initialization task by initializing the classification head weight vectors identically for all classes. In the paper

<p></p>
<span>&nbsp;&nbsp;&nbsp;&#9654;&nbsp;&nbsp;</span>Han-Jia Ye & Wei-Lun Chao (ICLR, 2022) How to train your MAML to excel in few-shot classification <d-cite key="DBLP:conf/iclr/YeC22"></d-cite>,
<p></p>

an approach called <strong>UnicornMAML</strong> is presented. It is explicitly motivated by the effect that different class-label assignments can have. Ye & Chao [2022] <d-cite key="DBLP:conf/iclr/YeC22"></d-cite> report that during testing, vanilla MAML can perform very differently for <ins>tasks with the same set of classes</ins>, which are just <ins>differently ordered</ins>. Namely, they report that classification accuracy can vary up to 15% in the one-shot setting and up to 8% in the five-shot setting. This makes MAML's performance quite unstable.
<br/><br/>


{% include figure.html path="assets/img/2023-05-01-classification-layer-initialization-in-maml/perm_final.png" class="img-fluid" %}

<p align = "center">
<em>Fig.1 Example of MAML and a class label permutation <d-cite key="DBLP:conf/iclr/YeC22"></d-cite>. We can see the randomness introduced, as $\mathbf{w_1}$ is supposed to interpret the input features as "unicorn" for the first task and as "bee" for the second. For both tasks, the class outputted as a prediction should be the same, as in human perception, both tasks are identical. This, however, is obviously not the case.</em>
</p>

The solution proposed is fairly simple: Instead of meta-learning $N$ weight vectors for the final layer, only a <ins>single vector</ins> $\mathbf{w}$ is meta-learned and used to initialize all $ \\{ \mathbf{w} \\}_{c=1}^N $ before the fine-tuning stage.

This forces the model to make random predictions before the inner loop, as $\hat{y_c}= h_{\mathbf{w}} (g_{\phi} (\mathbf{x}), c)$ will be the same for all $c \in [1,...,N ]$.

After the inner loop, the updated parameters have been computed as usual: $$ \theta' = \\{\mathbf{w_1}', \mathbf{w_2}', ..., \mathbf{w_N}', \phi'\\} $$. The gradient for updating the single classification head meta weight vector $\mathbf{w}$, is just the aggregation of the gradients w.r.t. all the single $\mathbf{w_c}$:

$$
\nabla_{\mathbf{w}} \mathcal{L_{\mathcal{T_i}}} (\mathcal{D^{test}}, \theta_i) = \sum_{c \in [N]} \nabla_{\mathbf{w_c}}
\mathcal{L_{\mathcal{T_i}}} (\theta_i, \mathcal{D^{test}})
$$

This collapses the models meta-parameters to $ \theta = \\{\mathbf{w}, \phi\\} $.
<br/><br/>


{% include figure.html path="assets/img/2023-05-01-classification-layer-initialization-in-maml/unicorn_maml_final.png" class="img-fluid" %}

<p align = "center">
<em>Fig.2 Overview of UnicornMAML <d-cite key="DBLP:conf/iclr/YeC22"></d-cite>. We can see that class label permutations don't matter anymore, as before fine-tuning, the probability of predicting each class is the same.</em>
</p>

This tweak to vanilla MAML makes UnicornMAML permutation invariant, as models fine-tuned on tasks including the same categories - just differently ordered - will now yield the same output predictions. Also, the method could be used with more challenging datasets where the number of classes varies without any further adaptation: It doesn't matter how many classification head weight vectors are initialized by the single meta-classification head weight vector.

Furthermore, the uniform initialization in Unicorn-MAML addresses the problem of memorization overfitting <d-cite key="DBLP:conf/iclr/YinTZLF20"></d-cite>. 
The phenomenon describes a scenario where a single model can learn all the training tasks only from the test data in the outer loop. 
This leads to a model that learns to perform the training tasks but also to a model that doesn't do any fine-tuning and thus fails to generalize to unseen tasks.

Yin et al. [2020] <d-cite key="DBLP:conf/iclr/YinTZLF20"></d-cite> illustrate memorization overfitting using a simple example: Imagine a 3D pose prediction problem,
where each task consists of 2D pictures of a certain object. The objects are rotated by some angle from an (unknown) canonical pose in every picture.
Each picture is labeled by the angle by which the object is rotated from the object's canonical pose.

In a memorization overfitting scenario, a model learns and memorizes the canonical pose of all the objects shown during training. 
This way, the model no longer needs to adapt during fine-tuning in the meta-training phase.
For correctly dealing with the test examples during training, it could just recognize which object it is looking at and calculate the angle from the remembered canonical pose.\\
This becomes a problem when unseen objects are shown to the model during meta-testing. Here, it would be critical to infer
the canonical pose from the training examples to infer the rotation angle for the test examples correctly. This, however,
was not learned by the model in this example.

When initializing the classification head identically for all classes, the model is forced to adapt during fine-tuning, 
as otherwise, it would predict only at the chance level.
This prevents memorization overfitting.

Ye & Chao [2022] <d-cite key="DBLP:conf/iclr/YeC22"></d-cite> benchmark UnicornMAML on MiniImageNet and TieredImageNet.
In the five-shot setting, the approach is claimed to outperform ProtoNet, ProtoMAML, MetaOptNet, MTL+E3BM, RFS-Distill, DeepEMD, MATE+MetaOpt
DSN-MR and FEAT. In the one-shot setting, UnicornMAML is reported to perform averagely compared with the other methods.

Let's finally think of how to interpret UnicornMAML: When meta-learning only a single classification head vector, one could say that rather than learning a mapping from features to classes, the weight vector instead learns a prioritization of those features that seem to be more relevant across tasks.

## Zero initialization
The second approach for initializing weights identically for all classes is proposed in the paper

<p></p>
<span>&nbsp;&nbsp;&nbsp;&#9654;&nbsp;&nbsp;</span>Chia-Hsiang Kao et al. (ICLR, 2022) MAML is a Noisy Contrastive Learner in Classification <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite>.
<p></p>

Kao et al. [2022] <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite> modify the original MAML by setting the whole classification head to zero before each inner loop. They refer to this MAML-tweak as the <strong>zeroing trick</strong>.

An overview of MAML with the zeroing trick is displayed below:

<div class="l-page">
  <iframe src="{{ 'assets/html/2023-05-01-classification-layer-initialization-in-maml/algorithm.html' | relative_url }}" frameborder='0' scrolling='no' width="100%"  height="400px"></iframe>
</div>


<p align = "center">
<em>Fig.3 MAML with the zeroing trick applied <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite>.</em>
</p>

Through applying the zero initialization, three of the problems addressed by UnicornMAML are solved as well:
- MAML, with the zeroing trick applied, leads to random predictions before fine-tuning. This happens as zeroing the whole classification head
is also a form of identical weight initialization for all classes. Thus, the zeroing trick solves the problem caused by
class label ordering permutations during testing.
- Through the random predictions before fine-tuning, memorization overfitting is prevented as well.
- The zeroing trick makes MAML applicable for datasets with a varying number of classes per task.

Interestingly, the motivation for applying the zeroing trick, stated by Kao et al. [2022] <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite>, is entirely different. In general, Kao et al. [2022] <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite> want to unveil in what sense MAML encourages its models to learn general-purpose feature representations. 
They show that under some assumptions, there is a supervised contrastive learning (SCL) objective underlying MAML.

In SCL, the label information is leveraged by pulling embeddings belonging to the same class closer together while increasing 
the embedding distances of samples from different classes <d-cite key="DBLP:conf/nips/KhoslaTWSTIMLK20"></d-cite>.
This is achieved by contrasting examples within a batch to each other. If two examples share the same label, the SCL loss is designed
to increase their embeddings' similarity. If the label differs, it enforces the examples embedding similarity to decrease.
The SCL loss contains an explicit similarity measure, which distinguishes it from supervised learning.

More specifically, Kao et al. [2022] <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite> show that, in MAML without the zeroing trick, the outer-loop update for the encoder follows a noisy SCL loss under the following assumptions:
1. The encoder weights are frozen in the inner loop (EFIL assumption)
2. There is only a single inner loop update step.<d-footnote>Note that FOMAML technically follows a noisy SCL loss without this assumption. However, when applying the zeroing trick, this assumption is needed again for stating that the encoder update is following an SCL loss</d-footnote>

A noisy SCL loss means that cases can occur where the loss forces the model to maximize similarities between embeddings from samples of different classes. The outer-loop encoder loss in this setting contains an "interference term" which causes the model to pull together embeddings from different tasks or to pull embeddings into a random direction, with the randomness being introduced by random initialization of the classification head. Those two phenomena are termed *cross-task interference*
and *initialization interference*. Noise and interference in the loss vanish when applying the zeroing trick, and the outer-loop encoder loss turns into a proper SCL loss. Meaning that minimizing this loss forces embeddings of the same class/task together while pushing embeddings from the same task and different classes apart. 

Those findings are derived using a general formulation of MAML, with a cross-entropy loss, and the details are available in the paper <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite>. Also, a slightly simpler example is stated to give an intuition of MAML's SCL properties. We will briefly summarize it in the following to share this intuition with you. 

In experiments on the mini-ImageNet and Omniglot datasets, a decent increase in performance is reported for MAML with the zeroing trick compared to vanilla MAML.

### MAML's SCL Intuition
To get an intuition of how MAML relates to SCL, let's look at the following setup: an N-way one-shot classification task using MAML with Mean Squared Error (MSE) between the one-hot encoded class label and the prediction of the model. Furthermore, the EFIL assumption is made, the zeroing trick is applied, only a single inner loop update step is used, and only a single task is sampled per batch.

In this setting, the classification heads inner-loop update for a single datapoint looks like this:

$$
\mathbf{w}' = \mathbf{w} - \alpha (-g_{\phi} (\mathbf{x}_{1}^{tr}) \mathbf{t}_{1}^{tr\top})
$$

$\mathbf{t}_1^{tr}$ refers to the one-hot encoded class label belonging to $\mathbf{x}_1^{tr}$. In words, the features extracted for training example $\mathbf{x}_1^{tr}$ are added to column $\mathbf{w}_c$, with $c$ being the index of 1 in $\mathbf{t}_1^{tr}$. For multiple examples, the features of all training examples labeled with class $c$ are added to the $c^{th}$ column of $\mathbf{w}$.

Now, for calculating the model's output in the outer loop, the model computes the dot products of the columns $$ \\{\mathbf{w'} \\}_{c=1}^N $$ 
and the encoded test examples $$ g_{\phi}(\mathbf{x}_1^{test}) $$.
To match the one-hot encoded label as well as possible, the dot product has to be large when $$ \mathbf{t}_1^{test} $$ = $$1$$ at 
index $$c$$, and small otherwise. We can see that the loss enforces embedding similarity for features from the same classes while enforcing 
dissimilarity for embeddings from different classes, which fits the SCL objective.

## Initialization using prototypes
A more sophisticated approach for last-layer initialization in MAML is introduced in the paper

<p></p>
<span>&nbsp;&nbsp;&nbsp;&#9654;&nbsp;&nbsp;</span>Eleni Triantafillou et al. (ICLR, 2020) Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples <d-cite key="DBLP:conf/iclr/TriantafillouZD20"></d-cite> .
<p></p>

As one might guess from the name, <strong>Proto-MAML</strong> makes use of Prototypical Networks (PNs) for enhancing MAML. Unlike the two initialization strategies presented above, Proto-MAML does not force the classification head weights to be initialized identically for all classes before fine-tuning. Instead, it calculates class-specific initialization vectors based on the training examples. This solves some of the problems mentioned earlier (see [Conclusion & Discussion](#conclusion--discussion)), but also it adds another type of logic to the classification layer.

Let's revise how PNs work when used for few-shot learning for understanding Proto-MAML afterward:

Class prototypes $$\mathbf{c}_{c}$$ are computed by averaging over train example embeddings of each class, created by a feature extractor $$g_{\phi}(\mathbf{x})$$.
For classifying a test example, a softmax over the distances (e.g., squared Euclidean distance) between class prototypes $$ \mathbf{c}_{c} $$ and example embeddings $$g_{\phi}(\mathbf{x}^{test})$$ is used, to generate probabilities for each class.

When using the squared Euclidean distance, the model's output logits are expressed as:

$$ 
\begin{align*}
&- \vert \vert g_{\phi}(\mathbf{x}) - \mathbf{c}_c \vert \vert^2 \\ =& −g_{\phi}(\mathbf{\mathbf{x}})^{\top} g_{\phi}(\mathbf{x}) + 2 \mathbf{c}_{c}^{\top} g_{\phi}(\mathbf{x}) − \mathbf{c}_{c}^{\top} \mathbf{c}_{c} \\ =& 2 \mathbf{c}_{c}^{\top} g_{\phi}(\mathbf{x}) − \vert \vert \mathbf{c}_{c} \vert \vert^2 + constant.
\end{align*}
$$

Note that the "test" superscripts on $\mathbf{x}$ are left out for clarity. $$−g_{\phi}(\mathbf{x})^{\top} g_{\phi}(\mathbf{x})$$ is disregarded here, as it's the same for all logits, and thus doesn't affect the output probabilities. When inspecting the left-over equation, we can see that it now has the shape of a linear classifier. More specifically, a linear classifier with weight vectors $$ \mathbf{w}_c = 2 \mathbf{c}_c^{\top} $$ and biases $$ b_c = \vert \vert \mathbf{c}_{c} \vert \vert^2 $$.

Returning to Proto-MAML, Triantafillou et al. [2020] <d-cite key="DBLP:conf/iclr/TriantafillouZD20"></d-cite> adapt vanilla MAML by initializing the classification head using the prototype weights and biases, as just discussed. The initialization happens before the inner loop for each task, and the prototypes are computed by MAML's own feature extractor. Afterward, the fine-tuning works as usual. Finally, when updating $\theta$ in the outer loop, the gradients flow also through the initialization of $$\mathbf{w}_c $$ and $$b_c$$, which is easy as they fully depend on $$ g_{\phi}(\mathbf{x})$$.

Note that because of computational reasons, Triantafillou et al. [2020] <d-cite key="DBLP:conf/iclr/TriantafillouZD20"></d-cite> refer to Proto-MAML as (FO-)Proto-MAML.

With Proto-MAML, one gets a task-specific, data-dependent initialization in a simple fashion, which seems super nice. For computing the model's output logits after classification head initialization, dot products between class prototypes and embedded examples are computed, which again seems very reasonable.

One could argue that in the one-shot scenario, Proto-MAML doesn't learn that much in the inner loop beside the initialization itself. This happens as the dot product between an embedded training example and one class prototype (which equals the embedded training example itself for one class) will be disproportionately high. For a k-shot example, this effect might be less, but still, there is always one training example embedding within the prototype to compare. Following this thought, the training samples would rather provide a useful initialization of the final layer than a lot of parameter adaptation. 

Proto-MAML is claimed to outperform the approaches, K-nearest neighbours, Finetune, MatchingNet, ProtoNet, fo-MAML and RelationNet on most sub-datasets of MetaDataset <d-cite key="DBLP:conf/iclr/TriantafillouZD20"></d-cite>, like ILSVRC-2012 or Omniglot. 

## What else is there?
Before proceeding to [Conclusion & Discussion](#conclusion--discussion), here are some pointers to methods that did not perfectly fit the topic but which are closely related:

The first method worth mentioning is called Latent Embedding Optimization (LEO) <d-cite key="DBLP:conf/iclr/RusuRSVPOH19"></d-cite>. The authors encode the training data in a low dimensional subspace, from which model parameters $\theta$ can be generated. In the example presented, $\theta$ consists only of $\mathbf{w}$, so for the first inner-loop iteration, this would perfectly fit our initialization topic. The low-dimensional code is generated using a feed-forward encoder, as well as a relation network. Using the relation network allows LEO to consider relations between the training examples of different classes. Very similar classes, for example, might require different decision boundaries than more distinct classes, hence the intuition.

LEO deviates from the initialization scheme, however, as optimization is done in the low dimensional subspace and not on the model's parameters directly. It is stated that optimizing in a lower dimensional subspace helps in low-data regimes.

Another related method is called MetaOptNet <d-cite key="DBLP:conf/cvpr/LeeMRS19"></d-cite>. In this approach, convex base learners, like support vector machines, are used as the classification head. Those can be optimized till convergence, which solves, e.g., the problem of varying performance due to random class label orderings.

## Conclusion and Discussion
To conclude, we've seen that a variety of problems can be tackled by using initialization strategies for MAML's linear classification head, including:
- Varying performance due to random class label orderings
- Ability of MAML to work on datasets where the number of classes per task varies
- Memorization overfitting
- Cross-task interference
- and Initialization interference.

Furthermore, for all the approaches presented, a decent gain in performance is reported in comparison to vanilla MAML. It seems, therefore, very reasonable to spend some time thinking about the last layer initialization.

Looking at the problems mentioned and variants discussed in more detail, we can state that all the different variants make MAML <strong>permutation invariant with regard to class label orderings</strong>. UnicornMAML and the zeroing trick solve it by uniform initialization of $\mathbf{w}$. In Proto-MAML, the initialization adapts to the class label assignments, so it's permutation invariant as well.

Also, all variants are compatible with <strong>datasets where the number of classes per task varies</strong>. In UnicornMAML, an arbitrary number of classification head vectors can be initialized with the single meta-learned classification head weight vector. When zero-initializing the classification head, the number of classes per task does not matter as well. In Proto-MAML, prototypes can be computed for an arbitrary number of classes, so again, the algorithm works on such a dataset without further adaption.

Next, UnicornMAML and the zeroing trick solve <strong>memorization overfitting</strong>, again by initializing $\mathbf{w}$ identically for all classes. Proto-MAML solves memorization overfitting as well, as the task-specific initialization of $\mathbf{w}$ itself can be interpreted as fine-tuning.

<strong>Cross-task interference</strong> and <strong>initialization interference</strong> are solved by the zeroing trick. For the other methods, this is harder to say, as the derivations made by Kao et al. [2022] <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite> are quite a case specific. Intuitively, Proto-MAML should solve cross-task interference, as the classification head is reinitialized after each task. 
Initialization interference is not solved by either ProtoMAML or UnicornMAML, as random initialization before the beginning of meta-training remains.

Note that in discussion with a reviewer, Kao et al. [2022] <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite> state that the main results they show are achieved by models which had the zeroing trick implemented but which didn't follow the EFIL assumption. They argue that using only the zeroing trick still enhances supervised contrastiveness. This kind of puts their whole theory into perspective, as without the EFIL assumption, MAML with the zeroing trick is neither an SCL algorithm nor a noisy SCL algorithm. Still, noticeable performance gains are reported though.

The question arises whether the whole theoretical background is needed or whether the zeroing tricks benefit is mainly the identical initialization for all classes, like in UnicornMAML. It would be nice to see how the single learned initialization vector in UnicornMAML turns out to be shaped and how it compares to the zeroing trick. While the zeroing trick reduces cross-task noise and initialization noise, a single initialization vector can weight some features as more important than others for the final classification decision across tasks.

In contrast to the uniform initialization approaches, we have seen Proto-MAML, where class-specific classification head vectors are computed for initialization based on the training data.

Finally, Ye & Chao [2022] <d-cite key="DBLP:conf/iclr/YeC22"></d-cite> compare the performance between Proto-MAML and UnicornMAML on MiniImageNet and TieredImageNet. UnicornMAML performs slightly better here in the one- and five-shot settings. 
Kao et al. [2022] <d-cite key="DBLP:conf/iclr/KaoCC22"></d-cite> report that MAML with the zeroing-trick outperforms unmodified MAML on the mini-ImageNet and Omniglot datasets. They do not provide a benchmark score, however.
