## RLHF for LLMs

### Opportunity for RL

#### Hallucination

+ Behavior cloning (**BC**) sometimes induces hallucination

  ![粘贴图像 0 3](https://ahrefs.com/blog/wp-content/uploads/2020/09/pasted-image-0-3.png)

  LLMs have "**knowledge graph**" stored in its weights; small scale fine-tuning learns to operate on the graph and outputs token prediction

  + **Information Asymmetry**

    + if labeler knows something while the net doesn't, then the model is encouraged to **hallucinate**;

    + if the net knows something but the labeler doesn't, then the model is encouraged to **withhold** information;

  + **Negativity Hypothesis**

    + BC only presents the model with **correct**/positive Q-A pairs, meaning the demonstrator controls the learning process entirely;

  + **Conclusion**: BC target should depend on network's knowledge (models that are trained using targets computed by another agent will always have hallucination problem)

+ **Uncertainty** formalization

  + **metric**: for single-word answers, **log loss** is a proper scoring function;

    ![image-20230729192411948](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20230729192411948.png)

  + align **log probability** prediction with **uncertainty** through RL

  + in principle, the reward model (RM) should have the same knowledge graph as the policy model

+ **Informativeness & correctness** tradeoff

  + Long-form answers are often mix of correct and wrong information;
  + **metric**: for long-form answers, a potential metric is to compare the generated text and human labeled text, and check if they are consistent;

+ **Limitations** of RLHF: why does ChatGPT still hallucinates:

  + Model has to guess sometimes;
  + the current ranking based reward model doesn't impose correct penalty;
  + label errors;



#### WebGPT: an alternate to fix hallucination

+ **Retrieval & citing sources**

  + current events, detailed knowledge;
  + Information not in pre-training;
  + verifiability;

+ GPTs are good at information extraction from grounded context:

  + RL modeling

    + **Observation space**:

      ![image-20230729201501256](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20230729201501256.png)

    + Action space:

      ![image-20230729201513423](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20230729201513423.png)

    + Training Process:

      ![image-20230729201916582](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20230729201916582.png)

    + Testing:

      + do RL on the reward model;
      + searching through sampled space and re-rank the answer trajectories



#### Open problems

+ Incentivizing models to **accurately** express its uncertainty in words?

  ​	P(A wins over B) $\propto e^{R(A)-R(B)}$

+ Current reward model methodology doesn't impose correct penalties. How to penalize hedging and wrong behaviors?

+ (P != NP) It is easier to verify proof/solution than to generate.



#### Insights

Long-form answers are often a mix of right and wrong statements, however, human feedback are often sparse and rendered once at last of  the token. Therefore the key to **balance the trade-off of informativeness and correctness** is **credit-assignment**. 



There are two types of statements that might induce a wrong conclusion: **wrong argument** and **improper inference**. Modeling text generation in the light of RL, 

Supposing that human gives a negative feedback for a long-form answer, this penalty should be assigned to 







#### RL or BC ? Why should we prefer RLHF?

I know it is intuitively making sense that **RL** gives agent an inner incentive to learn a policy while **BC** only technically force the agent to imitate good behaviors, but how to theoritically explain why RL is better than BC? Is it that RL has better sample efficiency, or is that RL gets on-time feedback on its actual behaviors so that the target aligns with the model's internal knowledge?



#### Train a Reward Model

##### Comparion data

+ **Cross-entropy loss**: train on a dataset of comparions between multi model outputs on the same input
+ **Batch Element Approach**: instead of shuffling the comparison dataset from each prompt and use this dataset for training the reward model, they consider all the comparions as a single batch element. Therefore the RM can learn from overall ranking pattern rather than memorizing individual comparisons (overfitting).

**Loss design**:

![image-20230805225806040](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20230805225806040.png)

where $r_\theta (x, y)$ is the scalar output of the reward model for prompt $x$ and completion $y$ with parameters $\theta$ $y_w$ is the preferred completion out of the pair of $y_w$ and $y_l$ 

The reward model will serve as the value function (critic model, as of actor) for following RL.



##### RL

+ a KL penalty is added to prevent the fine-tuned model drift too far away from the SFT model

+ loss function:

  ![image-20230805231043625](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/image-20230805231043625.png)





#### Harms of LLMs

+ production of biased outputs
+ leakage of private data
+ generation of misinformation (hallucination)
+ being used maliciously

Making signiﬁcant progress on these problems is hard since well-intentioned interventions on LM behavior can have side-effects. For instance, efforts to reduce the toxicity of LMs can reduce their ability to model text from under-represented groups, due to prejudicial correlations in the training data.

**Way to mitigate toxicity**

+ input:
  + data filtering
  + Data augmentation
  + Human-in-the-loop data collection
  + word embedding regularization
+ training:
  + Safety-specific control tokens
  + null space projection
  + objective function engineering
  + causal mediation analysis
  + Steering the generation of language models using a second (usually smaller) language model
+ output:
  + blocking certain words or n-grams



#### Training Strategy

+ Instruct GPT

  + staring with various GPT-3 pretained language models (adaptable to a wide range of downstream tasks)

  + **SFT**: using a **cosine** learning rate decay and residual dropout of 0.2, then select the final SFT based on the RM score on the validation set
  + **RM** (reward modeling): 6B parameters; Batch element strategy;
  + **RL**: a random customer prompt is presented and a response to the prompt is expected; a KL penalty is added to prevent the fine-tuned model drift too far away from the SFT model



### RL4LLMs

#### **Challenges**

+ training instability due to the **combinatorial action space**
+ lack of open-source libraries and benchmarks customized for LM alignment

**Metrics** to evaluate human preferences:

+ Pairwise learned preference models
+ BERTScore
+ BLEURT

these function are not **per-token differentiable**: they can only offer quality estimates for full generations. RL naturally caters to optimizing these **non-differentiable, scalar objectives** for sequential decision-making problems.

**LM_based MDP Formulation**

Given a supervised dataset $D = {(x^i, y^i)}^N_{i=1}$ of $N$ examples, where $x \in X$ is a language input and $y \in Y$ is the target string.

#### **Markov Decision Process** (MDP)

**MDP notation**: $<S, A, R, P, \gamma, T>$ with a finite vocabulary $V$; $s \in S$, $S$ is the state space; $a \in A$, $A$ is the action space (which consists of a token from $V$)

**An episode**: 

+ **start of episode**: Sample a datapoint $(x, y)$ from the dataset and ends when the current time $t$ exceeds the horizon $T$ or an end of sentence (EOS) token is generated;
+ initial **state** $s$: $s_0=x=(x_0, \dots, x_m)$, $s \in S$, $S$ is the state space and $x_m \in V$;
+ **action** $a_t$: an action in the environment $a_t \in A$ consists of a token from the vocabulary set $V$;
+ transition function $P$: $S \times A \to \Delta(S)$ **deterministically** appends an action $a_t$ to the end of the state $s_{t-1} = (x_0, \dots, x_m, a_0, \dots, a_{t-1})$
+ **end of episode**: final state $s_{T} = (x_0, \dots, x_m, a_0, \dots, a_{T})$, an episode-wise reward is generated  $R: S \times A \times Y \to R^1$ which depends on the $(s_T, y)$

**Reward Metrics** 

+ **n-gram overlap metrics**

  + BLEU
  + ROUGE
  + SacreBLEU
  + METEOR

+ **Model-based semantic metrics**

  + BertScore

    ![v2-c5b4c5e29aed5a6c4a9235c5a3c7f860_720w](https://pic1.zhimg.com/80/v2-c5b4c5e29aed5a6c4a9235c5a3c7f860_720w.webp)

  + BLEURT

+ **Task-specific metrics**

  + CIDER
  + SPICE (for captioning/commonsense generation)
  + PARENT (for data-to-text)
  + SummaCZS (for factuality of summarization)

+ **Diversity/fluency/naturalness metrics**

  + perplexity
  + MSSTR (Mean Segmented Type Token Ratio)
  + shannon entropy over unigrams and bigrams
  + ratio of distinct n-grams over the total number of n-grams
  + count of n-grams that appear only once in the entire generated text

+ **task-specific**, **model-based human preference metrics**

  + classiﬁers trained on human preference data (InstructGPT)



#### On-policy RL Training

+ **optimization goal**: parameterized control policy: $\pi_\theta: S \to \Delta(A)$ to maximize long term discounted rewards over a trajectory $E_{\pi}[\sum^T_{t=0}\gamma^t R(s_t, a_t)]$

+ **policy network**: the agent's policy $\pi_\theta$ is initialized from the pre-trained LM model $\pi_\theta = \pi_0$

+ **value network**: the value network $V_\phi$ used to estimate the value function is also initialized from $\pi_0$ except for the final layer which is randomly initialized to output a single scalar value

+ **reward regularization**: add KL penalty to the original reward metric to prevent model from deviating too far from the pre-trained (initialized) LM $\pi _0$:

  ![image-20230806152956112](/Users/zhaorunchen/Library/Application Support/typora-user-images/image-20230806152956112.png)

  the KL coefficient $\beta$ is adapted dynamically during training:

  ![image-20230806180453128](/Users/zhaorunchen/Library/Application Support/typora-user-images/image-20230806180453128.png)



#### NLP Models

##### GPT-2

**Dataset**: 

+ **WebText**: extracted from web pages which have been curated/filtered by humans (they use a data-filter heuristic that scrapes all outbound links from Reddit, a social media platform, which receive at least 3 karma), and eventually accumulated to 8 million documents for a total of 40GB of text.
+ Remove all Wikipedia documents from WebText since it is a common data source for other datasets and could complicate analysis due to overlapping training data with test evaluation tasks.



**Useful tokenizer mapping:**

Some words alone corresponds to a token, some other words (usually names and technical terminology words that rarely used) correspond to multiple tokens.

+  ".": 13, however: "..": 492 which is different !!

+ "Ġ": This is a special symbol that indicates the beginning of a word. It is used to separate words within a text. For example, in the token "Ġit", the "Ġ" indicates that the word "it" follows; e.g.
  + "Ġquestion": 1808 == "question"
  + "ĠNo": 1400 == "No"
  + "Ġwith": 351 == "with"
  + "Ġbruises": 42232 == "bruises"
  + "ĠRe": 797 + "iki": 5580 == "Reiki"
  + "Ġ": 220 == ""



#### NLP Datasets

[SQuAD][https://rajpurkar.github.io/SQuAD-explorer/]

DROP

HellaSwag

WMT Translation

FLAN

T0 (T0++ variant)



#### NLP Techniques

##### [Sampling](https://www.cnblogs.com/miners/p/14950681.html)

![1364705-20210629160616624-1774623193](https://raw.githubusercontent.com/BillChan226/Notebook/main/image/1364705-20210629160616624-1774623193.png)

Strategies for sampling:

+ **Greedy Search** (Maximization)

  + select the token with max probability
  + Problem: if we always select the token with max probability, many repeated sentences will be generated (get stuck in loops)

+ **[Beam Search](https://zhuanlan.zhihu.com/p/82829880)**

  + always select the top K sub-sequences with max probability at each step
  + a midway between exhaustive search and greedy search; when K == V (num of vabularies), it is equivalent to exhaustive search; when K == 1, it is equivalent to greedy search

+ [**Temperature Sampling**](https://zhuanlan.zhihu.com/p/427186055)

  + devide the original **log** probability by temperature $\tau$

  + a reweighting technique to deal with ill-proportioned dataset distribution

  ![img](https://pic3.zhimg.com/80/v2-9525f6b9d48ddd7eb7a2a610192977da_1440w.webp)

+ [**Top-K Sampling**](https://zhuanlan.zhihu.com/p/267471193)

  + first sort the tokens w.t.r. possibility, then set the possibility after $k^{th}$ token to be 0
  + this technique can efficiently deal with long-tail distribution, however for broad distribution where the top $K$ tokens are not distinguished in possibility, this technique might limit the explorability of the model

  ![img](https://pic2.zhimg.com/80/v2-e95fdf3ba7313637e6e1a73318311df1_1440w.webp)

+ Top-P Samping (Nucleus sampling)

  + to remedy top-K samping, top-p sampling sum up the tokens with max possibility until the sum exceeds threshold $p$
  + this technique can handle long-tail distribution and maintain diversity



#### [Hugging Face](https://huggingface.co/)

Hugging face is a public library that enclose many transformer models (mainly NLP and CV), datasets. It also provides modules for users to easily access the library and train/evaluate their own models.

##### Transformers

2 most often used models:

+ AutoModelForCausalLM
+ AutoModelForSeq2SeqLM

Tokenizer: AutoTokenizer

##### Datasets

+ [Loading Datasets]()

+ [Process Datasets][https://huggingface.co/docs/datasets/process]

  + Dataset is stored in this format:

    ```
    {"train": Dataset({features:['text', 'label'], num_rows: 1000}),
     "test": Dataset({features:['text', 'label'], num_rows: 100})}
    ```

    

  + **sort()** and **shuffle()** lowers computation speed (about 10 times), because there is an extra step to get the row index to read using the indices mapping, and most importantly, you aren’t reading contiguous chunks of data anymore. As an alternative, we can first convert the dataset to an iterative dataset:

    ```python
    iterable_dataset = dataset.to_iterable_dataset(num_shards=128)
    ```

    and then use:

    ```
    shuffled_iterable_dataset = iterable_dataset.shuffle(seed=42, buffer_size=1000)
    ```

  + **train_test_split()** can help split the dataset if it does not originally have one:

    ```
    dataset.train_test_split(test_size=0.1, shuffle=True)
    ```

    now we can use the train set:

    ```
    datasets["train"]
    ```

  + **shard()** will divide a very large dataset into a predefined number of chunks and return the chunk with the index you provide:

    ```
    dataset.shard(num_shards=4, index=0)
    ```

  + **map()** takes a function and operate this function on the dataset as indicated:

    ```python
    def add_prefix(example):
        example["sentence1"] = 'My sentence: ' + example["sentence1"]
        return example
      
    updated_dataset = small_dataset.map(add_prefix)
    updated_dataset["sentence1"][:2]
    ['My sentence: Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
    "My sentence: Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
    ]
    ```

    

