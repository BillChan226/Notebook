## Efficient RLHF Fine-tuning

**Experiment Design**

+ **Model**:

  + GPT-2:

    + GPT2: 137M

    + GPT2-xl: 1.5B

  + Llama-2:

    + Llama-2-7b

    + Llama-2-13b

    + Llama-2-30b

    + Llama-2-70b (TBD)

  + GPT-3.5

    + GPT-3.5.-turbo: 175B

      

+ **Task**:

  + **sentiment**: imdb
  + **harmlessness**: HH (Anthropic’s Helpful and Harmless (HH) dataset)

+ **Metric**:

  + GPT-4

    

1. **Win Rate** over vanilla RLHF, analysis on model size
2. Comparison with SOTA **outcome-supervised** and **process-supervised RLHF** algorithms