
-------------------------------------
## **Week 15/7/2024 - 21/7/2024**
<h6><p align="right">
    <a href="../README.md#my-learning">click to back..</a>
</p></h6>

[<h6 style="text-align: right;"> </h6>]()

### Table of Contents
- [Recurrent Neural Networks (RNNs)](#-recurrent-neural-networks-rnns)
- [Long Short-Term Memory (LSTM)](#-long-short-term-memory-lstm)
- [Web Scraping](#-web-scraping)
- [Pre-training, Fine-tuning and LoRA/QLoRA](#-pre-training-fine-tuning-and-loraqlora)
- 


This week, I learn:

### >> Recurrent Neural Networks (RNNs)


> #### ***Resources:***
> [9.4. Recurrent Neural Networks - d2l.ai book](https://www.d2l.ai/chapter_recurrent-neural-networks/rnn.html)  
>
> [Recurrent Neural Networks (RNNs), Clearly Explained!!! - StatQuest with Josh Starmer](https://www.youtube.com/watch?v=AsNTP8Kwu80)


<h6><p align="right">
    <a href="#week-1572024---2172024">go to top..</a>
</p></h6>

---
### >> Long Short-Term Memory (LSTM)
> #### ***Resources:***
> [Understanding LSTM Networks - colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
>
> [10.1. Long Short-Term Memory (LSTM)](https://www.d2l.ai/chapter_recurrent-modern/lstm.html)







<h6><p align="right">
    <a href="#week-1572024---2172024">go to top..</a>
</p></h6>

---
### >> Web Scraping
> #### ***Resources:***
> [Web Scraping With 5 Different Methods: All You Need to Know - Yennhi95zz from Medium](https://heartbeat.comet.ml/web-scraping-with-5-different-methods-all-you-need-to-know-403a59fceea0)


#### Method 1: BeautifulSoup and Requests

#### Method 2: Scrapy

#### Method 3: Selenium

#### Method 4: Requests and lxml

#### Method 5: How to Use LangChain for Web Scraping


<h6><p align="right">
    <a href="#week-1572024---2172024">go to top..</a>
</p></h6>
---

### >> Pre-training, Fine-tuning and LoRA/QLoRA
> #### ***Resources:***
> [The Novice's LLM Training Guide](https://rentry.org/llm-training)
>
> [Fine-Tuning LLMs : Overview, Methods, and Best Practices](https://www.turing.com/resources/finetuning-large-language-models)


#### 🧠 Pre-training:
- **Purpose**: Learn general language knowledge and patterns from a large and diverse dataset.
- **Dataset**: Massive, often in terabytes, containing varied and unlabeled text data.
- **Process**:
    - **Tokenization**: Convert text into token IDs using a tokenizer.
    - **Learning Objective**: Predict the next word in a sequence (Causal Language Modeling) or fill in missing words (Masked Language Modeling).
    - **Training**: Optimize model parameters to understand language structures, grammar, and semantics.
- **Outcome**: A model proficient in general language understanding, capturing broad linguistic knowledge.

<p align="center">
<img src="https://images.prismic.io/turing/6564bfb0531ac2845a2562f3_Finetuning_process_49bc08a9e9.jpg?auto=format,compress" width="80%" >
</p>

#### 🎯 Fine-tuning:
- **Purpose**: Specialize the pre-trained model for a specific task or domain.
- **Dataset**: Smaller, task-specific, and labeled data relevant to the desired task (e.g., instruction-response pairs).
- **Process**:
    - **Initialization**: Use the parameters from the pre-trained model.
    - **Task-specific Training**: Train on the labeled dataset to minimize a task-specific loss function.
    - **Optimization**: Adjust parameters using gradient-based algorithms (e.g., `SGD`, `Adam`).
    - **Enhancements**: Apply techniques like `learning rate scheduling`, `regularization`, and `early stopping` to prevent overfitting and improve generalization.
- **Outcome**: A model optimized for specific tasks, with enhanced performance in the targeted area.


#### 🛠️ LoRA (Low-Rank Adaptation)
- **Purpose**: Address the high computational cost of fine-tuning large language models.
- **Method**:
    - **Update Matrices**: Introduces pairs of rank-decomposition weight matrices to existing model weights.
    - **Training Focus**: Only the newly added weights are trained, preserving the pretrained weights.

- **Advantages**:
    - **Preservation of Pretrained Weights**: Maintains the frozen state of existing weights, preventing catastrophic forgetting and retaining existing knowledge.
    - **Portability**: The rank-decomposition matrices have significantly fewer parameters, making the trained weights easy to transfer and utilize in other contexts.
    - **Integration with Attention Layers**: LoRA matrices are incorporated into the attention layers, with an adaptation scale parameter controlling the model's adjustment to new data.
    - **Memory Efficiency**: Allows fine-tuning tasks to be run with less than 3x the compute required for native fine-tuning.


#### 🚀 QLoRA (Quantized Low Rank Adapters)
- **Purpose**: Further improve memory efficiency and maintain high performance during fine-tuning of large language models.
- **Key Innovations**:
    - **4-bit Quantization**: Backpropagation through a frozen, 4-bit quantized pretrained model into LoRA.
    - **New Data Type**: Utilizes 4-bit NormalFloat (NF4) to optimally handle normally distributed weights.
    - **Double Quantization**: Reduces average memory footprint by quantizing the quantization constants.
    - **Paged Optimizers**: Effectively manages memory spikes during the fine-tuning process.


<h6><p align="right">
    <a href="#week-1572024---2172024">go to top..</a>
</p></h6>

---
### Parameter Efficient Fine-Tuning (PEFT)

