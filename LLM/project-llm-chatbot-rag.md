# **Mini-Project: Build a Local Open-Source LLM Chatbot With RAG**
> Talking to PDF documents with Google’s Gemma-2b-it, LangChain, and Streamlit

## **Introduction**

Large Language Models (LLMs) store a lot of knowledge. They have two main problems:
- their knowledge stops at their last training
- and they sometimes give wrong answers (hallucinate)

By using the RAG technique, we can give pre-trained LLMs access to very specific information as additional context when answering our questions.

In this mini-project, we will implement `Google’s LLM Gemma` with additional `RAG` capabilities using the `Hugging Face` transformers library, `LangChain`, and the `Faiss` vector database.

---
<h3><strong>How to use</strong></h3>

### Step 1.
Clone this repo:
```bash
git clone https://github.com/leoneversberg/llm-chatbot-rag.git
```
and
```bash
cd llm-chatbot-rag
```

### Step 2. 
Install the requirements
```bash
pip install -r requirements.txt
```

### Step 3.
You need to create `.env` file containing the line
```
ACCESS_TOKEN = <your hugging face token>
```

### Step 4.
Run with
```python
streamlit run src/app.py
```
> [!NOTE]  
> If you do not have a compatible GPU, try setting `device="cpu"` for the model of `src/app.py` and remove the `quantization config` in line 31 of `src/model.py`.


---
## **Methodology & Results**
 
![Overview of the RAG pipeline implementation](https://pbs.twimg.com/media/GKBY6UpWsAAlr94?format=jpg&name=900x900)

### **Retrieval-Augmented Generation (RAG)** 
The basic idea of RAG is:
- Start with a knowledge base (e.g. Wikipedia documents) transformed into dense vector representations (embeddings) using encoder model.
- Transform a user's question into an embedding vector using the same encoder model.
- Find vectors similar to the question's embedding from the knowledge base using a similarity metric. This process is the `retriever component`.
- Comebine the question and the context from the retrieved documents and feed them into a LLMs, called `generator component`, to get the answer.



### **Generator Component: LLM Model**
The generator is an LLM that take text (a question) as input and produces new text as output. For this porject, I choose Google's recently released model [Gemma-2b-it](https://huggingface.co/google/gemma-2b-it).

> [!NOTE]  
> To use `Gemma` we need to agree to Google’s [terms of use](https://www.kaggle.com/models/google/gemma/license/consent). By verifying through Hugging Face we can pass our Hugging Face access token to the transformers API.

Now we can initialize our `Gemma LLM model`.
```python
!pip install torch transformers bitsandbytes accelerate
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN") # reads .env file with ACCESS_TOKEN=<your hugging face access token>

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=ACCESS_TOKEN)
quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                         bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(model_id, 
                                             device_map="auto", 
                                             quantization_config=quantization_config,
                                             token=ACCESS_TOKEN)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

```

Next, We build the `generator()` function can be used to simply answer a question or to answer a question with additional context (which we will retrieve from documents).

Now, we can write our `LLM model inference` function.
```python
def generate(question: str, context: str):
    if context == None or context == "":
        prompt = f"""Give a detailed answer to the following question. Question: {question}"""
    else:
        prompt = f"""Using the information contained in the context, give a detailed answer to the question.
            Context: {context}.
            Question: {question}"""
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer.encode(
        formatted_prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=250,
            do_sample=False,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = response[len(formatted_prompt) :]  # remove input prompt from reponse
    response = response.replace("<eos>", "")  # remove eos token
    return response

```
Here is a visual summary of the model inference process:
![](https://archive.is/BKoAo/d1fb51826346dd0af32bf4a015d8c73e9c04e6b3.webp)

Let’s test our model with a question without additional context:
```python
print(generate(question="How are you?", context=""))

# >> As an AI, I do not have personal experiences or feelings, so I cannot answer this question in the same way a human would. I am a computational system designed to assist with information and tasks.

# >> I am functioning well and ready to assist you with any questions or tasks you may have. If you have any specific questions or need assistance with a particular topic, please let me know.
```


### **Retriever Component: Encoder Model + Similarity Search**
The encoder model compresses text into a dense vector that encodes the information into a high-dimensional feature space.

For this project, I choose the `all-MiniLM-L12-v2` encoder model, which is **only 120 MB** in size and **encodes text into a 384-dimensional vector**. 
> A list of pre-trained encoder models can be found at [sbert.net](https://www.sbert.net/docs/pretrained_models.html) and [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) that ranks higher (to improve performance)

Let’s implement the encoder model:
```python
from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

encoder = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L12-v2', 
    model_kwargs = {'device': "cpu"}
)
```
We can test our encoder using the function `embed_query()`:
```python
embeddings = encoder.embed_query("How are you?")

print(embeddings)
# >> [-0.03747698292136192, -0.02319679595530033, ..., -0.07512704282999039]

print(len(embeddings))
# >> 384
```

### **Document Loader and Text Splitter**
Now let’s build our knowledge base from multiple PDF documents.

Since a single PDF file can have hundreds of pages, we need to break it down into smaller chunks that we can feed into a language model.

The idea is to store smaller chunks of our documents as vectors in a vector database, and then search for useful chunks using a similarity metric when we ask a new question.
```python
!pip install pypdf tiktoken langchain sentence-transformers
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load PDFs
loaders = [
    PyPDFLoader("/path/to/pdf/file1.pdf"),
    PyPDFLoader("/path/to/pdf/file2.pdf"),
]
pages = []
for loader in loaders:
    pages.extend(loader.load())

# split text to chunks
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
     ),
     chunk_size=256,
     chunk_overlap=32,
     strip_whitespace=True,
)

docs = text_splitter.split_documents(pages)
```
By using the function from_huggingface_tokenizer() we define that the length of our chunk size is measured by the number of tokens from our encoder model.

Let’s get some intuition about the chunk size and the chunk overlap:
```python
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

chunk_size=10
chunk_overlap=0
print(text_splitter.split_text(text))
# >> ['Lorem ipsum', 'dolor sit', 'amet,', 'consectetur', 'adipiscing', 'elit.']

chunk_size=20
chunk_overlap=10
print(text_splitter.split_text(text))
# >> ['Lorem ipsum dolor', 'ipsum dolor sit', 'dolor sit amet,', 'sit amet, consectetur', 'consectetur adipiscing', 'adipiscing elit.']
```

### **Vector Database**
Next, we create our vector database to store the encoded chunks of text from our documents. There are many choices for databases. For this tutorial, I’m going to use `Faiss`.

```python
!pip install faiss-cpu

from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

faiss_db = FAISS.from_documents(
    docs, encoder, distance_strategy=DistanceStrategy.COSINE
)
```
To compute the similarity between vectors, we can choose from different `DistanceStrategy` options. Common choices to compute similarity are `EUCLIDIAN_DISTANCE`, `COSINE`, and `DOT_PRODUCT`.

### **User Interface with Streamlit**
Streamlit can be used to quickly create a user interface for our chatbot.

Go  to `src/app.py` for code.

---
## **Conclusion**
RAG is an exciting technique that gives LLMs access to external knowledge.

One interesting use case is retrieving knowledge from PDF manuals, which can be hundreds of pages long.

Sometimes the LLM answers follow the given context very closely, and sometimes general knowledge is included in the answers. This could probably be fixed by using a different prompt template in the `model.generate()` function.

It is easy to imagine a near future where our home appliances will be connected to LLMs with access to their own user manual. Then, we could simply ask them for help with our problems.

## **References**
[1] P. Lewis et al. (2021), [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401v4), arXiv:2005.11401

[2] N. Muennighoff, N. Tazi, L. Magne, N. Reimers (2023), [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316), arXiv:2210.07316

[3] [How to Build a Local Open-Source LLM Chatbot With RAG - medium.com](https://towardsdatascience.com/how-to-build-a-local-open-source-llm-chatbot-with-rag-f01f73e2a131)

### **Programming Resources**
- Full working code: https://github.com/leoneversberg/llm-chatbot-rag
- Google’s Gemma model: https://huggingface.co/google/gemma-2b-it
- LangChain documentation for retrieval: https://python.langchain.com/docs/modules/data_connection/
- Streamlit chatbot documentation: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps









