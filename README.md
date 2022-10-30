# Projects-For-NLP-Bigenner

Typical projects for NLP beginners.

## Environment
Python 3.8 \
PyTorch 1.0 or above

## Tutorial

- PyTorch Tutorail: https://pytorch.org/tutorials/
- PyTorch Tutorial: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials
- NLP Tutorial: https://github.com/graykode/nlp-tutorial


## P1 Word Embeddings
### Learning Word Embeddings    
   
   - Paper1: Distributed Representations ofWords and Phrases and their Compositionality (NIPS 2013) 
   - Paper2: Learning Sentiment-Specific Word Embedding for Twitter Sentiment Classification (ACL 2014)
   - Paper3: Improving Implicit Discourse Relation Recognition with Discourse-specificWord Embeddings (ACL 2017)
   1. SimpCBOW: Learning general word embeddings based on unlabeled data using the CBOW model in Paper 1 (simplified here, just for demo). 
   2. DSWE: Learn task-specific word embedding via the Connective classification task as that in Paper 3, the labeled data is necessary. 
            DSWE is inspired by the model in Paper2, which is proposed to learn the sentiment-specific word embeddings.
   3. Data: https://pan.baidu.com/s/12uqXtjCBud0-A9Ilda1kOg?pwd=324q, password: 324q
            https://drive.google.com/file/d/1SeC3OcGI5MZh2G8oF5mTm1KP0OJUwhsQ/view?usp=sharing
   4. Result (call distance.py to show the top-n most similar words). The most similar words in CBOW and DSWE to 'not', 'good' and 'apple' are shown, respectively.
   
   
   <!--- ![DSWE](pic/test.jpg) -->
   CBOW
   ![DSWE](pic/cbow.jpg)
   
   DSWE
   ![DSWE](pic/dswe1.jpg)



## P2 Text Classification
### Document-level Sentiment Classificaiton 
   
   - Paper1: Hierarchical Attention Networks for Document Classification (NAACL 2016) 
   - Paper2: Learning Semantic Representations of Users and Products for Document Level Sentiment Classification (ACL 2015)
   - Paper3: Improving Review Representations with User Attention and Product Attention for Sentiment Classification (AAAI 2018)
   - Web: 
   1. Hierarchical Attention Model without User and Product Attributes (HAN) : P2 Text Classification/SentimentUP/HAN. 
   2. Hierarchical Attention Model With User and Product Attributes (HAN_UP) : P2 Text Classification/SentimentUP/HAN_UP. 
                   Different from the model in AAAI 2018, we leverage User and Prodcut attributes in different ways.
                   Specifically, the User attribute is incorporated at word-level, sentence-level and document level in a fusion way,  
                   while the Product attribute is incorporated into word-level and sentence-level attention mechanisms following AAAI 2018.
   4. Data: the Yelp 2013, Yelp 2014, IMDB datasets are provided by authors of Paper2. 
   5. Word embeddings: provided by authors of Paper3
   6. Download data and embeddings: https://drive.google.com/file/d/18dXcCXl5txAf-WaxlXgyuPVdUkbXCkmF/view?usp=sharing
   
   - ACC Results
   
   Model       |    IMDB      |    Yelp 2013  | Yelp 2014
   ----------- |--------------|---------------|-----------  
   HAN         |    48.4      |    64.5       |   65.1
   AAAI 2018   |    55.0      |    68.3       |   68.6
   HAN_UP      |    56.6      |    68.9       |   69.2




## P3 NER&Event
### Chinese Named Entity Recognition
- Paper1: Bidirectional LSTM-CRF Models for Sequence Tagging (Arxiv 2015)
- Paper2: A Local Detection Approach for Named Entity Recognition and Mention Detection (ACL 2017)
- Paper3: 基于深度学习的命名实体识别综述 (中文信息学报 2021)
- Paper4: 基于深度学习的中文命名实体识别最新研究进展综述 (中文信息学报 2022)
- Web: https://zhuanlan.zhihu.com/p/44042528

1. BERT + Softmax
2. BERT + CRF
3. BERT + Span

to be continiued...


## P4 Parser
### Dependency Parser 
   
   - Paper: Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations (TACL 2016) 
   1. Transition-based model: P4 parser/Dependency Parser/transition-parser 
   2. Graph-based model: P4 parser/Dependency Parser/graph-parser
   3. Data: https://drive.google.com/file/d/1z8Q-dIgqJSA4sWC69AXxJjY3VLtLWdeJ/view?usp=sharing
   4. Reference: https://github.com/elikip/bist-parser/


## P5 Seq2Seq
### Machine Translation 

to be continiued...



