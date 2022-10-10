# Projects-For-NLP-Bigenner

Typical projects for NLP beginners.

## Environment
python 3.8 \
pytorch 1.3 or above


## P2 Text Classification
### Document-level Sentiment Classificaiton 
   
   - Paper1: Hierarchical Attention Networks for Document Classification (NAACL 2016) 
   - Paper2: Learning Semantic Representations of Users and Products for Document Level Sentiment Classification (ACL 2015)
   - Paper3: Improving Review Representations with User Attention and Product Attention for Sentiment Classification (AAAI 2018)
   1. Hierarchical Attention Model (HAN), without User and Product Attributes: P2 Text Classification/SentimentUP/HAN. 
   2. Hierarchical Attention Model (HAN), With User and Product Attributes: P2 Text Classification/SentimentUP/HAN_UP. 
                   User attribute is incorporated at word-level, sentence-level and document level in a fusion way. 
                   Product attribute is incorporated into word-level and sentence-level attention mechanisms.
   4. data: the Yelp 2013, Yelp 2014, IMDB datasets are provided by authors of Paper2. 
   5. word embeddings: provided by authors of Paper3
   6. download data and embeddings: https://drive.google.com/file/d/18dXcCXl5txAf-WaxlXgyuPVdUkbXCkmF/view?usp=sharing


## P4 Parser
### Dependency Parser 
   
   - Paper: Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations 
   1. Transition-based model: P4 parser/Dependency Parser/transition-parser 
   2. Graph-based model: P4 parser/Dependency Parser/graph-parser
   3. data: https://drive.google.com/file/d/1z8Q-dIgqJSA4sWC69AXxJjY3VLtLWdeJ/view?usp=sharing
   4. reference: https://github.com/elikip/bist-parser/
