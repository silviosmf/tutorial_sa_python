# README
# Installation Instructions
# To ensure the Jupyter notebooks function correctly, you need to install the following Python libraries. You can install these libraries using pip.

# Natural Language Toolkit (nltk)
   pip install nltk

# NumPy
   pip install numpy
 
# pandas
   pip install pandas

# Matplotlib
   pip install matplotlib
 
# seaborn
   pip install seaborn

# spaCy
   pip install spacy
 
# PyTorch
# Visit https://pytorch.org/get-started/locally/ to install PyTorch according to your system's specifications.

# scikit-learn
   pip install scikit-learn
 
# gensim
   pip install gensim
 
# transformers
    pip install transformers
 
## Downloading CBOW 50 Dimensions Word Embeddings
# To work with Word2Vec models, it is necessary to have pre-trained word embeddings. For this purpose, the CBOW 50 dimensions file from NILC (Núcleo Interinstitucional de Linguística Computacional) can be used. This file contains word vectors trained using the CBOW (Continuous Bag of Words) model, a popular Word2Vec architecture.
# To download the CBOW 50 dimensions file, visit: [NILC Word Embeddings Repository](http://nilc.icmc.usp.br/nilc/index.php/repositorio-de-word-embeddings-do-nilc)
# The use of these pre-trained embeddings is crucial for achieving good performance in natural language processing tasks. Pre-trained embeddings like the CBOW 50 dimensions provide a rich representation of words in a high-dimensional space, capturing semantic relationships and context. This can significantly improve the quality of models for tasks like sentiment analysis.

## Additional Notes:
# Some libraries like `word_tokenize`, `stopwords`, `Dataset`, `CountVectorizer`, `accuracy_score` are part of the above-mentioned packages and will be installed with them.
# Make sure you have Python installed on your system before running these commands.
# It's recommended to use a virtual environment for Python projects to avoid conflicts between different projects' dependencies.