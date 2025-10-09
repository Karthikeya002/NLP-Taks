# NLP-Tasks
# 📚 NLP Lab Tasks Collection

A comprehensive collection of Natural Language Processing tasks covering fundamental to advanced concepts. This repository contains 12 hands-on NLP implementations.

## 🗂️ Tasks Overview

### 🔍 **Tasks 1-7: Foundation & Core NLP**

#### **Task 1: 🎯 Regular Expressions**
- **Purpose**: Pattern matching and text extraction
- **Techniques**: Metacharacters, character classes, quantifiers, alternation
- **Applications**: SSN extraction, date parsing, text pattern matching

#### **Task 2: ✂️ Tokenization**
- **Tools**: NLTK vs Transformers
- **Methods**: `word_tokenize()` and BERT tokenizer
- **Output**: Comparative tokenization results

#### **Task 3: 🧠 Text Generation with LSTM**
- **Model**: Sequential LSTM with Embedding
- **Features**: Lemmatization, sequence padding
- **Training**: 10 epochs with 100% accuracy

#### **Task 4: 📊 Document Morphology Analysis**
- **Process**: Stopword removal, frequency distribution
- **Output**: Most common words analysis
- **Key Terms**: language, natural, NLP, computers

#### **Task 5: 📝 N-gram Language Model**
- **N-grams**: Uni, Bi, and Tri-grams
- **Corpus**: Reuters dataset
- **Feature**: Text generation using probability distributions

#### **Task 6: ⚖️ N-gram Smoothing**
- **Algorithm**: Laplace smoothing
- **Purpose**: Handle unseen n-grams
- **Implementation**: Custom text generation

#### **Task 7: 🏷️ POS Tagging with HMM**
- **Model**: Hidden Markov Model
- **Data**: Treebank corpus
- **Application**: Part-of-speech tagging

### 🚀 **Tasks 8-12: Advanced Applications**

#### **Task 8: ⚡ Advanced POS Taggers**
- **Models**: HMM vs Maximum Entropy
- **Accuracy**: HMM: 95.04%, MaxEnt: 96.52%
- **Comparison**: Performance evaluation on standard text

#### **Task 9: 📈 Tagger Performance Comparison**
- **Metrics**: Accuracy scoring
- **Models**: HMM vs Log-Linear
- **Result**: Both achieved 100% on test sentence

#### **Task 10: 🌐 Web Text Analysis**
- **Tools**: BeautifulSoup + spaCy
- **Extraction**: Nouns, Verbs, Adjectives, Entities
- **Source**: HTML document parsing

#### **Task 11: 🔗 Text Chunking from Web**
- **Architecture**: LSTM-based chunker
- **Source**: Wikipedia NLP page
- **Output**: Text segmentation into chunks

#### **Task 12: 📊 Data Visualization**
- **Libraries**: Matplotlib & Seaborn
- **Charts**: Bar plots comparison
- **Purpose**: Result visualization techniques

## 🛠️ Technologies Used

- **Python Libraries**: NLTK, spaCy, BeautifulSoup, Transformers
- **ML Frameworks**: TensorFlow/Keras, PyTorch
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: NumPy, Pandas

## 📦 Installation & Setup

```bash
# Install required packages
pip install nltk spacy beautifulsoup4 transformers tensorflow torch matplotlib seaborn pandas numpy

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('popular')"

🎯 Key Learning Outcomes
✅ Regular expression mastery

✅ Tokenization techniques

✅ Neural text generation

✅ Morphological analysis

✅ N-gram modeling & smoothing

✅ POS tagging with statistical models

✅ Web scraping for NLP

✅ Data visualization for NLP results

📈 Performance Highlights
HMM Tagger: 95.04% accuracy

MaxEnt Tagger: 96.52% accuracy

LSTM Model: 100% training accuracy

Web Extraction: Successful HTML parsing
