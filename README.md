# Lab-NLP-Deep-Learning

Clone the repository: `git clone https://github.com/elbourkadi/LAB2-NLP-DEEP-LEARNING.git`

This repository includes a project aimed at building deep learning skills by developing neural network architectures for natural language processing (NLP) using PyTorch and Sequence Models.

## Objective
The main purpose of this lab is to get familiar with PyTorch and build deep neural network architectures for NLP tasks using Sequence Models.

## Work to Do

### Part 1: Classification Task
1. **Data Collection**: Use web scraping libraries (Scrapy / BeautifulSoup) to collect Arabic text data on a single topic from several websites and prepare a dataset in the following format:

   | Text (Arabic Language) | Score |
   |-------------------------|-------|
   | Text 1                 | 6     |
   | Text 2                 | 7.5   |

   The score represents the relevance of each text (ranging from 0 to 10).

2. **Data Preprocessing**: Establish an NLP pipeline with steps like tokenization, stemming, lemmatization, stop word removal, discretization, etc., to clean and prepare the collected dataset.

3. **Model Training**: Train models using RNN, Bidirectional RNN, GRU, and LSTM architectures. Perform hyperparameter tuning to achieve the best performance.

4. **Evaluation**: Evaluate the models using standard metrics and additional metrics like BLEU score.

### Part 2: Transformer (Text Generation)
1. **Fine-Tuning**: Install PyTorch-Transformers and load the GPT-2 pre-trained model. Fine-tune it on a customized dataset (you can generate your own dataset).

2. **Text Generation**: Use the fine-tuned model to generate a new paragraph based on a given sentence.

## Tools
- Google Colab or Kaggle
- GitLab/GitHub

### Tutorial
You can follow this tutorial: [PyTorch-Transformers Tutorial](https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7)

## Results
This lab highlights the power of sequence models in NLP tasks, showcasing techniques like web scraping for data collection, preprocessing pipelines, and fine-tuning of pre-trained models for classification and text generation.

## Key Learnings
This lab provided hands-on experience with PyTorch and Sequence Models for NLP. It emphasized the importance of building robust preprocessing pipelines, experimenting with different neural architectures (RNNs, GRUs, LSTMs, Transformers), and understanding their respective strengths in NLP tasks. Fine-tuning pre-trained models like GPT-2 demonstrated the potential of leveraging transfer learning for advanced tasks such as text generation.

