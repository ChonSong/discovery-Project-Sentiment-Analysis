# Comprehensive Sentiment Analysis Project
## A Deep Learning Approach to Social Media Sentiment Classification

**Authors**: Discovery Project Team  
**Date**: January 2025  
**Objective**: Develop and optimize neural network architectures for sentiment analysis using multiple deep learning approaches

---

## Abstract

This comprehensive project implements and compares multiple neural network architectures for sentiment analysis of social media data from the Exorde dataset. We systematically progress through 12 key phases: from basic model implementation to advanced optimization techniques, incorporating insights from foundational literature in natural language processing and deep learning. Our implementation includes RNN, LSTM, GRU, and Transformer architectures with various enhancements including attention mechanisms, bidirectional processing, and pre-trained embeddings.

The project demonstrates a methodical approach to machine learning model development, progressing from baseline implementations to sophisticated optimized models. We achieve significant performance improvements through systematic hyperparameter tuning, architectural enhancements, and advanced training techniques, ultimately reaching competitive F1 scores on multi-class sentiment classification.

---

## Table of Contents

1. [Setup & Prerequisites](#1-setup--prerequisites)
2. [Core Utilities & Model Definitions](#2-core-utilities--model-definitions)
3. [Data Acquisition](#3-data-acquisition)
4. [Model Visualization](#4-model-visualization)
5. [Enhanced Architecture Comparison](#5-enhanced-architecture-comparison)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Foundational Improvements (Baseline V2)](#7-foundational-improvements-baseline-v2)
8. [Advanced Training Demonstration](#8-advanced-training-demonstration)
9. [Final Hyperparameter Optimization](#9-final-hyperparameter-optimization)
10. [Comprehensive Error Analysis](#10-comprehensive-error-analysis)
11. [Final Model Training](#11-final-model-training)
12. [Final Report Generation](#12-final-report-generation)

---

## Literature Review

Our approach is grounded in foundational research in natural language processing and deep learning. This section reviews five key papers that inform our architectural choices and optimization strategies.

### 1. "Attention Is All You Need" (Vaswani et al., 2017)

**Citation**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

**Key Contributions**:
- Introduced the Transformer architecture based solely on self-attention mechanisms
- Demonstrated superior performance to RNNs/LSTMs while enabling parallelization
- Established the foundation for modern language models (BERT, GPT, etc.)

**Application to Our Project**:
This paper provides the theoretical foundation for our Transformer implementation. We leverage the self-attention mechanism to capture long-range dependencies in social media text, which often contains complex linguistic structures. Our implementation includes positional encodings and multi-head attention as described in the original paper, adapted for sentiment classification tasks.

### 2. "Bidirectional LSTM-CRF Models for Sequence Tagging" (Huang et al., 2015)

**Citation**: Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. arXiv preprint arXiv:1508.01991.

**Key Contributions**:
- Demonstrated the effectiveness of bidirectional processing for sequence understanding
- Showed that backward context is crucial for understanding linguistic meaning
- Established bidirectional LSTMs as a standard for sequence processing

**Application to Our Project**:
This research validates our implementation of bidirectional variants for RNN, LSTM, and GRU models. For sentiment analysis, understanding both preceding and following context is crucial. For example, in "The movie was not bad at all," the sentiment is only clear when considering the complete phrase. Our bidirectional models capture this dual-context information effectively.

### 3. "A Structured Self-Attentive Sentence Embedding" (Lin et al., 2017)

**Citation**: Lin, Z., Feng, M., Santos, C. N. D., Yu, M., Xiang, B., Zhou, B., & Bengio, Y. (2017). A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130.

**Key Contributions**:
- Introduced self-attention for sentence-level representations
- Provided interpretable attention weights showing model focus
- Demonstrated superior performance over simple pooling strategies

**Application to Our Project**:
This paper directly informs our attention-enhanced RNN, LSTM, and GRU models. Instead of using only the final hidden state, we implement self-attention mechanisms that weight the importance of each word in the sequence. This approach is particularly valuable for sentiment analysis where specific words or phrases carry disproportionate emotional weight.

### 4. "GloVe: Global Vectors for Word Representation" (Pennington et al., 2014)

**Citation**: Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).

**Key Contributions**:
- Introduced global matrix factorization approach to word embeddings
- Captured both global and local statistical information
- Demonstrated strong performance on word analogy and similarity tasks

**Application to Our Project**:
This research supports our use of pre-trained embeddings to initialize our models. GloVe embeddings provide rich semantic representations learned from large corpora, giving our models a significant head start compared to random initialization. This is especially important for sentiment analysis where semantic relationships between words are crucial for understanding emotional nuances.

### 5. "Bag of Tricks for Efficient Text Classification" (Joulin et al., 2016)

**Citation**: Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016). Bag of tricks for efficient text classification. arXiv preprint arXiv:1607.01759.

**Key Contributions**:
- Introduced FastText for efficient text classification
- Demonstrated that simple approaches can be highly effective
- Showed the importance of n-gram features and subword information

**Application to Our Project**:
While we focus on deep learning approaches, this paper provides important baseline insights. It reminds us that complex models must significantly outperform simpler alternatives to justify their computational cost. We use this perspective to validate that our neural networks provide meaningful improvements over traditional bag-of-words approaches.

---

## Project Structure

The project is broken down into the following key stages, each corresponding to one or more of the original Python scripts:

### Setup & Prerequisites: 
Import libraries and configure the environment.

### Core Utilities & Model Definitions: 
Define all helper functions, training loops, and neural network architectures.

### Data Acquisition: 
Download the dataset.

### Model Visualization: 
Generate diagrams of the model architectures to understand their structure.

### Enhanced Architecture Comparison: 
Conduct a broad comparison across all model variants to identify the most promising candidates.

### Hyperparameter Tuning: 
Perform a grid search on key hyperparameters for the top models.

### Foundational Improvements (Baseline V2): 
Establish a stronger baseline using the insights from tuning.

### Advanced Training Demonstration: 
Showcase the impact of pre-trained embeddings and advanced regularization.

### Final Hyperparameter Optimization: 
Perform a focused, final search for the best hyperparameters.

### Comprehensive Error Analysis: 
Qualitatively assess the best model's mistakes to understand its weaknesses.

### Final Model Training: 
Train the single best model on the full dataset with the optimized configuration.

### Final Report Generation: 
Compile all results and visualizations into a summary report.

This notebook provides a comprehensive walkthrough of each phase, with detailed explanations, code implementations, and analysis results. Each section builds upon the previous ones, creating a complete narrative of our model development and optimization journey.
