\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{hyperref}
\geometry{margin=1in}

\title{nanoGPT: Transformer-based Language Model}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Overview}
\nanoGPT{} is a minimalistic implementation of a GPT-style transformer model, designed for efficient training and inference on large-scale token datasets. It is inspired by the original Transformer architecture introduced in \textit{Attention is All You Need} (Vaswani et al., 2017), and fine-tuned for performance, compactness, and extensibility.

\section*{Transformer Block}

Each transformer block in the model can be described as:

\[
\text{TransformerBlock}(x) = x + \text{Dropout}\left( \text{FFN}\left( \text{LayerNorm}\left( x + \text{MultiHeadSelfAttention}(\text{LayerNorm}(x)) \right) \right) \right)
\]

\noindent where:
\begin{itemize}
  \item \textbf{LayerNorm} stabilizes training.
  \item \textbf{MultiHeadSelfAttention} enables the model to attend to different parts of the input sequence.
  \item \textbf{FFN} is a feed-forward network applied to each position separately.
  \item \textbf{Dropout} prevents overfitting.
\end{itemize}

\section*{Model Architecture}

The model input and output flow is defined as:

\[
x_0 = \text{TokenEmbedding}(input)
\]
\[
x_{i+1} = \text{TransformerBlock}(x_i), \quad \text{for } i = 0, \dots, N-1
\]
\[
\hat{y} = \text{Softmax}(\text{Linear}(x_N))
\]

\section*{Loss Function}

The training objective is to minimize the negative log-likelihood of the next token (causal language modeling):

\[
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{<t})
\]

\section*{Optimization}

Training uses the AdamW optimizer with the following enhancements:
\begin{itemize}
  \item \textbf{Cosine Learning Rate Decay}:
    \[
    \eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}} \pi\right)\right)
    \]
  \item \textbf{FlashAttention} for efficient memory usage and faster attention computation.
  \item \textbf{DDP (Distributed Data Parallel)} to enable scalable training across multiple GPUs.
\end{itemize}

\end{document}
