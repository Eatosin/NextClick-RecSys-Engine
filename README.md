# NextClick: Real-Time Session-Based Recommendation Engine 🎬

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/AI-PyTorch-orange)
![RecSys](https://img.shields.io/badge/Domain-Recommendation%20Systems-purple)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/eatosin/NextClick-RecSys-Engine/blob/main/NextClick_Engine.ipynb)

## ⚡ The Problem
Traditional Recommender Systems (Matrix Factorization) are static. They ignore the **User's Current Context**.
*   If a user watches 3 Action movies in a row, they want *Action* right now, even if they watched *Romance* last month.
*   We need a model that understands **Sequential Dependency**.

## 🧠 The Solution: SASRec (Self-Attentive Sequential Recommendation)
**NextClick** utilizes a Transformer architecture (Multi-Head Attention) to model user sessions as a sequence.
*   **Architecture:** 2-Layer Transformer Encoder.
*   **Task:** Next-Item Prediction (Cloze Task).
*   **Data:** MovieLens 1M (Sequentialized).

## ⚙️ How it works
1.  **Ingest:** User clicks `[Item A, Item B, Item C]`.
2.  **Embed:** Items are converted to Vectors + Positional Encodings.
3.  **Attend:** The model uses Self-Attention to weigh recent items higher than older items.
4.  **Predict:** Output is a ranked list of Top-K items for the next step.

## 📊 Performance
*   **Training Speed:** ~10,000 sequences/sec on T4 GPU.
*   **Inference Latency:** < 50ms (PyTorch CPU).

## 👨‍💻 Author
**Owadokun Tosin Tobi**
*AI Engineer specializing in Deep Learning & Systems*
