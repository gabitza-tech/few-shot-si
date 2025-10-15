# Closed-Set Speaker Identification using Few-Shot Transductive Learning

[![Conference](https://img.shields.io/badge/EUSIPCO-2025-blue)](https://www.eusipco.org/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./2025_eusipco_camera_ready.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation and experimental setup for the paper:

> **Closed-Set Speaker Identification using Few-Shot Transductive Learning**  
> *Gabriel Pîrlogeanu, Ana Neacșu, Horia Cucu, Jean-Christophe Pesquet, Ismail Ben Ayed*  
> Presented at **EUSIPCO 2025**

📄 [Read the paper](./2025_eusipco_camera_ready.pdf)

---

## 🧠 Overview

Closed-set unseen speaker identification plays a critical role in applications such as forensics, fraud detection, and speaker retrieval.  
We address this task through the **Few-Shot for A Single Class (FSAiC)** method — a **tuning-free transductive learning** approach designed for identifying an unseen speaker from a large watchlist using only a few short utterances.

- **Closed-set scenario**: all queries come from enrolled watchlist speakers unseen during training.  
- **Few-shot transductive setting**: single speaker per query set, with 1–5 utterances.
- **Scalable** to hundreds of classes.
- **Outperforms** state-of-the-art inductive and transductive baselines.

---

## 🧭 Method Overview

<!-- Place pipeline / diagram image here -->
<p align="center">
  <img src="eusipco_setup.drawio (3).svg" alt="FSAiC pipeline diagram" width="80%">
</p>

We propose **FSAiC** — Few-Shot for A Single Class — which leverages a maximum likelihood formulation tailored to the single-class query scenario.  
Unlike conventional few-shot learning setups, our method efficiently handles large support sets while remaining tuning-free.

---

## 📦 Installation

```bash
# clone repo
git clone https://github.com/<your-username>/few-shot-speaker-identification.git
cd few-shot-speaker-identification

# create and activate environment
conda create -n fewshot-si python=3.10
conda activate fewshot-si

# install dependencies
pip install -r requirements.txt

