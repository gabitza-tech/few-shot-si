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

## Extracted embeddings and pretrained models

The pretrained models were trained using the [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN) repo and the already extracted features from the audio splits used in the paper can be found [here](https://drive.google.com/drive/folders/1LnUDekondromtCUSC_jFzLV8LB8DYXIT?usp=sharing).

## 📦 Installation

Conda installation:

```bash
# clone repo
git clone https://github.com/gabitza-tech/few-shot-si.git
cd few-shot-speaker-identification

# create and activate environment
conda env create -f environment.yml
```
## 🧪 Running Experiments

You can run few-shot evaluations using our scripts in src/.

▶️ Single Experiment (Default Setting)

```
export PYTHONPATH=$(pwd)
python3 src/few_shot.py <embeddings_path> <out_dir> False <k_shots> <seed>
```

- <embeddings_path> — path to the precomputed embeddings (.npy or similar)

- <out_dir> — directory where the results will be saved

- <use_mean>=False — whether to run transductive iterations (default: False)

- <k_shots> — number of support shots per speaker (e.g., 1, 3, or 5)

- <\seed> — random seed for reproducibility

🔁 Multiple Experiments

To automate multiple runs with different parameters (e.g., multiple seeds or shot configurations):

```
export PYTHONPATH=$(pwd)
python3 src/run_few_shot.py
```

## 🧭 Acknowledgements

- Backbone training adapted from [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)

## 🧾 Citation

```
@inproceedings{pirlogeanu2025fewshot,
  title={Closed-Set Speaker Identification using Few-Shot Transductive Learning},
  author={Gabriel Pîrlogeanu and Ana Neacșu and Horia Cucu and Jean-Christophe Pesquet and Ismail Ben Ayed},
  booktitle={Proc. European Signal Processing Conference (EUSIPCO)},
  year={2025}
}
```

## 📬 Contact

For questions, collaborations, or clarifications:

Gabriel Pîrlogeanu — `gabriel.pirlogeanu@upb.ro`
