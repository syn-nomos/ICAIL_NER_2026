# Revisiting Greek Legal NER: Building an Improved Dataset

**Submitted to ICAIL 2026**

This repository contains the source code, datasets, and experimental results for the paper **"Revisiting Greek Legal NER: Building an Improved Dataset"**.

## Abstract

Named Entity Recognition (NER) constitutes a critical component of Legal Information Retrieval and knowledge graph construction. In this paper, we revisit and significantly enhance the widely-used Greek Legal NER dataset to address its critical limitations, such as severe annotation sparsity and systemic structural artifacts.

To overcome these challenges, we propose a refined annotation schema and implement a semi-automatic annotation pipeline that integrates a **Multi-Agent System** with a **Human-in-the-Loop (HITL)** verification protocol. This methodology enabled the generation of a high-density Silver Standard training set and a manually curated Gold Standard test set. The resulting dataset achieves a **2.7x increase** in total annotations while restoring the semantic integrity of fragmented entities.

## 📊 Dataset Statistics (v2)

Overview of the entity distribution across Train, Development, and Test splits in the Greek Legal NER v2.

| Metric / Type | Train Set | Dev Set | Test Set | Avg Length |
| :--- | :---: | :---: | :---: | :---: |
| **Total Sentences** | **17,679** | **4,909** | **3,879** | - |
| `ORG` | 7,948 | 1,155 | 1,508 | 3.6 |
| `LEG-REFS` | 3,982 | 1,213 | 1,073 | **10.8** |
| `PUBLIC-DOCS` | 2,958 | 817 | 759 | **9.5** |
| `GPE` | 4,168 | 1,382 | 583 | 2.1 |
| `LOCATION` | 4,956 | 113 | 567 | 1.4 |
| `PERSON` | 2,018 | 295 | 402 | 2.3 |
| `DATE` | 2,598 | 515 | 463 | 4.0 |
| `FACILITY` | 401 | 30 | 82 | 3.6 |
| **Total Entities** | **29,029** | **5,520** | **5,437** | **5.95** |

---

## 🛠️ Data Preprocessing & Tokenization

To ensure consistency across all dataset splits (Train, Dev, Test) and maximize compatibility with modern NLP pipelines, we standardized the tokenization process using **spaCy**.

- **Tokenizer**: `spacy` (Library v3.x)
- **Model**: `el_core_news_sm` (Greek Small Model)
- **Methodology**: All splits were re-tokenized to strictly follow the linguistic rules of the spaCy Greek model.

## ⚙️ Experimental Setup

To evaluate the quality of the dataset, we utilized the **[LEXTREME](https://github.com/JoelNiklaus/LEXTREME)** benchmark framework. We evaluated five diverse Transformer architectures, ranging from general-purpose multilingual models to domain-specific Greek encoders:

1. **Multilingual MiniLM**: A lightweight, distilled version of XLM-R designed for efficiency and speed.
2. **DistilBERT (Multilingual)**: A compact, distilled version of mBERT.
3. **XLM-R Base**: A strong cross-lingual baseline model pre-trained on 100 languages.
4. **Greek BERT**: A monolingual BERT model pre-trained specifically on general domain Greek corpora.
5. **Greek Legal RoBERTa**: A domain-adapted version of XLM-R, further pre-trained on Greek legal documents.

**Hyperparameters:** All models were fine-tuned for **15 epochs** with a batch size of 16 and a learning rate of 2e-5.

## 🚀 Model Performance

### 🏆 Overall Comparison (Micro-F1 & Macro-F1)

| Model Architecture | Micro-F1 (%) | Macro-F1 (%) |
| :--- | :---: | :---: |
| Multilingual MiniLM | 59.66 | 54.68 |
| DistilBERT (Multilingual) | 57.32 | 53.38 |
| **XLM-R-Base (Multilingual)** | **61.58** | **58.91** |
| Greek BERT (Uncased) | 60.71 | 57.73 |
| Greek Legal RoBERTa | 60.36 | 58.02 |

### 🔍 Detailed Performance (XLM-R-Base)

Breakdown of the best performing model (XLM-R-Base) per entity type.

| Entity Type | F1 Score (%) | Support |
| :--- | :---: | :---: |
| `ORG` | 54.55 | 1,508 |
| `LEG-REFS` | **69.40** | 1,073 |
| `PUBLIC-DOCS` | 32.29 | 759 |
| `GPE` | 49.88 | 583 |
| `LOCATION` | 68.12 | 567 |
| `DATE` | **76.51** | 463 |
| `PERSON` | **80.13** | 402 |
| `FACILITY` | 24.21 | 82 |
| **Total (Micro)** | **61.58** | **5,335** |

---

## 📂 Repository Structure

- `data/`: Contains the NER datasets in CoNLL format.
    - `conll/`: Train, Dev, and Test splits.
    - `statistics/`: Detailed reports on dataset distribution.
- `src/`: Source code of the system.
    - `phase1_hybrid/`: The initial hybrid prediction system (RoBERTa + RegEx + Lexicons).
    - `phase2_council/`: The specific "Council of Agents" refinement mechanism.
- `app/`: The Human-in-the-Loop Annotation Interface (Streamlit).

## 🛠️ Setup & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Annotation Interface:**
   ```bash
   streamlit run app/app.py
   ```

3. **Reproducibility:**
   - Train Phase 1 Model:
     ```bash
     python -m src.phase1_hybrid.train_ner --input_file data/conll/train_v2.conll
     ```
   - Run Multi-Agent Refinement:
     ```bash
     python -m src.phase2_council.multi_agent_refinement
     ```
