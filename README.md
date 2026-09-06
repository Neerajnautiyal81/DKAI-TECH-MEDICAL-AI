# DKAI-TECH-MEDICAL-AI

A practical medical AI project developed for the DKAI Tech take-home assignment.

This repository contains three independent tasks covering:

1. Vision-Language Model (VLM) integration for preliminary clinical image analysis
2. Lightweight Retrieval-Augmented Generation (RAG) for medical question answering
3. Clinical text preprocessing and instruction-tuning dataset preparation

Task 1 and Task 2 are mandatory. Task 3 is an optional bonus task.

---

# Project Structure

DKAI-TECH-MEDICAL-AI/

├── README.md

├── Task-1/
│   ├── clinical_vlm_colab.ipynb
│   └── outputs/
│       ├── sample_outputs.json
│       └── sample_outputs.md

├── Task-2/
│   ├── medical_rag_colab.ipynb
│   └── medical_corpus/
│       ├── 01_hypertension.md
│       ├── 02_type2_diabetes.md
│       ├── 03_asthma.md
│       ├── 04_copd.md
│       ├── 05_pneumonia.md
│       ├── 06_stroke.md
│       ├── 07_acs_mi.md
│       ├── 08_sepsis.md
│       ├── 09_anaphylaxis.md
│       ├── 10_malaria.md
│       ├── 11_tuberculosis.md
│       ├── 12_ckd.md
│       ├── 13_anaemia.md
│       ├── 14_uti.md
│       ├── 15_hypothyroidism.md
│       └── 16_amr_stewardship.md

└── Task-3/
    ├── clinical_dataset_pipeline.ipynb
    ├── data/
    │   └── raw_clinical_data.csv
    └── outputs/
        ├── cleaned_clinical_dataset.csv
        └── clinical_instruction_dataset.jsonl

---

# Task 1 — Vision-Language Model Integration

## Objective

The first task implements an open-source Vision-Language Model capable of running under limited compute resources such as the free Google Colab environment.

The system is designed as a preliminary clinical imaging assistant. It accepts a medical image and a user question, then produces a cautious, structured interpretation.

## Model Selection

Model used:

**HuggingFaceTB/SmolVLM-Instruct**

SmolVLM-Instruct was selected because it is a relatively lightweight open-source VLM suitable for experimentation in a resource-constrained Google Colab environment.

The model uses an image encoder together with a language model to process both visual and textual information.

The notebook uses the Hugging Face Transformers ecosystem for model loading and inference.

## Memory Optimization

The notebook includes memory-conscious model loading strategies, including:

- Reduced-precision inference using BF16 or FP16 depending on available hardware
- `device_map="auto"` for automatic device placement
- `low_cpu_mem_usage=True`
- Optional 4-bit NF4 quantization using BitsAndBytes

The 4-bit quantization path is optional and can be enabled depending on the available Colab environment.

## Reusable Inference Function

A reusable inference function is implemented so that the model can be called with:

- A PIL image
- A local image path
- A publicly accessible image URL
- A user-provided medical question

The function applies the clinical system prompt, performs model inference, decodes the generated response, and returns the answer together with relevant inference metadata.

## Clinical System Prompt

The system prompt instructs the VLM to behave as a preliminary clinical imaging assistant.

The main principles are:

- Describe only visible findings supported by the image
- Avoid inventing patient demographics or medical history
- Clearly communicate uncertainty
- Avoid presenting image interpretation as a definitive diagnosis
- Refuse or qualify responses for unreadable or non-medical images
- Distinguish observations from possible interpretations
- Encourage professional clinical review when appropriate
- Provide a structured response

## Medical Images

Inference was demonstrated using publicly available medical images, including chest radiographs.

The notebook uses public image URLs rather than requiring large image files to be stored inside the repository.

Sample outputs are stored in:

`Task-1/outputs/sample_outputs.json`

and

`Task-1/outputs/sample_outputs.md`

## Task 1 Design

The overall flow is:

Medical Image + User Question

→ Image Preprocessing

→ SmolVLM-Instruct

→ Clinical System Prompt

→ Generated Interpretation

→ Structured Output

The design intentionally keeps the inference interface reusable so that additional medical images can be evaluated without changing the core pipeline.

## Task 1 Limitations

This system is a demonstration and not a clinical diagnostic system.

Important limitations include:

- VLM outputs can contain hallucinations
- Image quality can strongly affect the response
- The model may miss subtle abnormalities
- The model may produce clinically plausible but incorrect interpretations
- The system does not have access to patient history, laboratory results, prior imaging, or other clinical context
- Public images may not represent real-world clinical diversity
- Model performance can vary depending on hardware and precision settings
- Generated text should not be treated as a medical diagnosis

The output should therefore be considered preliminary and educational rather than a substitute for a qualified healthcare professional.

---

# Task 2 — Lightweight Medical RAG System

## Objective

The second task implements a standalone Retrieval-Augmented Generation system for answering text-based medical questions using a small medical knowledge corpus.

The system retrieves relevant information from the medical corpus before generating an answer.

## Architecture

The pipeline is:

Medical Knowledge Documents

→ Text Cleaning

→ Paragraph-Based Chunking

→ BAAI/bge-small-en-v1.5 Embeddings

→ FAISS Vector Index

→ Top-3 Retrieval

→ Retrieved Context

→ Qwen2.5-1.5B-Instruct

→ Grounded Answer

The retrieval and generation components are separated so that the knowledge base can be updated independently of the language model.

## Medical Knowledge Corpus

The corpus contains 16 medical reference documents covering topics such as:

- Hypertension
- Type 2 diabetes
- Asthma
- COPD
- Pneumonia
- Stroke
- Acute coronary syndrome / myocardial infarction
- Sepsis
- Anaphylaxis
- Malaria
- Tuberculosis
- Chronic kidney disease
- Anaemia
- Urinary tract infection
- Hypothyroidism
- Antimicrobial resistance and stewardship

The corpus is stored in:

`Task-2/medical_corpus/`

## Embedding Model

Embedding model:

**BAAI/bge-small-en-v1.5**

This is an open-source sentence embedding model selected to provide a lightweight embedding pipeline suitable for CPU or limited-resource environments.

A retrieval-specific query prefix is used when generating query embeddings.

## Vector Database

FAISS is used for vector storage and similarity search.

The implementation uses:

**FAISS IndexFlatIP**

This provides exact inner-product similarity search.

The retrieved results are ranked by vector similarity.

## Chunking Strategy

The medical reference documents are cleaned and divided into paragraph-based chunks.

The configuration used in the notebook includes:

- Maximum chunk size of approximately 180 words
- One paragraph of overlap between adjacent chunks
- Top 3 retrieved chunks for each question

The resulting corpus contains 55 chunks.

The chunk statistics were:

- Minimum chunk size: approximately 101 words
- Mean chunk size: approximately 148 words
- Maximum chunk size: approximately 204 words

## Retrieval Strategy

For every user question:

1. The question is converted into an embedding.
2. FAISS searches the medical corpus.
3. The three most relevant chunks are retrieved.
4. The retrieved chunks are inserted into the generation prompt.
5. The language model generates an answer using the retrieved context.

The retrieval pipeline is intentionally simple and lightweight.

## Generation Model

Instruction model:

**Qwen/Qwen2.5-1.5B-Instruct**

The model was selected because it is lightweight enough for experimentation while providing instruction-following capability.

The generation prompt instructs the model to answer using only the retrieved medical context.

This reduces the chance of the language model introducing unsupported information from its general knowledge.

## Evaluation

The notebook includes a small retrieval evaluation over seven sample questions.

Evaluation results:

- Hit@3: **1.00**
- MRR@3: **1.00**
- All seven questions returned exactly three retrieved documents.

These results indicate that the small evaluation set successfully retrieved the expected relevant documents.

This is a limited evaluation and should not be interpreted as a comprehensive benchmark of medical retrieval quality.

## Sample Questions

The notebook demonstrates the system using multiple medical questions covering topics represented in the corpus.

The evaluation includes both in-corpus medical questions and an out-of-scope question to examine the behavior of the retrieval system when the required information is not directly represented in the knowledge base.

## Task 2 Design Decisions

The system prioritizes simplicity and reproducibility.

Design choices include:

- A small trusted medical corpus instead of a large general web crawl
- Open-source embeddings
- FAISS for lightweight vector search
- Paragraph-based chunking for readable retrieved context
- Top-3 retrieval to keep the context small
- A lightweight instruction model for answer generation
- Explicit grounding instructions in the generation prompt

## Task 2 Limitations

Important limitations include:

- The corpus is small and cannot cover the full medical domain
- Retrieval quality depends on corpus coverage and document quality
- Similarity search can retrieve text that is semantically related but not clinically sufficient
- The system does not perform expert-level medical verification
- The language model can still generate incorrect statements
- The evaluation set is small
- Retrieval metrics were evaluated on a limited set of questions
- The system does not replace clinical guidelines, professional judgment, or medical literature review
- The knowledge corpus is static unless the documents are updated and the index is rebuilt

---

# Task 3 — Clinical Dataset Preparation Pipeline

## Objective

Task 3 is the optional bonus task.

The pipeline prepares noisy clinical text for potential future instruction-tuning use.

The workflow starts with a selected subset of public MTSamples clinical text and introduces representative data-quality issues before cleaning and filtering the records.

## Source Data

The pipeline uses the public MTSamples dataset.

A subset of 250 clinical records is selected for the preprocessing demonstration.

The original selected records are stored in:

`Task-3/data/raw_clinical_data.csv`

## Noise Simulation

The notebook creates a noisy working dataset containing examples of common data-quality problems such as:

- HTML formatting artifacts
- OCR-like artifacts
- Messy whitespace
- Duplicate records
- Empty records
- Very short records
- Excessively long records

This allows the preprocessing pipeline to demonstrate how these issues can be detected and handled.

## Cleaning Pipeline

The preprocessing workflow includes:

1. Cleaning formatting artifacts
2. Removing duplicate records
3. Removing empty records
4. Filtering very short or uninformative records
5. Handling excessively long records
6. Normalizing whitespace
7. Resetting the dataframe index
8. Reporting before-and-after dataset statistics

## Output Files

The cleaned dataset is exported as:

`Task-3/outputs/cleaned_clinical_dataset.csv`

The instruction-tuning dataset is exported as:

`Task-3/outputs/clinical_instruction_dataset.jsonl`

The JSONL output follows a message-based instruction-tuning structure containing user and assistant messages.

## Dataset Statistics

The notebook reports dataset statistics before and after preprocessing.

These statistics are generated directly during notebook execution and are intended to make the effect of each cleaning step transparent.

## Task 3 Design Decisions

The preprocessing pipeline aims to improve data quality without attempting to rewrite clinical meaning.

The main principle is to remove formatting and quality problems while preserving useful clinical text.

The pipeline is intentionally lightweight and can be adapted to larger clinical text datasets.

## Task 3 Limitations

Important limitations include:

- The dataset is a relatively small demonstration subset
- Synthetic noise does not represent every type of real clinical data corruption
- Simple length-based filtering can remove records that may still contain useful information
- Automated cleaning may occasionally remove meaningful formatting
- The pipeline does not perform expert clinical annotation
- The resulting dataset should be reviewed before being used for instruction tuning
- Public datasets may contain biases or documentation patterns that do not generalize to all clinical environments

---

# How to Run

All three tasks are implemented as Jupyter/Google Colab notebooks.

Google Colab is recommended because the notebooks use open-source machine learning models and may benefit from GPU acceleration.

---

# Task 1 Setup

Open:

`Task-1/clinical_vlm_colab.ipynb`

Recommended environment:

- Google Colab
- Python 3
- GPU runtime recommended

The notebook installs the required Python packages and downloads the selected Hugging Face model.

Run the notebook cells from top to bottom.

The notebook performs:

1. Package installation
2. Hardware detection
3. Model configuration
4. Model loading
5. Clinical prompt configuration
6. Reusable inference function definition
7. Public medical image loading
8. Image inference
9. Sample output generation
10. Results and limitations review

The generated sample outputs are stored under:

`Task-1/outputs/`

---

# Task 2 Setup

Open:

`Task-2/medical_rag_colab.ipynb`

Recommended environment:

- Google Colab
- Python 3
- CPU is supported
- GPU is recommended for faster language-model generation

The notebook installs the required packages, including:

- sentence-transformers
- FAISS
- Transformers
- PyTorch

The main pipeline is executed from top to bottom.

The notebook:

1. Installs dependencies
2. Defines configuration
3. Loads the medical corpus
4. Cleans and chunks the documents
5. Creates embeddings
6. Builds the FAISS index
7. Loads Qwen2.5-1.5B-Instruct
8. Defines retrieval and generation functions
9. Runs sample questions
10. Evaluates retrieval performance
11. Displays retrieval and generation results

The medical corpus is already included in:

`Task-2/medical_corpus/`

---

# Task 3 Setup

Open:

`Task-3/clinical_dataset_pipeline.ipynb`

Recommended environment:

- Google Colab
- Python 3

The notebook downloads/loads the public MTSamples source data, selects the demonstration subset, creates noisy examples, performs preprocessing, reports statistics, and exports the cleaned datasets.

The raw input file is included in:

`Task-3/data/raw_clinical_data.csv`

The generated outputs are stored in:

`Task-3/outputs/`

---

# Reproducibility

The notebooks contain their required package installation and configuration steps.

Random seeds are used where applicable to improve reproducibility.

Task 1 uses public medical image URLs.

Task 2 includes the medical knowledge corpus directly in the repository.

Task 3 includes the selected raw clinical dataset subset used by the preprocessing demonstration.

Because machine-learning models can behave differently across hardware, library versions, and numerical precision settings, exact generated text may vary slightly between runs.

---

# Assumptions

The project makes the following assumptions:

- The notebooks are primarily intended to run in Google Colab.
- Internet access is available when downloading models or public resources.
- GPU availability is optional for some components but recommended for faster inference.
- The medical knowledge corpus is treated as the authoritative context for the RAG demonstration.
- Model-generated clinical information is considered preliminary and must not be treated as definitive medical advice.
- Publicly available data and images are used for demonstration and educational purposes.

---

# Overall Limitations and Safety

This repository demonstrates engineering approaches for medical AI systems.

It is not intended to provide medical diagnosis, treatment recommendations, or clinical decision-making.

The VLM, RAG system, and text-generation models can all produce incorrect or incomplete outputs.

Medical AI systems require careful validation, appropriate datasets, clinical expertise, monitoring, and safety controls before deployment in real clinical environments.

The examples in this repository should therefore be considered a technical demonstration rather than a production-ready clinical application.

---

# Summary

This project demonstrates three complementary medical AI workflows:

**Task 1 — Medical Vision-Language Model**

Uses SmolVLM-Instruct to process public medical images and generate cautious preliminary observations using a reusable inference interface.

**Task 2 — Medical RAG**

Uses BAAI/bge-small-en-v1.5 embeddings, FAISS retrieval, and Qwen2.5-1.5B-Instruct to answer medical questions using retrieved context from a small medical corpus.

**Task 3 — Clinical Dataset Preparation**

Uses a preprocessing pipeline to clean noisy clinical text, remove low-quality records, report dataset statistics, and generate instruction-tuning JSONL data.

Together, the tasks demonstrate practical experience with multimodal AI, retrieval-augmented generation, open-source language models, medical text processing, and resource-constrained machine-learning workflows.

---

# Repository

GitHub repository:

DKAI-TECH-MEDICAL-AI

The repository contains all completed mandatory tasks and the optional Task 3 bonus work in clearly separated folders.