# Task 1 — Vision-Language Model for Preliminary Clinical Image Analysis

## Overview

This task implements a lightweight preliminary clinical imaging assistant using the open-source **SmolVLM-Instruct** vision-language model.

The system accepts a medical image and a clinical question, then generates a text-based description of visible features and uncertainty.

This project is intended for educational and research purposes only. It is not a medical device and must not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

## Model Selection

### Selected Model

**SmolVLM-Instruct**

- Hugging Face model: `HuggingFaceTB/SmolVLM-Instruct`
- Approximately 2.25 billion parameters
- Architecture: Idefics3 with a SigLIP vision encoder and SmolLM2 language model
- Open-source and ungated
- Suitable for resource-constrained Google Colab environments

### Why SmolVLM-Instruct?

SmolVLM-Instruct was selected because it provides image-and-text understanding while remaining small enough to run on a free-tier Google Colab GPU.

The main selection criteria were:

- Open-source availability
- Multimodal image and text support
- Reasonable memory requirements
- Compatibility with Google Colab
- Support for instruction-based prompting
- Ability to provide natural-language descriptions of medical images

The relatively small model size makes it more practical for experimentation than larger vision-language models under limited GPU memory.

---

## Memory Optimization

The notebook uses several memory-conscious techniques:

### Reduced Precision

The model is loaded using `float16` or `bfloat16` when supported by the available hardware.

This substantially reduces memory consumption compared with full `float32` model weights.

### Automatic Device Placement

`device_map="auto"` is used through Accelerate so that model components can be placed automatically on available hardware.

### Low CPU Memory Usage

`low_cpu_mem_usage=True` is used during model loading to reduce unnecessary host-memory duplication while loading model weights.

### Optional 4-bit Quantization

An optional 4-bit NF4 quantization path using BitsAndBytes is included in the notebook.

It is disabled by default because SmolVLM-Instruct fits within the target Colab GPU environment without requiring quantization. Quantization can be enabled if GPU memory is more constrained.

---

## Clinical Imaging System Prompt

The notebook defines a dedicated system prompt for preliminary clinical image analysis.

The prompt instructs the model to:

1. Describe only features that are visually supported by the image.
2. Separate visible observations from interpretation.
3. Communicate uncertainty clearly.
4. Avoid giving a definitive diagnosis.
5. Avoid inventing patient demographics or clinical history.
6. Avoid unsupported medical claims.
7. State when an image is unreadable or unsuitable for interpretation.
8. Recommend professional clinical review when appropriate.
9. Avoid fabricating information that is not visible in the image.
10. Use a structured response format covering observations and uncertainty.

The system prompt is intended as a safety-oriented instruction layer. It does not guarantee that the model will always follow every instruction.

---

## Reusable Inference Function

The notebook implements a reusable function:

`analyze_medical_image(image, user_question)`

The function accepts:

- A PIL image
- A local image path
- An image URL
- A clinical question or instruction

It then:

1. Loads/prepares the image.
2. Applies the clinical system prompt.
3. Builds the multimodal input.
4. Runs the SmolVLM-Instruct model.
5. Decodes the generated response.
6. Returns the generated answer together with useful runtime metadata.

This allows the same inference pipeline to be reused across different medical images and questions.

---

## Public Medical Image Demonstrations

The notebook demonstrates the model on publicly available medical images.

### Image 1 — Normal Chest Radiograph

A publicly available PA chest radiograph is used as a normal example.

The question asks the model to comment on whether the cardiac silhouette appears within normal limits and to explain what can and cannot be assessed from the image.

### Image 2 — Lobar Pneumonia

A publicly available chest radiograph showing lobar pneumonia is used as an abnormal example.

Using the same imaging modality for the normal and abnormal examples provides a simple qualitative comparison of model behaviour.

### Image 3 — Barton's Fracture

An additional wrist radiograph showing a Barton's fracture is included as an optional third demonstration.

This tests whether the model can identify a different anatomical region and imaging context.

The medical images are sourced from Wikimedia Commons and are used as publicly available examples for educational demonstration.

---

## Sample Outputs

The generated model outputs are stored in the `outputs/` directory.

The output files are:

- `sample_outputs.json` — structured output, runtime metadata, model information, questions, and generated responses.
- `sample_outputs.md` — human-readable version of the sample outputs.

The outputs are preserved as generated by the model and may contain errors or unsupported statements.

---

## How to Run

### Google Colab

1. Open `clinical_vlm_colab.ipynb` in Google Colab.
2. Select a GPU runtime if available.
3. Run the notebook cells from top to bottom.
4. Install the required dependencies in the installation section.
5. Allow the model and processor to download from Hugging Face.
6. Check the detected GPU and data type.
7. Load the SmolVLM-Instruct model.
8. Run the clinical system-prompt section.
9. Run the reusable inference function.
10. Run the medical-image demonstration sections.
11. Generate and save the sample outputs.

A GPU is recommended for practical inference speed.

---

## Design Choices

The implementation prioritizes:

- A small open-source VLM suitable for Colab.
- Memory-efficient model loading.
- A reusable inference function.
- A dedicated clinical imaging system prompt.
- Publicly available medical image examples.
- Structured sample outputs.
- Explicit limitations and ethical considerations.

The system is designed as a demonstration of multimodal inference rather than a clinically validated diagnostic system.

---

## Limitations

### 1. General-Purpose Model

SmolVLM-Instruct is a general-purpose vision-language model and is not a clinically validated diagnostic model.

It was not developed as a replacement for radiologists or other qualified medical professionals.

### 2. Image Quality

Image resolution, exposure, positioning, projection, and other image characteristics can affect the generated response.

The notebook also applies image processing and resizing, meaning that subtle findings may not remain visible to the model.

### 3. Hallucination

The model may generate plausible-sounding findings that are not actually present in the image.

A fluent response should therefore not be interpreted as evidence that the response is correct.

### 4. Missed Findings

The model may fail to identify subtle abnormalities.

Failure to mention an abnormality should not be interpreted as evidence that the abnormality is absent.

### 5. False Positives and False Negatives

The model can produce both incorrect positive findings and incorrect negative findings.

Both can be harmful in a clinical setting.

### 6. Limited Demonstration Set

Only a small number of publicly available medical images are used in this notebook.

These examples are demonstrations of the pipeline and are not sufficient for measuring clinical accuracy or generalization.

### 7. No Clinical Validation

This project does not provide quantitative clinical validation.

A proper evaluation would require a representative labelled dataset, predefined evaluation metrics, expert ground truth, and comparison with appropriate clinical baselines.

### 8. System Prompt Limitations

The system prompt is an instruction to the model rather than a hard safety constraint.

The model may still fail to follow the requested response structure or may generate unsupported claims.

### 9. Educational Use Only

The generated outputs must not be used to make medical decisions.

Professional clinical review is required for real-world medical interpretation.

---

## Repository Structure

```text
Task-1/
├── clinical_vlm_colab.ipynb
├── README.md
└── outputs/
    ├── sample_outputs.json
    └── sample_outputs.md


## Assignment Coverage

| Assignment Requirement | Implementation |
|---|---|
| Explain model selection | SmolVLM-Instruct selection and justification |
| Memory optimization | Reduced precision, Accelerate, automatic device placement, low CPU memory usage, optional 4-bit path |
| Reusable inference function | `analyze_medical_image()` |
| Clinical imaging system prompt | Dedicated clinical imaging assistant prompt |
| At least two public medical images | Normal chest radiograph and lobar pneumonia radiograph |
| Sample outputs | `outputs/sample_outputs.json` and `outputs/sample_outputs.md` |
| Design choices and limitations | Documented in this README and notebook |


## Saving Outputs

After running the notebook, the generated sample results are saved in the `outputs/` directory:

- `outputs/sample_outputs.json`
- `outputs/sample_outputs.md`

When running the notebook in Google Colab, these files can be downloaded from the Colab Files panel and placed in the repository's `Task-1/outputs/` folder.

## Disclaimer

This project is for educational and research purposes only.

The model outputs are generated by a general-purpose vision-language model and may be incorrect, incomplete, or misleading. They must not be treated as medical diagnoses or used for clinical decision-making.