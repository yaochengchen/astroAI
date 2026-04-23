這是一份為你準備的英文版 `how_to_run.md` 文件。你可以通過下方的鏈接下載。

您的 Markdown 文件已準備好。
[file-tag: how_to_run.md]

```python
markdown_content = """# How to Run the AstroAI Pipeline

Follow these steps to set up the environment, process the data, and run the inference.

## 1. Downloads
Download the required code, dataset, and models from Zenodo:

```bash
# Code
wget "[https://zenodo.org/records/18612199/files/ambra-dipiano/astroAI-v1.2.2.zip?download=1](https://zenodo.org/records/18612199/files/ambra-dipiano/astroAI-v1.2.2.zip?download=1)"

# Dataset
wget "[https://zenodo.org/records/11086320/files/simulations_dataset_v1.zip?download=1](https://zenodo.org/records/11086320/files/simulations_dataset_v1.zip?download=1)"

# Models
wget "[https://zenodo.org/records/11086320/files/crta_models_v1.zip?download=1](https://zenodo.org/records/11086320/files/crta_models_v1.zip?download=1)"
```

## 2. Installation & Data Preparation
Configure the environment and install dependencies:

```bash
python installprod5.py
```

## 3. Data Processing
Execute the following steps sequentially to process your data:

```bash
# Generate noisy/clean data pairs
python make_noisy_clean_from_dl3.py -f conf_pairs.yml -o /home/cyc/software/astroAI/dataset/simulations_dataset_v1/dl3_z20_irf/output_pairs

# Preprocess the dataset
python -m astroai.tools.preprocess_ds -f conf_clean.yml
```

## 4. Inference
Once processing is complete, proceed to the inference stage:

* Open and execute: `CNN_inference.ipynb`
"""

with open("how_to_run.md", "w") as f:
    f.write(markdown_content)
```