# EstraNet: An Efficient Shift-Invariant Transformer Network for Side Channel Analysis

This repository contains the implementation of **EstraNet**, an efficient shift-invariant transformer network for Side-Channel Analysis.  
For more details, refer to the [paper](https://tches.iacr.org/index.php/TCHES/article/view/11255).

---
## Repository Structure
- **`fast_attention.py`** – Implements the proposed GaussiP attention layer.
- **`normalization.py`** – Implements the layer-centering normalization.
- **`transformer.py`** – Defines the EstraNet model architecture.
- **`train_trans.py`** – Training and evaluation script for EstraNet.
- **`data_utils.py`** – Utilities for loading ASCADf and ASCADr datasets.
- **`data_utils_ches20.py`** – Utilities for loading the CHES20 dataset.
- **`evaluation_utils.py`** – Computes guessing entropy for ASCAD datasets.
- **`evaluation_utils_ches20.py`** – Computes guessing entropy for CHES20 dataset.
- **`run_trans_\<dataset\>.sh`** – Bash scripts with predefined hyperparameters for specific datasets, where `<dataset>` is one of:
  - **ASCADf** ([fixed key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key))
  - **ASCADr** ([random key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key))
  - **CHES20** ([CHES CTF 2020](https://ctf.spook.dev/))

---

## Data Pre-processing:
- For the **CHES CTF 2020** dataset, the traces are multiplied by a constant `0.004` to keep the feature value range within **[-120, 120]**.

---

## Tested on
- Python 3.8.10  
- absl-py == 2.3.1 
- numpy == 1.24.3
- scipy == 1.10.1
- h5py == 3.11.0
- tensorflow == 2.13.0

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suvadeep-iitb/EstraNet.git
   cd EstraNet
   ```
2. **Install dependencies (Python >= 3.8 recommended):**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set dataset path in the bash script:**
   ```
   Open run_trans_\<dataset\>.sh and set the dataset path variable properly.
   ```
4. Train EstraNet:
   ```bash
   bash run_script_\<dataset\>.sh train
   ```
5. Perform Evaluation:
   ```bash
   bash run_script_\<dataset\>.sh test
   ```

----

## Citation:
```
@article{DBLP:journals/tches/HajraCM24,
  author       = {Suvadeep Hajra and
                  Siddhartha Chowdhury and
                  Debdeep Mukhopadhyay},
  title        = {EstraNet: An Efficient Shift-Invariant Transformer Network for Side-Channel
                  Analysis},
  journal      = {{IACR} Trans. Cryptogr. Hardw. Embed. Syst.},
  volume       = {2024},
  number       = {1},
  pages        = {336--374},
  year         = {2024},
  url          = {https://doi.org/10.46586/tches.v2024.i1.336-374},
  doi          = {10.46586/TCHES.V2024.I1.336-374},
  timestamp    = {Sat, 08 Jun 2024 13:14:59 +0200},
  biburl       = {https://dblp.org/rec/journals/tches/HajraCM24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
