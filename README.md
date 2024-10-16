# EstraNet: An Efficient Shift-Invariant Transformer Network for Side Channel Analysis

This repository contains the implementation of EstraNet, an efficient shift invariant transformer network for Side Channel Analysis ([paper](https://tches.iacr.org/index.php/TCHES/article/view/11255)).

The implementation is composed of the following files:
* **fast_attention.py:** It contains the code of the proposed GaussiP attention layer.
* **normalization.py:** It contains the code of layer-centering layer.
* **transformer.py:** It contains the code of the EstraNet model.
* **train_trans.py** It contains the code for training and evaluating the EstraNet model.
* **data_utils.py:** It contains the code for reading data from the ASCADf or ASCADr dataset.
* **data_utils_ches20.py:** It contains the code for reading data from the CHES20 dataset.
* **evaluation_utils.py:** It contains the code for computing the guessing entropy for the ASCAD datasets.
* **evaluation_utils_ches20.py:** It contains the code for computing the guessing entropy for the CHES20 dataset.
* **run_trans_\<dataset\>.sh:** It is the bash script with proper hyper-parameter setting to perform experiments 
on dataset \<dataset\> where \<dataset\> is one of ASCADf ([ASCAD fixed key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key)), ASCADr ([ASCAD random key](https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key)) and CHES20 ([CHES CTF 2020](https://ctf.spook.dev/)).


## Data Pre-processing:
* The traces of the CHES CTF 2020 dataset have been multiplied by the constant 0.004 to keep the range of the feature values within [-120, 120].

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
