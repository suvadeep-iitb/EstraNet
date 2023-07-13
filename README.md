# EstraNet: An Efficient Shift-Invariant Transformer Network for Side Channel Analysis

This repository contains the implementation of EstraNet, an efficient shift invariant transformer network for Side Channel Analysis. EstraNet has linear time 
and memory complexity. Therefore, it is scalable for power traces with lengths greater 10000.

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
on dataset \<dataset\> where \<dataset\> is one of ASCADf, ASCADr and CHES20.
