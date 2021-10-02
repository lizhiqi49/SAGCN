# SAGCN

---

## Overview



---

## Software requirements

* Python 3.8.8
* PyTorch 1.5.0
* torch_geometric

## Installation guide
* When installing pytorch, the versions of pytorch, torchvision, cuda and cudnn should be matched. Here is our installation command as a reference.
  ```
  conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
  ```
* torch_geometric is a powerful API of many kinds of Graph Neural Network based on Pytorch. There are 4 dependency packages to be installed before installing torch_geometric, and those version should be matched with the versions of cuda and pytorch.
  
  ```
  $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
  $ python setup.py install or pip install torch-geometric
  ```

---

## Repo content explanation

### TCGA_dataset_project

#### TCGA_clinical_data folder

* contains the clinical information of dataset from TCGA, in *biospecimen*, *clinical* and *finish* folders; 
* two script, `merge_file.py` and `statistic.py` to summarize the information;
* the summary of clinical information: `clinical_info_statistic.csv`;

#### code

* *preprocessing* folder contains the scripts to preprocess the dataset.
* *sagcn* folder:
  * `createDataset.py`: create methylation-interaction graphs as input of SAGCN model
  * `sagcn_kfold.py`: train SAGCN model based on the created graphs and extract DEMs from every input graph;
  * `selectProbes.py`: select the final DEMs from SAGCN's output
* *sample_classification* folder contains the scripts to perform clinical sample classification based on the DEMs from SAGCN
* *performance_evaluation* folder contains the scripts to perform performance evaluation on SAGCN model and clinical classification.

#### data

contains the intermediate files generated from pipeline.

#### prediction_results

contains the prediction results of SAGCN model and the follow clinical sample classification.

#### perf_evaluation_results

contains the evaluation results based on the prediction results.



### GEO_dataset _project

###



### compared_methods

#### classic_methods

###

#### state_of_the_art

###

### case_study

###





