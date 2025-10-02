# Knowledge Graph-Driven Medicine Recommendation System using Graph Neural Networks on Longitudinal Medical Records

### Folder Specification
- src/
  - dataloader.py: load the processed MIMIC-IV data for training
  - KGDNet_Model.py: the full model of KGDNet
  - KGDNet_Training.py: the full code for training KGDNet
  - util.py
  - COGNet_ablation.py: ablation models of COGNet
  - baselines/
    - data_processing.ipynb: data processing code for baseline models
    - models.py: full code for all baseline models
    - layers.py
    - LR.py
    - LEAP.py
    - RETAIN.py
    - GAMENet.py
    - SafeDrug.py
    - MICRON.py
    - COGNet.py
- data/ **(For a fair comparision, we use the same data and pre-processing scripts used in [Safedrug](https://github.com/ycq091044/SafeDrug))**
  - mapping files that collected from external sources
    - drug-atc.csv: drug to atc code mapping file
    - drug-DDI.csv: this a large file, could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
    - ndc2atc_level4.csv: NDC code to ATC-4 code mapping file
    - ndc2rxnorm_mapping.txt: NDC to xnorm mapping file
    - idx2drug.pkl: drug ID to drug SMILES string dict
  - Under MIMIC Dataset policy, we are not allowed to distribute the datasets. Practioners could go to https://physionet.org/content/mimiciii/1.4/ and requrest the access to MIMIC-III dataset and then run our processing script to get the complete preprocessed dataset file.
  - dataset processing scripts
    - DataProcessing.ipynb: is used to process the MIMIC original dataset.

### Step 1: Data Processing

- Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You need to be a credentialed user)

- go into the folder and unzip three main files (PROCEDURES_ICD.csv.gz, PRESCRIPTIONS.csv.gz, DIAGNOSES_ICD.csv.gz)

- download the DDI file and move it to the data folder
  download https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing

- change the path in DataProcessing.ipynb and processing the data to get the ehr_records.pkl

### Step 2: Package Dependency

- first, install the rdkit conda environment
```python
conda create -c conda-forge -n KGDNet
conda activate KGDNet
```

- then, in KGDNet environment, install the following packages
```python
pip install scikit-learn, dill, dnc
```
Now, install Pytorch and Pytorch Geometric
```python
pip install torch

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cu121.html
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-2.2.1+cu121.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

### Step 3: run the code

```python
python KGDNet_training.py
```

The following arguments can be provided:

    usage: KGDNet_training.py [-h] [--test] 
                              [--resume_path RESUME_PATH]
                              [--lr LR] [--target_ddi TARGET_DDI]
                              [--kp KP] [--dim DIM]
    
    optional arguments:
      -h, --help            Show this help message and exit
      --test                Test mode
      --resume_path RESUME_PATH
                            Resume path
      --lr LR               Set the learning rate
      --batch_size          Set the batch size 
      --embed_dim           Embedding dimension size
      --ddi_thresh          Set DDI threshold

Partial credit to previous reprostories:
- https://github.com/sjy1203/GAMENet
- https://github.com/ycq091044/SafeDrug
- https://github.com/ycq091044/MICRON
- https://github.com/BarryRun/COGNet
