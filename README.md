# ETHOS: Enhanced Transformer for Health Outcome Simulation

ETHOS is a novel application of the transformer deep-learning architecture, originally
conceptualized for natural language processing2. This architecture, a cornerstone in large language
model (LLM) development, is repurposed in ETHOS to analyze health-related data, moving beyond the
textual focus of traditional LLMs. ETHOS is designed to process Patient Health Timelines (PHTs) -
detailed tokenized chronological records of health-related events - to predict future health
timelines.

![project_scheme](./figures/mimic_figure1_paper.png)

### Installation

Ensure your development environment is set up by following these steps:

#### Setting up the Environment

Python 3.10 or higher is required. 

[Optional] We strongly encourage to use a virtual environment for example Conda.
To create a new conda env:

```bash
conda create --name ethos python=3.10
conda activate ethos
```

Fetch the project and set it up in the development mode (`-e`) and install all necessary dependencies for
running notebooks and scripts by executing:
```bash
git clone https://github.com/ipolharvard/ethos-paper
cd ethos-paper
pip install -e .[all]
```

### Data Preparation

Prepare your dataset for analysis with these steps:

1. **Downloading the dataset:** Obtain the raw MIMIC-IV data
   from [PhysioNet](https://physionet.org/content/mimiciv/2.2/).
2. **Splitting the data:** Use the `scripts/data_train_test_split.py` script to divide the dataset
   into training and testing sets (this step requires 128GB RAM).
   <br>
   An example assuming the script is executed from the project root:
   ```bash
    python scripts/data_train_test_split.py ethos/data/mimic-iv-2.2
   ```
   The script generates a dataset folder with a `_Data` suffix. Ensure the processed data is
   named `mimic-iv-2.2_Data` or update the folder name in `ethos/tokenize/constants.py` accordingly.
3. **Organizing Dataset Directories:** Place the datasets in the project's data directory as
   follows:
   ```
   PROJECT_ROOT
   ├── pyproject.toml
   ├── README.md
   ├── scripts
   └── ethos
       └── data 
           ├── icd10cm-order-Jan-2021.csv.gz
           ├── drug_coding.py
           ├── mimic-iv-2.2              <== raw data from physionet.org
           ├── mimic-iv-2.2_Data         <== processed data (train/test split)
           └── mimic-iv-2.2_Data_parquet <== optional (for faster loading `scripts/convert_csv_to_parquet.py`)
   ```

### Usage


We provide the following resources for reproducing the results from our paper (requires Google Account):
- the pre-trained model with vocabulary at [Google Drive (≈0.5GB)](https://drive.google.com/file/d/1c8_OQadiHe0ZOoOdZuF-m0N3fbRnE1EP/view?usp=sharing)
- the results of inference at [Google Drive (≈0.8GB)](https://drive.google.com/file/d/1BgywarK7osx8xcyzZamOgSBhMJZqKlPy/view?usp=sharing)

Execute the project pipeline as follows:

1. **Dataset tokenization:** Assuming that the MIMIC dataset is prepared, convert the dataset from
   the tabular format to the tokenized format, starting with the training set, followed by the testing
   set using the training set's vocabulary. In case, you want to use the pre-trained model provided
   by us, you have to pass the vocabulary also when tokenizing the training dataset 
   (`-j` option specifies the number of processes for the separator injection phase):
   ```bash
   ethos tokenize- mimic train -j 50
   ```
   Followed by:
   ```bash
   ethos tokenize- mimic test -v <path_to_vocab> -j 50
   ```
   Refer to `ethos tokenize- --help` and `scripts/run_tokenization.sh` for more details.
2. **Model training:** Train the model on the tokenized dataset. Monitor the training process with
   wandb (the wandb API key is needed). An example configuration for running on 8 GPUs
   (adjust `--nproc_per_node` and `--gradient_accumulation_steps` to match your setup):

   ```bash
   torchrun --no_python --standalone --nproc_per_node=8 ethos train \
     --data_train <data_path> \
     --val_frac 0.04 \
     --vocab <vocab-path> \
     --batch_size 32 \
     --block_size 2048 \
     --n_layer 6 \
     --n_head 12 \
     --n_embd 768 \
     --dropout 0.3 \
     --lr $LR \
     --min_lr $MIN_LR \
     --log_interval 5 \
     --eval_interval 1000 \
     --gradient_accumulation_steps 8 \
     --max_iters 1000000 \
     --lr_decay_iters 50000 \
     --eval_iters 50 \
     --ctx_no_grad \
     --out_dir "out/mimic_layer_6_batch_32_do_0.3"
   ```
   Refer to `ethos train --help` and `scripts/run_inference.sh` for more details.
3. **Model evaluation:** Use the trained model for predicting future health timelines based on
   scenarios found in ethos/datasets, like ICU readmission or hospital mortality. 
   ```bash
   ethos infer \
     --test readmission \
     --data <data-path in ethos/data> \
     --vocab <vocab-path in ethos/data> \
     --model <model-path> \
     --out_dir <output-path>
   ```
   Refer to `ethos infer --help` and `scripts/run_inference.sh` for more details.
4. **Generating results:** Use the Jupyter notebooks to generate all the results and figures:
    - `notebooks/mimic_paper_results_agg.ipynb`
    - `notebooks/embedding_analysis.ipynb`