# Implementation of Fast Temporal Sentence Grounding

## About
This project implements an effective and efficient framework for the fast temporal sentence grounding (TSG) task. TSG aims to localize a target segment in an untrimmed video semantically according to a given sentence query. Conventional TSG methods follow the top-down or bottom-up strategy with a time-consuming framework, which is inefficient and inflexible for a large number of untrimmed videos in real-world applications. This project presents an end-to-end framework that models hours-long videos in a single network execution. Specially, the framework is structured in a coarse-to-fine manner, where context knowledge is extracted from non-overlapping video clips (anchors), followed by the supplementation of highly responsive anchors to the query for detailed content knowledge. Therefore, the introduced approach enhances efficiency and enables the capture of long-range temporal correlations in overlong videos for more precise video grounding.

## 🚀 Preparation

### 1. Install dependencies
The code requires python and we recommend you to create a new environment using conda.

```bash
conda create -n soonet python=3.8
```

Then install the dependencies with pip.

```bash
conda activate soonet
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### 2. Download data
- You should request access to the MAD dataset from [official webpage](https://github.com/Soldelli/MAD). Noded that all our experiments are implemented on MAD-v1.
- Upon completion of the download, extract the zip file contents and allocate the data to the "data/mad" directory.

### 3. Data preprocess

Use the following commands to convert the annotation format and extract the sentence features.

```bash
python preprocess/proc_mad_anno.py
python preprocess/encode_text_by_clip.py
```

The final data folder structure should looks like
```
data
└───mad/
│    └───annotations/
│        └───MAD_train.json
│        └───MAD_val.json
│        └───MAD_test.json
│        └───train.txt
│        └───val.txt
│        └───test.txt
│    └───features/  
│        └───CLIP_frame_features_5fps.h5
│        └───CLIP_language_features_MAD_test.h5
│        └───CLIP_language_sentence_features.h5
│        └───CLIP_language_tokens_features.h5
```

## 🔥 Experiments

### Training

Run the following commands for training model on MAD dataset:

```bash
python -m src.main --exp_path /path/to/output --config_name soonet_mad --device_id 0 --mode train
```

Please be advised that utilizing a batch size of 32 will consume approximately 70G of GPU memory. 
Decreasing the batch size can prevent out-of-memory, but it may also have a detrimental impact on accuracy.

### Inference

Once training is finished, you can use the following commands to inference on the test set of MAD.

```bash
python -m src.main --exp_path /path/to/training/output --config_name soonet_mad --device_id 0 --mode test
```

