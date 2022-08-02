# MyVoice: Machine Translation for American Sign Language

This repo contains the code for ASL training and evaluation, including inference on Edge device. Contributions and achievments from this code are available in [TODO Link to our paper](http://sign2text.github.io/myvoice) 

The repo includes code for:  
1. Data preparation and corrections for [How2Sign dataset](https://how2sign.github.io/) 
2. Training and evaluation using the [Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://github.com/neccam/slt)
3. Running inference on an edge device for one or more videos and creating subtitles.

This repo also includes [augmented How2Sign metadata dataset](./data/h2s/how2sign_realigned_gls.zip) to included glosses and curated sentences.  

## Requirements
* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

* The following packages are included in this distribution. Please read and understand the licensing from each before modifying or distributing the packages
*   Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation. [License terms](./src/slt/LICENSE). [Github repo](https://github.com/neccam/slt)  
*   Video Features [License terms](./src/video_features/LICENSE). [Github repo](https://github.com/v-iashin/video_features)


## Data Preparation

### How2Sign Dataset
[How2Sign Data preparation](./how2sign_dataprep.md) Please follow the detailed instructions provided here.

## Training & Evaluation
SLT code is included into this repo to ease training and enable inference. To run training, modify the config files as needed. Sample config files and the best performing config files are included in the repo.

Training includes evaluation against the test data. So a separate evaluation is not needed, unless you want to test with different beam sizes.

### Configs included
|Config                      |  Description  |
|----------------------------|---------------|
|h2s_gls_i3d_l20_lr01_e5.yaml|Sample config that runs training for 5 epochs for sentences less than 20 characters|  
|h2s_gls_i3d_all_cosine.yaml |Config file training on all data using Cosine LR scheduler|

### Execution
You can use the below command to submit the job in background
```sh
cd myvoice
nohup python3 src/slt/signjoey train configs/h2s_gls_i3d_l50_lr01.yaml > logs/h2s_gls_l20_lr01.log 2>&1 &
```

### Evaluation
Training performs evaluation as well. You may use the command below to run evaluation only runs do determine best beam sizes
```sh
cd myvoice
nohup python3 src/slt/signjoey test configs/h2s_gls_i3d_l50_lr01.yaml > logs/h2s_gls_l20_lr01_eval.log 2>&1 &
```

## Inference
If the best performance was obtained using `wlasl_gls.yaml` config, and you 
want to obtain translation for video in `data/infer_in/61916.mp4`, you can run the following

1. To run inference and get translated text on the same device where training was run, you can run

```sh
cd myvoice
python3 translate configs/wlasl_gls.yaml data/infer_in/61916.mp4
```
   To create a video file with translated text shown as captions and place file in data/infer_out folder, run
```sh
cd myvoice
python3 config configs/wlasl_gls.yaml data/infer_in/61916.mp4 data/infer_out
```
Omit the `infer_out` folder to get caption file as `data/infer_in/61916_caption.mp4`

2.  To run inference on an edge device, copy the following files, and then run inference as in 1. above. For the below commands, we use, `training-mc` to refer to training machine and `edge-mc` is Edge machine. `myvoice` is cloned on both training and the edge device in home folder. Best performing config was `configs/wlasl_gls.yaml` and model was written to `models/wlasl` folder. 
The below commands assume that you have `ssh` setup between `edge-mc` and `training-mc`. If not, follow any other process to get the referenced files to the `edge-mc`.  

```sh
# On edge-mc
cd myvoice
scp training-mc:~/myvoice/configs/wlasl_gls.yaml configs/
mkdir models/wlasl
scp training-mc:~/myvoice/models/wlasl/best.ckpt configs/wlasl/
scp training-mc:~/myvoice/models/wlasl/*.vocab configs/wlasl/
```

`best.ckpt` is a symbolic link created during training to point to the best checkpoint.




## Imported Libraries
We reference and use the following libraries. In order for these libraries to be included into the process pipelines or current versions, we had to make the following modifications.

### video_features - to extract features
https://github.com/v-iashin/video_features provides capabilities to extract features from videos using different models.  

The source code hardcodes the path to model checkpoints and hence doesn't work to include feature extraction in a pipeline like on a Edge device, or parallel processing with video formmating.

In this code, the following scripts are modified to use relative paths to identify checkpoint files

Files modified:
src/video_features/models/i3d/extract_i3d.py  
src/video_features/models/pwc/extract_pwc.py
src/video_features/models/raft/extract_raft.py
src/video_features/utils/utils.py 


**Note:** Only `extract_i3d.py` is used in `aslsignjoey`. Other files are changed to ensure conformance with `extract_i3d.py`.  

### SLT Joint End-To-End Sign Language Recognition and Translation
https://github.com/neccam/slt is used for training and evaluation. However, this code runs only on `torch==1.4.0` and `torchtext==0.5.0`. For the code to run on Jetson NX Xavier device that does not run 1.4, the following scripts were modified to use legacy code if newer versions are installed.

Files modified:
src/slt/signjoey/data.py
src/slt/signjoey/dataset.py
src/slt/signjoey/helpers.py
src/slt/signjoey/model.py
src/slt/signjoey/prediction.py
src/slt/signjoey/search.py 
src/slt/signjoey/training.py 
src/slt/signjoey/vocabulary.py 

Additionally, src/slt/signjoey/__main__.py is changed to paths so that the script can be run from `myvoice` and process inference.








