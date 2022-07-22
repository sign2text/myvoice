# myvoice

## Data Preparation
How2Sign datasets are published at https://how2sign.github.io/


## Imported Libraries
We reference and use the following libraries. In order for these libraries to be included into the process pipelines or current versions, we had to make the following modifications.

### video_features - to extract features
https://github.com/v-iashin/video_features provides capabilities to extract features from videos using different models.  

The source code hardcodes the path to model checkpoints and hence doesn't work to include feature extraction in a pipeline like on a Edge device, or parallel processing with video formmating.

In this code, the following scripts are modified to use relative paths to identify checkpoint files

Files modified:
models/i3d/extract_i3d.py  
models/pwc/extract_pwc.py
models/raft/extract_raft.py

**Note:** Only `extract_i3d.py` is used in `aslsignjoey`. Other files are changed to ensure conformance with `extract_i3d.py`.  









