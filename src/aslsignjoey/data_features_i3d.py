"""I3D Feature Extractor

Wrapper module around https://github.com/v-iashin/video_features to generate
features from a video. Functions in this module allow video_features to be 
extracted for single piece inference or to be included in a pipeline.

`get_tensor` is the main function in this module.

"""
#%%
from aslutils import set_path, get_device, get_outfilename

ROOT = set_path()

import numpy as np
import pandas as pd
from pandarallel import pandarallel
import torch
import os
from tqdm import tqdm

from omegaconf import OmegaConf
from video_features.models.i3d.extract_i3d import ExtractI3D
from video_features.utils.utils import build_cfg_path, action_on_extraction

from typing import Union, Tuple, List

extractor, model, class_head, device = None, None, None, None

#%%
def get_tensor(infile:str, outdir:str=None, device_ovr=None) -> Tuple[str,torch.Tensor]:
    """Extracts I3D tensor from a `infile`

    Parameters
    ----------
    infile    : video file to extract features from
    outdir    : Optional directory in which to save the output 
                If specified, a numpy array is saved in outdir as infile.npy
    device_ovr: Device to use. If None, determines device based on cuda availability

    Returns
    -------
    (file, tensor): Tuple containing output file name or None, and the tensor
    """ 

    assert os.path.isfile(infile), f"File {infile} not found or is not a file"

    global extractor, model, class_head, device
    if extractor is None or model is None or class_head is None:
        extractor, model, class_head = get_model()

    if device is None:
        device, use_cuda = get_device() if device_ovr is None else (device_ovr, False)

    features = extractor.extract(device, model, class_head, infile)
    out_file = None
    if outdir:
        out_file = get_outfilename(infile, out_folder=outdir,new_extn=".npy")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        np.save(out_file, features['rgb'])        
    return (out_file, torch.from_numpy(features['rgb']))

#%%
def get_model(device:torch.device = None,
                 stack_size:int = 10,
                 step_size: int = 10,
                 flow_type: str = 'raft',
                 streams: str = 'rgb') -> tuple:
    """Loads the ExtractI3D model from video_features

    """
    # Select the feature type
    feature_type = 'i3d'
    if not device:
        device, use_cuda = get_device() 

    # Load and patch the config
    args = OmegaConf.load(os.path.join(ROOT / "src/video_features",
                          build_cfg_path(feature_type)))
    # stack size 10 means that 1 feature is extrated from the combination of 10 frames
    args.stack_size = stack_size # decrease stack size to be able to extract features from shorter videos, default is 64
    args.step_size = step_size # decrease stack size to be able to extract features from shorter videos, default is 64
    # args.extraction_fps = 25 # all videos are at 25 fps, no need to specify this
    args.flow_type = flow_type # 'pwc' is not supported on Google Colab (cupy version mismatch)
    args.streams = streams 

    # The below values are not used in pipeline - can be anything, but are required
    # Hence hardcoded and not exposed as parameters
    args.video_paths = [] # list of videos
    args.on_extraction='save_numpy'
    args.output_path = './extracted_features'    

      # Load the model
    extractor = ExtractI3D(args)
    model, class_head = extractor.load_model(device)
    return (extractor, model, class_head)


# %%
