#%%
from aslutils import set_path, get_device
ROOT = set_path()

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
def gen_tensors(indata:Union[str, List[str], pd.DataFrame],
                   df_field:str = None, 
                   tensor_field:str = None,
                   output_dir:str = None,
                   device:torch.device =None, 
                   reload_model:bool =False,
                   no_parallel:bool = True ) -> Union[Tuple[str,torch.Tensor], pd.DataFrame] :
    """Retrieve tensors for one of more videos
        
    """
    if type(indata) == pd.DataFrame:
        assert df_field and tensor_field, \
        f"Both df_field:{df_field} and tensor_field:{tensor_field} "
        f"must be valid for a dataframe"

    if device is None:
        device, use_cuda = get_device()

    global extractor, model, class_head
    if extractor is None or model is None or class_head is None or reload_model:
        print("Loading extractor")
        extractor, model, class_head = get_model(device)

    result = None
    if type(indata) == str:
        result = get_tensor(indata,outdir,device)
    elif type(indata) == list:
        #TODO improve performance on list processing by co
        sign_tensors = []
        if no_parallel:
            #Avoid the overhead of converting to a dataframe
            for f in tqdm(indata):
                sign_tensors.append(get_tensor(f,outdir,device))
            result = (indata, sign_tensors)
        else:
            #Convert to a dataframe and parallelize to improve performance
            df = pd.DataFrame({'files' : indata})
            df = get_tensor_df(df,device,'files','tensor')
            sign_tensors = df['tensor'].to_list()
            result = (indata, sign_tensors)
    elif type(indata) == pd.DataFrame:
        result = get_tensor_df(indata,outdir,device,df_field, tensor_field,no_parallel)

    return result

def get_tensor_df(df:pd.DataFrame, outdir:str, device:torch.device = None,
        df_field:str = None, tensor_field:str = None, no_parallel:bool=False) -> pd.DataFrame:
    if no_parallel:
        tqdm.pandas() 
        df[tensor_field] = df.progress_apply(get_tensor_row,args=(outdir,device))
    else:
        pandarallel.initialize(progress_bar=True)
        df[tensor_field] = df.parallel_apply(get_tensor_row,args=(outdir,device,))
    return df


def get_tensor_row(row:pd.Series, outdir:str, device:torch.device) -> torch.Tensor:
    return get_tensor(row[infile], outdir, device)
#%%
def get_tensor(infile:str, outdir:str, device_ovr=None) -> torch.Tensor:
    assert os.path.isfile(infile), f"File {infile} not found or is not a file"

    global extractor, model, class_head, device
    if extractor is None or model is None or class_head is None:
        extractor, model, class_head = get_model()

    if device is None:
        device, use_cuda = get_device() if device_ovr is None else (device_ovr, False)


    features = extractor.extract(device, model, class_head, infile)
    if outdir:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        action_on_extraction(features, infile, outdir, on_extraction='save_numpy')
    return torch.from_numpy(features['rgb'])

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

