#%%
import os, sys
from glob import glob
from pathlib import Path
from typing import List, Tuple
import yaml
import logging 

import torch


ADD_LIB = ['src','src/slt','src/video_features']
#%%
# def get_outfile(orgfile:str, suffix:str) -> str:
#     """Appends suffix to the filename, while preserving extension correctly"""
#     return os.path.join(os.path.dirname(orgfile),
#                         os.path.basename(orgfile).replace(".",suffix + "."))
#%%
def get_outfilename(org_file:str, out_folder:str=None, new_extn:str=None, suffix:str=None) -> str:
    in_file_split = os.path.splitext(os.path.basename(org_file))
    in_dir = os.path.dirname(org_file)
    out_base = in_file_split[0] + suffix if suffix else in_file_split[0]
    out_ext = in_file_split[1]
    if new_extn:
        out_ext = new_extn if new_extn[0] == "." else "." + new_extn
    out_dir = out_folder if out_folder else in_dir
    return os.path.join(out_dir, out_base + out_ext)
    
#%%
def get_rep_folder(orgfile:str) -> str:
    FILE = Path(orgfile).resolve()
    return str(FILE.parents[0])

#%%
def get_files(in_path:str, in_pattern:str) -> List[str]:
    """Gets files in in_path matching in_pattern - a thin wrapper around glob"""
    if os.path.isdir(in_path):
        files = glob(os.path.join(in_path,in_pattern))
    elif os.path.isfile(in_path):
        files = [in_path]
    else:
        raise FileNotFoundError(f"{in_path} is neither a file nor a folder")
    return files
#%%
def get_device(use_cuda_ovr:bool = True) -> Tuple[torch.device, bool]:
    """Gets the torch device to use based on cuda availability and overrides

    Parameters
    ----------
    use_cuda_ovr : Override to not use cuda even if present. 
                   Passing False will return cpu even if cuda is present.
                   Passing True has no effect, since cuda will be used only if available

    Returns
    -------
    torch.device : device to use - type cuda or cpu
    use_cuda : Whether cuda could be used
    """ 
    use_cuda = use_cuda_ovr and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return (device, use_cuda)

#%%
def set_path(paths:List[str] = ADD_LIB, baselevel:int =2) -> Path:
    """ Set system path to include sub libraries and return application root

        Inspiration from YOLOV5 code

        Deterimes in the baselevel of the application, 
        based on the program's location in the application. 
        Adds `paths` from the base level into system path

        Parameters
        ----------
        paths     : list of paths relative to application's base to add to system path
        baselevel : depth of the executing file from the application root                    

        Returns
        -------
        Path : the root path of the application, relative to current working directory 
            like `data`, `config` etc. can be referenced

    """
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[baselevel]  # Project src directory.

    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    
    for p in paths:
        if str(ROOT / p) not in sys.path:
            sys.path.append(str(ROOT / p))  

    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative from current working directory
    return ROOT

#%%
def nullable_string(val:str):
    """ Returns None is val is zero length. Useful for parsing command line args"""
    return None if not val else val

#%%
def get_logger(log_file:str = None,verbose:bool=True, append:bool=False) -> logging.Logger:
    """
    Create a logger for generic purposes. Adapted from slt/signjoey/helpers.py

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :param verbose:  include debug statements
    :return: logger object
    """

    """Set up logging to file and console."""
    logger = logging.getLogger(__name__)
    if verbose:
        FORMAT = "%(asctime)s %(levelname)-8s [%(module)s:%(funcName)s:%(lineno)d] %(message)s"  
    else:
        FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
    loglevel = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(loglevel)

    formatter = logging.Formatter(FORMAT)
    if log_file is not None:
        if append:
            filemode_val = 'a'
        else:
            filemode_val = 'w'
        fh_handler = logging.FileHandler(filename=log_file,mode=filemode_val)
        fh_handler.setLevel(loglevel)
        fh_handler.setFormatter(formatter)
        logger.addHandler(fh_handler)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(loglevel)
    formatter = logging.Formatter(FORMAT)
    console.setFormatter(formatter)

    logger.addHandler(console)

    return logger
# %%
def load_config(cfg_file):
    """Loads the cfg_file and optionally collapses levels if combine_dicts""" 
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg