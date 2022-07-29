"""Extract features for a dataset

This script extracts features from video clips.
The script has been written for processing How2Sign datasets,
typically downloaded and extracted using `data_extract.py` 
and formatted using `data_format.py`  
However, this can be used with any data by changing the parameters.

This script expects a CSV file containing the list of clips to extract features for. 
Typically this file would have been generated as output of `data_format`

Example CSV file based on the (HOW2SIGN)[https://how2sign.github.io/#download]
 data is:  

|Field        | Comment                |  
|-------------|------------------------|  
|VIDEO_ID     | Not relevant           |  
|VIDEO_NAME	  | Not relevant |   
|SENTENCE_ID  | Not relevant           |  
|SENTENCE_NAME| Video file name. File with this name + .mp4 must exist in the input folder |
|START        | Not relevant    |  
|END          | Not relevant   |  
|SENTENCE     | Not relevant |  



All videos to be formatted should be in one common folder.

`get_tensor` can be used to extract features from individual video. 
This function doesn't rely on any command line argument, 
allowing it to be used during inference as well.

"""
#%%
import aslutils
from aslutils import nullable_string 
ROOT = aslutils.set_path()

import argparse
import os
import torch
from typing import List, Tuple



import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from data_features_i3d import get_tensor as get_tensor_i3d

device = None

RESULT_COLS = ["FEATURES","FT_MESSAGE","FT_MESSAGE_DATA"]

#%%
def extract_features(args:argparse.Namespace) -> pd.DataFrame:
    """Extract features based on command line arguments

    See `data_features.py --help` for more information on arguments

    pandas DataFrame containing three additional columns
    ['FEATURES','FT_MESSAGE','FT_MESSAGE_DATA']
    -   `Features` is True is video if the record had features extracted.  
    -   `FT Message` has the summary message about the row
    -   `FT Message Data` contains further details about the row process. 
        Especially useful for rows that did not extract.

    """
    assert os.path.isfile(args.csv), f"{args.csv} not found, or is not a file"
    assert os.path.isdir(args.in_folder), f"{args.in_folder} not found, or is not a folder"
    
    if args.ext[0] != ".":
        args.ext = "." + args.ext  

    os.makedirs(args.out_folder, exist_ok=True)

    df = pd.read_csv(args.csv, sep="\t")
    # Delegate the work to child process that accepts a dataframe
    df = extract_features_df(df, args)

    status_field, summ_field, detail_field = tuple(RESULT_COLS)

    # TODO: Get logger and send output to logger
    print(f"Summary: {df.groupby(summ_field).size().to_string()}")

    outfile = aslutils.get_outfilename(args.csv,
                       out_folder = aslutils.get_rep_folder(args.out_folder))
    #Save only the 
    df[df[status_field] == True].to_csv(outfile, sep="\t", index=False)
    print(f"Formatted info saved to {outfile}")

    if args.sfx_log:
        logfile = aslutils.get_outfilename(outfile,suffix=args.sfx_log)
#        logfile = aslutils.get_outfile(outfile, args.sfx_log)
        df.to_csv(logfile, sep="\t", index=False)
        print(f"Log saved to {logfile}")

    return df

#%%
def extract_features_df(df:pd.DataFrame, args:argparse.Namespace) -> pd.DataFrame:
    """Extracts features for videos based on metadata in df and args.

    Performs parallel data processing using all the available cores. 
    To turn off parallel processing pass `--no-parallel` in the args.

    This function can be called directly from other modules.
    e.g. if the calling function already has a dataframe in `df` then
    ```
    import data_format
    args = data_features.parse_args(["", "./data/h2s/interim/fmt", "./data/h2s/interim/ft"])
    df = data_format.format_videos_df(df, args)
    ```

    Parameters:
    -----------
    df: Pandas DataFrame the metadat that describes clip elements
    args: command line arguments specifying what to process

    Return:
    -------
    pandas DataFrame containing three additional columns
    ['FEATURES','FT_MESSAGE','FT_MESSAGE_DATA']
    -   `Features` is True is video if the record had features extracted.  
    -   `FT Message` has the summary message about the row
    -   `FT Message Data` contains further details about the row process. 
        Especially useful for rows that did not extract.

    """

    # While parallel runs are generally faster, if an error occurs,
    # message from parallel run is cryptic, making it difficult to find cause.
    # If user specified --no-parallel then don't run in parallel
    if args.no_parallel:
        tqdm.pandas() # This is only required if pandarallel errors out
        df[RESULT_COLS] = df.progress_apply(extract_rowclip, args=(args,), axis=1, 
                            result_type="expand")
    else:
        pandarallel.initialize(progress_bar=True)
        df[RESULT_COLS] = df.parallel_apply(extract_rowclip, args=(args,), axis=1, 
                            result_type="expand")
    return df

#%%
def extract_rowclip(row:pd.Series, args:argparse.Namespace) -> list:
    """Extracts features from a video clip based on row defintion

    Parameters:
    -----------
    row: Pandas DataFrame row containing the fields that describe clip elements
    args: command line arguments specifying what to process

    Return:
    -------
    list[bool, str, str] -> ['Features','FT_Message','FT_Message Data']
    -   `Features` is True is video for the record was extracted.  
    -   `Message` has the summary message about the row
    -   `Message Data` contains further details about the row process. 
        Especially useful for rows that did not extract.
    """
    result = None
    global device 

    infile_base = row[args.video_in]+args.ext
    infile = os.path.join(args.in_folder,infile_base)
    #File Check
    if not os.path.isfile(infile):
        result = [False,
                  f"File not found",
                  f"File {infile}"] 

    if result is None and args.filter:
        if not row[args.video_in] in args.filter:
            result = [False,
                    f"Video filter not matched",
                    f"Actual {row[args.video_in]}. Desired {args.filter}"] 

    #Finally process the extract
    if result is None:
        #No errors were found to discard video creation.
        if device is None:
            device, use_cuda = aslutils.get_device(args.use_cuda)

        out_file, features = get_tensor(infile, args.out_folder, device, args.type)
        if features is None or len(features) == 0:
            result = [False,
                    f"Feature Failed",
                    f"Received empty features {features.shape}"]


    if result is None:
        result = [True,
                f"Feature created",
                f"To file {out_file}, Shape:{features.shape} "]

    return result

#%%
def get_tensor(infile, outdir, device, feature_type) -> Tuple[str, torch.Tensor]:
    """
    Parameters
    ----------
    infile    : video file to extract features from
    outdir    : Optional directory in which to save the output 
                If specified, a numpy array is saved in outdir as infile.npy
    device_ovr: Device to use. If None, determines device based on cuda availability
    feature_type: Type of features to extract. Currently supported types are: 
    'i3d' - Extract using https://iashin.ai/video_features/models/i3d/ 

    Returns
    -------
    (file, tensor): Tuple containing output file name or None, and the tensor
    """    
    features = None
    if feature_type == "i3d":
        out_file, features = get_tensor_i3d(infile, outdir,device_ovr=device)
    return (out_file,features)
    

#%%
def parse_args(inline_options:List[str] = None, 
               known = False) -> argparse.Namespace:
    """Parse command line arguments

    Parses arguments either for command line or from inline_options
    
    Args:
    inline_options -- either list of key=value args for parsing.
                      By default, system command lines arguments are parsed
    known          -- Only known parameters are processed. 
                      Extra parameters are ignored. Helpful when parsing options 
                      passed from VSC.

    Returns:
    argparse.Namespace -- containing a dictionary of all arguments

    Example:
    parse_args() - for parsing command line options
    parse_args(['./data/raw/train.csv','./data/processed/extract/train','./data/processed/formatted/train'])
                 - the above will default all options
    data_features.py --help -- displays help with all parameters and defaults 
    """
    parser = argparse.ArgumentParser("Feature Extract")
    # Required parameters
    parser.add_argument("csv",          type=str,   help="CSV file containing the extract details")
    parser.add_argument("in_folder",    type=str,   help="Folder containing the formatted files")
    parser.add_argument("out_folder",   type=str,  help="Folder to write features to. If none, dataframe is returned with tensors (default: %(default)s)")

    # Less likely to change
    parser.add_argument("--type",       choices=["i3d","iv3"],  default="i3d", help="Type of features to extract (default: %(default)s)")
    parser.add_argument("--ext",        type=str,   default=".mp4",         help="Extension for input files (default: %(default)s)")
    # Column Names
    parser.add_argument("--video-in",   type=str,   default="SENTENCE_NAME",   help="Field in in_file containing the input video file name (default: %(default)s)")

    #Output naming
    parser.add_argument("--sfx-log",    type=nullable_string,  default="_log", help="Suffix to use for log file (default: %(default)s). Specify blank to skip")

    #Debug options
    parser.add_argument("--filter", type=str, nargs="+", help="Optional list of videos to process. Filter applied on the video-in specified (default: None)")
    parser.add_argument("--no-parallel", action="store_true", default=False,    help="Do not run in parallel")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Turn off cuda processing even if available (default: %(default)s)")

    if known is None:
        args = parser.parse_known_args(inline_options)[0] if inline_options else parser.parse_known_args()[0]
    else:
        args = parser.parse_args(inline_options) if inline_options else parser.parse_args()

    args.use_cuda = not args.no_cuda

    return args
#%%
if __name__ == "__main__":
    args = parse_args() 
    extract_features(args)