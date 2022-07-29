#%%
from aslutils import nullable_string, get_device, set_path
ROOT = set_path()
import argparse
from typing import List

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm


device = None
get_tensor_fn = None

def extract_features(args:argparse.Namespace) -> pd.DataFrame:
    assert os.path.isfile(args.csv), f"{args.csv} not found, or is not a file"
    assert os.path.isdir(args.in_folder), f"{args.in_folder} not found, or is not a folder"
    
    if args.ext[0] != ".":
        args.ext = "." + args.ext  

    os.makedirs(args.out_folder, exist_ok=True)

    df = pd.read_csv(args.csv, sep="\t")
    # Delegate the work to child process that accepts a dataframe
    df = extract_features_df(df, args)

    # TODO: Get logger and send output to logger
    print(f"Summary: {df.groupby('MESSAGE').size().to_string()}")

    outfile = os.path.join(get_rep_folder(args.out_folder),os.path.basename(args.csv))
    df[df.FEATURES].to_csv(outfile, sep="\t", index=False)
    print(f"Formatted info saved to {outfile}")

    if args.sfx_log:
        logfile = get_outfile(outfile, args.sfx_log)
        df.to_csv(logfile, sep="\t", index=False)
        print(f"Log saved to {logfile}")

    return df

#%%
def extract_features_df(df:pd.DataFrame, args:argparse.Namespace) -> pd.DataFrame:
    """Formats videos based on metadata in df and args.

    Performs parallel data processing using all the available cores. 
    To turn off parallel processing pass `--no-parallel` in the args.

    This function can be called directly from other modules.
    e.g. if the calling function already has a dataframe in `df` then
    ```
    import data_format
    args = data_format.parse_args(["", "./data/h2s/interim/ext", "./data/h2s/interim/fmt"])
    df = data_format.format_videos_df(df, args)
    ```

    Parameters:
    -----------
    df: Pandas DataFrame the metadat that describes clip elements
    args: command line arguments specifying what to process

    Return:
    -------
    list[bool, str, str] -> ['Features','Message','Message Data']
    -   `Features` is True is video if the record had features extracted.  
    -   `Message` has the summary message about the row
    -   `Message Data` contains further details about the row process. 
        Especially useful for rows that did not format.

    """
    result_cols = ["FEATURES","MESSAGE","MESSAGE_DATA"]

    # While parallel runs are generally faster, if an error occurs,
    # message from parallel run is cryptic, making it difficult to find cause.
    # If user specified --no-parallel then don't run in parallel
    if args.no_parallel:
        tqdm.pandas() # This is only required if pandarallel errors out
        df[result_cols] = df.progress_apply(extract_rowclip, args=(args,), axis=1, 
                            result_type="expand")
    else:
        pandarallel.initialize(progress_bar=True)
        df[result_cols] = df.parallel_apply(extract_rowclip, args=(args,), axis=1, 
                            result_type="expand")
    return df

#%%
def extract_rowclip(row:pd.Series, args:argparse.Namespace) -> list:
    """Formats video contents from a clip based on row defintion

    Parameters:
    -----------
    row: Pandas DataFrame row containing the fields that describe clip elements
    args: command line arguments specifying what to process

    Return:
    -------
    list[bool, str, str] -> ['Features','Message','Message Data']
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
            device, use_cuda = get_device(args.use_cuda)

        outfile = os.path.join(args.out_folder,infile_base)
        get_tensor(infile, outfile, device)

    if result is None:
        result = [True,
                f"Feature created",
                f"To file {outfile}"]

    return result


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
    parser.add_argument("out_folder",   type=nullable_string,   default=None, help="Folder to write features to. If none, dataframe is returned with tensors (default: %(default)s)")

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
    parser.add_argument("--no-cuda", action="store_true", default=False, help="Turn of cuda processing even if available (default: %(default)s)")

    if known is None:
        args = parser.parse_known_args(inline_options)[0] if inline_options else parser.parse_known_args()[0]
    else:
        args = parser.parse_args(inline_options) if inline_options else parser.parse_args()

    args.use_cuda = not args.no_cuda

    if args.type == "i3d":
        from data_features_i3d import get_tensor
    else:
        raise NotImplementedError(f"Option {args.type} is not implemented")
    return args
#%%
if __name__ == "__main__":
    args = parse_args() 
    format_videos(args)