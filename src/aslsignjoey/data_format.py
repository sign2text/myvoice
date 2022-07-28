"""Video Clip Formatter

This script formats the video clips and prepares the data 
for feature extraction.
The script has been written for processing How2Sign datasets,
typically downloaded and extracted using `data_extract.py`
However, this can be used with any data by changing the parameters.

This script expects a CSV file containing the list of clips to be formatted. 
Typically this file would have been generated as output of `data_extract`

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

`format_file` can be used to format individual file. 
This function doesn't rely on any command line argument, 
allowing it to be used during inference as well.

"""
#%%
import argparse
import os
import cv2 
from tqdm import tqdm

import pandas as pd
from pandarallel import pandarallel

from typing import List, Tuple

from aslutils import get_outfile, get_rep_folder, nullable_string

#%%
def format_videos(args: argparse.Namespace) -> None:
    """Format video for feature extraction.

    See `data_format.py --help` for arguments used

    """
    assert os.path.isfile(args.csv), f"{args.csv} not found, or is not a file"
    assert os.path.isdir(args.in_folder), f"{args.in_folder} not found, or is not a folder"

    if args.ext[0] != ".":
        args.ext = "." + args.ext  

    os.makedirs(args.out_folder, exist_ok=True)

    df = pd.read_csv(args.csv, sep="\t")
    # Delegate the work to child process that accepts a dataframe
    df = format_videos_df(df, args)

    
    # TODO: Get logger and send output to logger
    print(f"Summary: {df.groupby('MESSAGE').size().to_string()}")

    outfile = os.path.join(get_rep_folder(args.out_folder),os.path.basename(args.csv))
    df[df.FORMATTED].to_csv(outfile, sep="\t", index=False)
    print(f"Formatted info saved to {outfile}")

    if args.sfx_log:
        logfile = get_outfile(outfile, args.sfx_log)
        df.to_csv(logfile, sep="\t", index=False)
        print(f"Log saved to {logfile}")

    return df

#%%
def format_videos_df(df:pd.DataFrame, args:argparse.Namespace) -> pd.DataFrame:
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
    list[bool, str, str] -> ['Extracted','Message','Message Data']
    -   `Extracted` is True is video for the record was formatted.  
    -   `Message` has the summary message about the row
    -   `Message Data` contains further details about the row process. 
        Especially useful for rows that did not format.

    """
    result_cols = ["FORMATTED","MESSAGE","MESSAGE_DATA"]

    # While parallel runs are generally faster, if an error occurs,
    # message from parallel run is cryptic, making it difficult to find cause.
    # If user specified --no-parallel then don't run in parallel
    if args.no_parallel:
        tqdm.pandas() # This is only required if pandarallel errors out
        df[result_cols] = df.progress_apply(format_rowclip, args=(args,), axis=1, 
                            result_type="expand")
    else:
        pandarallel.initialize(progress_bar=True)
        df[result_cols] = df.parallel_apply(format_rowclip, args=(args,), axis=1, 
                            result_type="expand")
    return df

#%%
def format_rowclip(row:pd.Series, args:argparse.Namespace) -> list:
    """Formats video contents from a clip based on row defintion

    Parameters:
    -----------
    row: Pandas DataFrame row containing the fields that describe clip elements
    args: command line arguments specifying what to process

    Return:
    -------
    list[bool, str, str] -> ['Extracted','Message','Message Data']
    -   `Extracted` is True is video for the record was extracted.  
    -   `Message` has the summary message about the row
    -   `Message Data` contains further details about the row process. 
        Especially useful for rows that did not extract.
    """
    result = None

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

        outfile = os.path.join(args.out_folder,infile_base)
        format_video(infile, outfile, fps=args.fps, frame_size=args.frame_size)

    if result is None:
        result = [True,
                f"Formatted",
                f"To file {outfile}"]

    return result

#%%
def format_video(infile:str, outfile:str, 
                fps:int=0, frame_size:Tuple[int,int]=(224,224)) -> list:
    """Formats a given video clip

    This function can be imported into other modules making it convenient 
    for single inference methods

    Parameters
    ----------
    infile    : location of input video file
    outfile   : file to write output to
    fps       : frames per second for output. If 0, uses infiles frames per second
    frame_size: (width, height ).

    Returns
    -------
    list[bool, str, str] -> ['Extracted','Message','Message Data']
    -   `Extracted` is True is video for the record was extracted.  
    -   `Message` has the summary message about the row
    -   `Message Data` contains further details about the row process. 
    """
    result = None
    cap = cv2.VideoCapture(infile)
    if fps == 0:
        fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), 
                                fps, frame_size)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = format_frame(frame, frame_size)
            writer.write(frame) # save frame into video file
    except Exception as e:
        result = [False,"Failed", str(e)]
        
    finally:
        cap.release()
        writer.release()
    
    return result
            

#%%
def format_frame(frame, frame_size:Tuple[int,int]) :
    """Formats a single frame for feature extraction

    Parameters
    ----------
    frame: numpy.ndarray of a single frame from video
    frame_size: (width, height)

    Returns
    -------
    numpy.ndarray : frame
    """
    oframe = crop_center_square(oframe) # cut center of frame
    oframe = cv2.resize(frame, frame_size) # resize frame
    return oframe

#%%
def crop_center_square(frame):

    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


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
    data_video_format.py --help -- displays help with all parameters and defaults 
    """
    parser = argparse.ArgumentParser("Video Format")
    # Required parameters
    parser.add_argument("csv",          type=str,   help="CSV file containing the extract details")
    parser.add_argument("in_folder",    type=str,   help="Folder containing the extracted files")
    parser.add_argument("out_folder",   type=str,   help="Folder to write formated files to")

    # Normally changeable parameters
    parser.add_argument("--frame-size",     type=int,   nargs=2, default=[224,224], help="Width Height specification for new video (default: %(default)s)")
    parser.add_argument("--fps",        type=int,   default=0,                  help="Feature per second for new video (default: %(default)s). Zero means use source FPS")

    # Less likely to change
    parser.add_argument("--ext",        type=str,   default=".mp4",         help="Extension for input files (default: %(default)s)")
    # Column Names
    parser.add_argument("--video-in",   type=str,   default="SENTENCE_NAME",   help="Field in in_file containing the input video file name (default: %(default)s)")

    #Output naming
    parser.add_argument("--sfx-log",    type=nullable_string,  default="_log", help="Suffix to use for log file (default: %(default)s). Specify blank to skip")

    #Debug options
    parser.add_argument("--filter", type=str, nargs="+", help="Optional list of videos to process. Filter applied on the video-in specified (default: None)")
    parser.add_argument("--no-parallel", action="store_true", default=False,    help="Do not run in parallel")

    if known is None:
        args = parser.parse_known_args(inline_options)[0] if inline_options else parser.parse_known_args()[0]
    else:
        args = parser.parse_args(inline_options) if inline_options else parser.parse_args()
    args.frame_size = tuple(args.frame_size)
    return args

#%%
if __name__ == "__main__":
    args = parse_args() 
    format_videos(args)
    
