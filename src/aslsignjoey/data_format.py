"""Video Clip Formatter

This script formats the video clips and prepare the data 
for feature extraction.
The script has been written for processing How2Sign datasets,
typically downloaded and extracted using `data_video_extract.py`
However, this can be used with any data by changing the parameters.

This script expects a CSV file containing the list of clips to be formatted. 
Typically this file would have been generated as output of `data_video_extract`

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


"""
#%%
import argparse
import os
import cv2 

from tqdm import tqdm

import pandas as pd
from pandarallel import pandarallel

from typing import List

from utils import get_outfile

#%%
def format_video(args):
    """Format video for feature extracttion.

    Extracts videos based on parameters specified in CSV file


    """
    assert os.path.isfile(args.csv), f"{args.csv} not found, or is not a file"
    assert os.path.isdir(args.in_folder), f"{args.in_folder} not found, or is not a folder"

    if args.ext[0] != ".":
        args.ext = "." + args.ext  

    os.makedirs(args.out_folder, exist_ok=True)

    df = pd.read_csv(args.csv, sep="\t")
    tqdm.pandas() # This is only required if pandarallel errors out
    pandarallel.initialize(progress_bar=True)
    result_cols = ["FORMATTED","MESSAGE","MESSAGE_DATA"]

    # While parallel runs are generally faster, if an error occurs,
    # message from parallel run is cryptic, making it difficult to find cause.
    # If user specified --no-parallel then don't run in parallel
    if args.no_parallel:
        df[result_cols] = df.progress_apply(format_clip, args=(args,), axis=1, 
                            result_type="expand")
    else:
        df[result_cols] = df.parallel_apply(format_clip, args=(args,), axis=1, 
                            result_type="expand")

    
    # TODO: Get logger and send output to logger
    print(f"Summary: {df.groupby('MESSAGE').size().to_string()}")

    if args.sfx_extract:
        outfile = get_outfile(args.csv,args.sfx_extract)
        df[df.FORMATTED].to_csv(outfile, sep="\t", index=False)

    if args.sfx_log:
        outfile = get_outfile(args.csv,args.sfx_log)
        df.to_csv(outfile, sep="\t", index=False)

    return df

#%%
def format_clip(row, args):
    result = None
    props_arr = [0,0,0]

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

        cap = cv2.VideoCapture(infile)
        if args.fps == 0:
            args.fps = cap.get(cv2.CAP_PROP_FPS)

        outfile = os.path.join(args.out_folder,infile_base)
        writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), 
                                 args.fps, args.frame_size)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame) # cut center of frame
                frame = cv2.resize(frame, args.frame_size) # resize frame
                writer.write(frame) # save frame into video file

        finally:
            cap.release()
            writer.release()
        

    if result is None:
        result = [True,
                f"Formatted",
                f"To file {outfile}"]

    return result

#%%
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


#%%
def nullable_string(val):
    return None if not val else val

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
    parser = argparse.ArgumentParser("Video Extract")
    # Required parameters
    parser.add_argument("csv",          type=str,   help="CSV file containing the extract details")
    parser.add_argument("in_folder",    type=str,   help="Folder containing the raw files")
    parser.add_argument("out_folder",   type=str,   help="Folder to write extracted files to")

    # Normally changeable parameters
    parser.add_argument("--frame-size",     type=int,   nargs=2, default=[240,240], help="Width Height specification for new video (default: %(default)s)")
    parser.add_argument("--fps",        type=int,   default=0,                  help="Feature per second for new video (default: %(default)s). Zero means use source FPS")

    # Less likely to change
    parser.add_argument("--ext",        type=str,   default=".mp4",         help="Extension for input files (default: %(default)s)")
    # Column Names
    parser.add_argument("--video-in",   type=str,   default="SENTENCE_NAME",   help="Field in in_file containing the input video file name (default: %(default)s)")

    #Output naming
    parser.add_argument("--sfx-extract",type=nullable_string,  default="_format",     help="Suffix to use for extracted only file (default: %(default)s. Specify blank to skip")
    parser.add_argument("--sfx-log",    type=nullable_string,  default="_format_log", help="Suffix to use for log file (default: %(default)s). Specify blank to skip")

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
    format_video(args)
    
