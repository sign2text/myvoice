"""Video Clip Extractor

This script extracts clips from videos between specific time frames for 
all videos in a given CSV file. 
The script has been written for processing How2Sign datasets, however, 
it can be used with any data by changing the parameters.

This script expects a CSV file containing the list of clips to be extracted, 
along with clip names. 

Example CSV file based on the (HOW2SIGN)[https://how2sign.github.io/#download]
 data is:  

|Field        | Comment                |  
|-------------|------------------------|  
|VIDEO_ID     | Not relevant           |  
|VIDEO_NAME	  | Video file name. File with this name + .mp4 must exist in the in_folder |   
|SENTENCE_ID  | Not relevant           |  
|SENTENCE_NAME| Name used to create the extracted clip. Extracted file is stored in the out_folder |
|START        | Time start clipping at  |  
|END          | Time to end clipping   |  
|SENTENCE     | Used to filter dataset for processing |  


All raw videos from which to extract clips should be in one common folder.

Examples:
--------
1. Happy path command to extract validation videos of How2Sign dataset based 
on realigned clips times.

```
python3 data_extract '../../data/raw/how2sign_realigned_val.csv' \
                       '../../data/raw/val/' \
                       '../../data/processed/val/' 
```

2. To extract only a sample frames for sentences containing less than ten words
and having 3 or more frames, pass the --max-words and --min-frames.

```
python3 data_extract '../../data/raw/how2sign_realigned_val.csv' \
                       '../../data/raw/val/' \
                       '../../data/processed/val/' \
                        --max-words 10 --min-frames 3
```

For help on all other parameters, see  
```
python3 data_extract --help
```

By default, extracted file information will be saved in `outfolder's` parent folder
which can be used in the next step of `data_format` as input.

"""
#%%
import argparse
import os
import cv2 
from tqdm import tqdm
from typing import List

import pandas as pd
from pandarallel import pandarallel

from aslutils import nullable_string, get_rep_folder, get_outfile

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip



#%%
class VideoClipProperties(object):
    """Holds common properties for a video clip"""
    def __init__(self, video_path:str):
        self.video_path = video_path

        self.exists = os.path.isfile(video_path)
        if self.exists:
            self._load_video()
            

    def _load_video(self):
        """Load the common properties using CV2"""
        cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count/self.fps 
        cap.release()

    def to_list(self):
        """Return common properties as a list"""
        return [self.frame_count, self.fps, self.duration]

    def list_names():
        """Defines the element names for the properties given by to_list"""
        return ["Frames","FPS","Duration"]
#%%
def extract_videos(args) -> pd.DataFrame:
    """Extract videos from multiple files

    Extracts videos based on parameters specified in CSV file

    Example:
    ```
    args = parse_args(['../../data/raw/how2sign_realigned_val.csv',
                       '../../data/raw/val/',
                       '../../data/processed/val/',
                       '--max-words','10',
                       '--min-frames','3',
                       '--video-in','VIDEO_NAME',
                       '--clip-type','clip'])
    df = extract_video(args)
    ```
    will create clips of `.mp4` files in `../../data/raw/val` that contain
    3 or more frames, and less than 10 words based from the video file 
    specified in column VIDEO_NAME in the CSV file 
    `../../data/raw/how2sign_realigned_val.csv`.

    See `data_extract.py --help` for more information on parameters

    Unless `--sfx-extract` is passed as empty, a csv file containing the data about
    extracted files will saved in same folder as `csv` file with file name including the suffix.

    Similarly, unless `--sfx-log` is passed as empty, a csv file include status 
    of each record will be saved. 

    """
    assert os.path.isfile(args.csv), f"{args.csv} not found, or is not a file"
    assert os.path.isdir(args.in_folder), f"{args.in_folder} not found, or is not a folder"

    if args.ext[0] != ".":
        args.ext = "." + args.ext  

    os.makedirs(args.out_folder, exist_ok=True)

    df = pd.read_csv(args.csv, sep="\t")
    tqdm.pandas()
    pandarallel.initialize(progress_bar=True)
    result_cols = ["EXTRACTED","MESSAGE","MESSAGE_DATA"]
    # While parallel runs are generally faster, if an error occurs,
    # message from parallel run is cryptic, making it difficult to find cause.
    # If user specified --no-parallel then don't run in parallel
    if args.no_parallel:
        df[result_cols] = df.progress_apply(extract_clip, args=(args,), axis=1, 
                            result_type="expand")
    else:
        df[result_cols] = df.parallel_apply(extract_clip, args=(args,), axis=1, 
                            result_type="expand")

    
    # TODO: Get logger and send output to logger
    print(f"Summary: {df.groupby('MESSAGE').size().to_string()}")

    outfile = os.path.join(get_rep_folder(args.out_folder),os.path.basename(args.csv))
    df[df.EXTRACTED].to_csv(outfile, sep="\t", index=False)
    print(f"Extracts saved to {outfile}")

    if args.sfx_log:
        logfile = get_outfile(outfile, args.sfx_log)
        df.to_csv(logfile, sep="\t", index=False)
        print(f"Log saved to {logfile}")

    return df

#%%
def extract_clip(row, args) -> list:
    """ Extracts video contents from a clip based on row defintion

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
    props_arr = [0,0,0]

    infile_base = row[args.video_in]+args.ext
    infile = os.path.join(args.in_folder,infile_base)
    #File Check
    if not os.path.isfile(infile):
        result = [False,
                  f"File not found",
                  f"File {infile}"] 
    #Word Count check
    if result is None:
        word_count = len(row[args.sentence].split())
        result = check_words(word_count,args.max_words) 

    if result is None and args.filter:
        if not row[args.video_in] in args.filter:
            result = [False,
                    f"Video filter not matched",
                    f"Actual {row[args.video_in]}. Desired {args.filter}"] 


    #Minimum frames check - Getting video properties is costly
    #So this should preferably the last check
    if result is None:
        video_props = VideoClipProperties(infile)
        prop_arr = video_props.to_list()
        result = check_frames(video_props, args.min_frames, "Input")

    #Check video properties and extract durations
    if result is None:
        clip_start = row[args.video_start]
        clip_end = row[args.video_end]
        clip_duration = clip_end - clip_start 
        if video_props.duration <= clip_start:
            result = [False,
                      "Clip smaller than start",
                      f"Actual Frames {video_props.frame_count} "
                      f"FPS: {video_props.fps} "
                      f"Duration: {video_props.duration} "
                      f"Desired {clip_start}:{clip_end} of duration {clip_duration} "]
        if video_props.duration <= clip_end:
            result = [False,
                      "Clip smaller than end",
                      f"Actual Frames {video_props.frame_count} "
                      f"FPS: {video_props.fps} "
                      f"Duration: {video_props.duration} "
                      f"Desired {clip_start}:{clip_end} of duration {clip_duration} "]

    #Finally process the extract
    if result is None:
        #No errors were found to discard video creation.
        outfile = os.path.join(args.out_folder,row[args.video_out] + args.ext)
        if args.clip_type == "clip":
            with VideoFileClip(infile) as video:
                newclip = video.subclip(row[args.video_start],row[args.video_end])
                newclip.write_videofile(outfile)
        else:
            ffmpeg_extract_subclip(infile,
                            row[args.video_start],row[args.video_end],
                            outfile)
        out_video = VideoClipProperties(outfile)
        result = check_frames(out_video, args.min_frames, "Output")
        if not result is None:
            os.remove(outfile)

    if result is None:
        result = [True,
                f"Extracted",
                f"To file {outfile} "
                f"Duration {out_video.duration:.2f}s "
                f"Frames {out_video.frame_count} "
                f"FPS {out_video.fps}"]

    return result


#%%
def check_words(word_count:int ,max_words: int) -> list:
    """ Utility function to check if sentence word_count is within the desired limit

    Parameters
    ----------
    word_count : number of words in the sentence
    max_words  : maximum words desired in the videos

    Returns
    -------
    list[bool, str, str] -> ['Extracted','Message','Message Data'] 
    """
    result = None
    if word_count > max_words:
        result = [False,
                  f"Words exceed max requested",
                  f"Words {len(sent_words)} requested {max_words}"]
    return result

def check_frames(video_prop:VideoClipProperties, min_frames:int, prefix="Input"):
  result = None
  if video_prop.frame_count <= min_frames:
     result = [False,
               f"{prefix} frames fewer than requested",
               f"{prefix} frames {video_prop.frame_count} requested {min_frames}"]
  return result

def get_outfile(orgfile, suffix):
    return os.path.join(os.path.dirname(orgfile),
                        os.path.basename(orgfile).replace(".",suffix + "."))



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
    parse_args(['./data/raw/train.csv','./data/raw/videos','./data/processed/extract_videos'])
                 - the above will default all options
    extract_video --help -- displays help with all parameters and defaults 
    """
    parser = argparse.ArgumentParser("Video Extract")
    # Required parameters
    parser.add_argument("csv",          type=str,   help="CSV file containing the extract details")
    parser.add_argument("in_folder",    type=str,   help="Folder containing the raw files")
    parser.add_argument("out_folder",   type=str,   help="Folder to write extracted files to")

    # Normally changeable parameters
    parser.add_argument("--max-words",  type=int,   default=1000,           help="Retain sentences containing less than max_words (default: %(default)s)")
    parser.add_argument("--min-frames", type=int,   default=2,              help="Filter out videos containing less than min_frames (default: %(default)s)")

    #Output naming
    parser.add_argument("--sfx-log",    type=nullable_string,  default="_log",         help="Suffix to use for log file (default: %(default)s). Specify blank to skip")

    #Debug options
    parser.add_argument("--clip-type",choices=["ffmpeg","clip"],  default="ffmpeg",  help="Which extract method to use (default: %(default)s).")
    parser.add_argument("--filter", type=str, nargs="+", help="Optional list of videos to process. Filter applied on the video-in specified (default: None)")
    parser.add_argument("--no-parallel", action="store_true", default=False,    help="Do not run in parallel")

    # Less likely to change
    parser.add_argument("--ext",        type=str,   default=".mp4",             help="Extension for input files (default: %(default)s)")
    # Column Names
    parser.add_argument("--video-in",   type=str,   default="VIDEO_NAME",       help="Field in csv containing the input video file name (default: %(default)s)")
    parser.add_argument("--video-out",  type=str,   default="SENTENCE_NAME",    help="Field in csv containing the output video file name (default: %(default)s)")
    parser.add_argument("--video-start",type=str,   default="START_REALIGNED",  help="Field in csv containing the clip start time (default: %(default)s)")
    parser.add_argument("--video-end",  type=str,   default="END_REALIGNED",    help="Field in csv containing the clip end time (default: %(default)s)")
    parser.add_argument("--sentence",   type=str,   default="SENTENCE",         help="Field in csv containing the sentence (default: %(default)s)")



    if known is None:
        args = parser.parse_known_args(inline_options)[0] if inline_options else parser.parse_known_args()[0]
    else:
        args = parser.parse_args(inline_options) if inline_options else parser.parse_args()
    return args

#%%
if __name__ == "__main__":
    args = parse_args() 
    extract_videos(args)
    
