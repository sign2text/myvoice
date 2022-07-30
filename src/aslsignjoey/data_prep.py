"""Prepare data for training

This module combines features with metadata required for training the 
SLT model.

Required / positional Parameters
--------------------------------
csv       : CSV file containing records to process
in_folder : Folder containing the numpy arrays typically generated from data_features.py
out_file  : Path and filename to write output to

Output
------
Generates pickle file that can be used as input to the model training 
If errors occur, will write a csv file with errors. 
This file would include a message column describing the error

data_prep.py --help -- displays help with all parameters and defaults 


"""
#%%
import argparse 
import os
import gzip, pickle
import re
from typing import List
from tqdm import tqdm

from aslutils import nullable_string, get_outfilename

import numpy as np
import pandas as pd 
from pandarallel import pandarallel
import torch
#%%
def prepare_data(args):
    """Prepares data for use in training

    Main workhorse implementing the module function. 
    Process the CSV and creates a pickle file use in training.
    Pickle stores a list of dictionaries containing, gloss,text,sign 
    and other fields required
    Only successful records are written to pickle file. 
    All failed records are written to a CSV file in the output folder

    See `data_prep.py --help` and module documentation for parameters
    """
    df = pd.read_csv(args.csv, sep="\t")
    
    dataset = []
    # Parallel processing not implemented since result should be in list
    tqdm.pandas()
    dataset = df.progress_apply(get_rowdata, args=(args,), axis=1).to_list()

    success_data = []
    failed_data = []
    for d in dataset:
        success = d['success']
        d.pop('success')
        if success:
            d.pop('message')
            success_data.append(d)
        else:
            failed_data.append(d)
    success_file, failed_file = None, None
    if len(success_data) > 0:
        success_file = get_outfilename(args.out_file,new_extn='.pkl')
        with (gzip.open(success_file,"wb")) as fp:
            pickle.dump(success_data,fp)

    if len(failed_data) > 0:
        failed_file = get_outfilename(args.out_file,new_extn='.csv', suffix=args.sfx_failed)
        fd = pd.DataFrame(failed_data)
        fd.to_csv(failed_file, index=False, sep="\t")

    print(f"Processed {len(df)} records from {args.csv}.\n"
          f"Success:{len(success_data)}, Failed:{len(failed_data)}")

    if success_file: print(f"Successful records written to {success_file}")     
    if failed_file: print(f"Failed records written to {failed_file}")     
    
    

#%%
def get_rowdata(row,args):
    """Process each row and returns a dictionary containing required fields"""
    fields = {}
    npyfile = os.path.join(args.in_folder,row[args.video_in] + args.npy_ext)
    mp4file = get_outfilename(npyfile,new_extn=args.video_ext)
    fields['gloss'] = _get_gloss(row, args)
    fields['text']  = _get_text(row,args) 
    fields['name']  = _get_name(row,args)
    fields['signer'] = _get_signer(row, args)
    if os.path.isfile(npyfile):
        fields['sign']  = torch.from_numpy(np.load(npyfile, allow_pickle=True)).float()
        fields['success'] = True
        fields['message'] = None
    elif os.path.isfile(mp4file):
        fields['sign'],fields['success'],fields['message'] = _generate_tensor(row, args, mp4file)
    else:
        fields['sign'] = None
        fields['success'] = False
        fields['message'] = f"Neither npy nor mp4 file found. npy:{npyfile} mp4:{mp4file}"
    return fields

#%%
def _get_name(row, args):
    """Generates the name field"""
    field = None
    if args.name and args.name in row.index:
        field = row[args.name]
    elif args.video_in and args.video_in in row.index:
        field = row[args.video_in]
    if args.name_prefix:
        field = args.name_prefix + field
    else:
        field = os.path.basename(args.in_folder) + '/' + field
    return field

#%%
def _get_gloss(row, args):
    """Generates the gloss field"""
    field = None
    if args.gloss and args.gloss in row.index:
        field = row[args.gloss]
    else:
        field = row[args.text].upper()
    return field

#%%
def _get_text(row, args):
    """Returns the text field. 

    In future implements, this function could perform text corrections
    """
    return row[args.text]     
#%%
def _get_signer(row, args):
    """Extracts signer information from other fields, or returns a default"""
    field = None
    if args.signer in row.index:
        field = row[args.signer]
        if args.signer_pat:
            match = re.search(args.signer_pat,field)
            field = match.group() if match else ""
            field = args.signer_prefix + field
    else:
        field = args.signer
    return field
#%%
def _generate_tensor(row,args, file):
    """Not implemented"""
    return (None, False, f"MP4 preparation not implemented. File {file} found")
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
    """
    parser = argparse.ArgumentParser("Data Preparation")
    # Required parameters
    parser.add_argument("csv",          type=str,   help="CSV file containing the extract details")
    parser.add_argument("in_folder",    type=str,   help="Folder containing the extracted files")
    parser.add_argument("out_file",     type=str,   help="File to write successful records to")


    # Less likely to change
    parser.add_argument("--npy-ext",        type=str,   default=".npy",         help="Extension for input files (default: %(default)s)")
    parser.add_argument("--video-ext",        type=str,   default=".mp4",         help="Extension for input files (default: %(default)s)")

    # Column Names
    parser.add_argument("--video-in",   type=str,   default="SENTENCE_NAME",   help="Field in csv containing the input video file name (default: %(default)s)")
    parser.add_argument("--gloss",       type=str,   default="gloss", help="Field in csv containing the glosses (default %(default)s)")
    parser.add_argument("--text",       type=str,   default="curated_sentence", help="Field in csv containing the text (default %(default)s)")
    parser.add_argument("--signer",     type=str,   default="VIDEO_NAME", help="Field in csv containing the signer name (default %(default)s). If column is not found, this value will be used as signer")
    parser.add_argument("--signer-pat", type=nullable_string,   default="-\d+-", help="Pattern to match to extract signer from signer field (default %(default)s). Pass empty to ignore")
    parser.add_argument("--signer-prefix", type=nullable_string, default="Signer", help="Prefix to use to generate signer (default %(default)s) Ignored if column is not present in data" )
    parser.add_argument("--name",        type=str, default="SENTENCE_NAME", help="Field in csv to use as name (default %(default)s). If none, video_in field will be used" )
    parser.add_argument("--name-prefix", type=nullable_string, default="", help="Prefix to use to generate name (default %(default)s). If none, trailing element of in-folder is used" )

    #Output naming
    parser.add_argument("--sfx-failed",    type=nullable_string,  default="_failed", help="Suffix to use for failed records (default: %(default)s)")
 
    #Debug options
    parser.add_argument("--filter", type=str, nargs="+", help="Optional list of videos to process. Filter applied on the video-in specified (default: None)")
    parser.add_argument("--no-parallel", action="store_true", default=False,    help="Do not run in parallel")

    if known is None:
        args = parser.parse_known_args(inline_options)[0] if inline_options else parser.parse_known_args()[0]
    else:
        args = parser.parse_args(inline_options) if inline_options else parser.parse_args()

    return args

#%%
if __name__ == "__main__":
    args = parse_args() 
    prepare_data(args)
    