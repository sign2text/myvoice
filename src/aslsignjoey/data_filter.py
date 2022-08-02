"""Filter input files based on different criteria

"""
#%%
import gzip
import pickle
import pandas as pd
import random
import re
import os
from utils import get_outfilename
from typing import List

#%%
def filter_signers(csv_file:str, pickle_file:str, out_path=None,
                   csv_field:str="VIDEO_NAME", pattern="-\d+-",
                   pickle_field:str = "name"):
    """Wrapper around individual functions to split a pickle_file

    Parameters
    ----------
    csv_file   : file containing csv_field from which matching pattern is obtained as signer
    pickle_file: 
    """
    signers = find_signers(csv_file, csv_field, pattern=pattern)
    split_pickle(pickle_file=pickle_file, out_path=out_path,
                 filter=signers, field=pickle_field,
                 match_type = "pattern")

def split_pickle(pickle_file:str, filter:List[str], 
                out_path:str=None, field:str="name", match_type:str="pattern"):
    """Wrapper around pickle_filter to split files
    
    Splits the pickle_file into different files based on 
    list of pattern matches in a given field.
    Output files are automatically suffixed with the match pattern
    """
    assert os.path.isfile(pickle_file), f"File {pickle_file} not found, or is not a file"
    if out_path:
        os.makedirs(out_path,exist_ok=True)
        out_dir = out_path
    else:
        out_dir = os.path.dirname(pickle_file)
    fbase = os.path.basename(pickle_file)
    for filt in filter:
        out_file = get_outfilename(pickle_file, out_folder=out_dir, suffix=filt)
        filter_pickle(pickle_file, out_file, 
            filter=filt, 
            field=field, 
            match_type=match_type)

def filter_pickle(pickle_file:str, out_file:str,
    filter:str, field:str="name", match_type:str="pattern"):
    """Filters a pickle file based on criteria specified

    Parameters
    ----------
    pickle_file: path for pickle file to process
    out_file   : path to store outfile. If path doesn't exist, it is created
    filter     : filter to apply
    field      : field to apply filter to
    match_type : type of match. One of [`pattern`, `exact`]
                 if pattern, it can match any regex pattern
                 if exact, whole field is matched.
    """
    assert os.path.isfile(pickle_file), f"File {pickle_file} not found, or is not a file"
    assert out_file, f"Output file must be specified"

    with gzip.open(pickle_file,"rb") as pf:
        loaded_file = pickle.load(pf)
    if match_type == "exact":
        filt_data = [x for x in loaded_file if x[field] == filter]
    elif match_type == "pattern":
        filt_data = [x for x in loaded_file if re.search(filter, x[field])]
    else:
        raise ValueError(f"Unknown value for match_type: {match_type}. Accepted: pattern,exact")

    print(f"Found {len(filt_data)} out of {len(loaded_file)} "
          f"matching {filter} in field {field} with match_type:{match_type}")

    out_path = os.path.dirname(out_file)
    os.makedirs(out_path,exist_ok=True)
    with gzip.open(out_file, 'wb') as ofile:
        pickle.dump(filt_data, ofile)

    print(f"Output written to {out_file}")


def find_signers(csv_file:str,
        field='VIDEO_NAME', pattern="-\d+-") -> list:
    
    assert(os.path.isfile(csv_file)), f"File {csv_file} not found, or is not a file"

    df = pd.read_csv(csv_file,sep="\t")
    df["signer"] = df[field].map(lambda x: re.search(pattern,x).group())
    return list(set(df["signer"]))
    

def sample_pickle(pickle_file:str, count:int=5, randomize=False) -> list:
    """Samples a pickle file and displays count records"""
    with gzip.open(pickle_file,"rb") as pf:
        loaded_file = pickle.load(pf)
    if randomize:
        indexes = random.sample(range(len(loaded_file)),count)
    else:
        indexes = list(range(count))
    print(f"Filtered indexes {indexes} out of {len(loaded_file)} records")
    return [loaded_file[i] for i in indexes]


#%%


