#!/bin/bash
  
if [ $# -ne 3 ]
then
   echo "Usage: $0 <indir> <outdir> <prefix>"
   echo "       <indir> is starting directory for input videos e.g. ./data/h2s/raw"
   echo "       <outdir> is starting directory for output videos e.g. ./data/h2s/interim/extract"
   echo "       <prefix>  is csv file prefix" 
   echo "                e.g. if csv files are how2sign_realigned_[train|val|test].csv then pass \"how2sign_realigned_\""
   exit 1
fi

# Change these values as needed.
RAW="$1"
EXTRACT="$2"
DATASET="$3"
SUBDIRS=( "train" "val" "test" )
CSV=".csv"

sudo chown -R $USER $PWD

if [ ! -d "${RAW}" ]
then
    echo "***ERROR***: Raw folder $RAW does not exist"
    exit 1
fi

for d in "${SUBDIRS[@]}"; do
    file="${RAW}/${DATASET}${d}${CSV}"
    rawdir="${RAW}/${d}"
    extdir="${EXTRACT}/${d}"
    if [ ! -f $file ]
    then
       echo "**WARN**: File $file not found. Skipped..."
    elif [ ! -d $rawdir ]
    then
       echo "**WARN**: Folder $rawdir not found. Skipped..."    
    else
       python3 src/aslsignjoey/data_extract.py "${file}" "${rawdir}" "${extdir}"
    fi
done
