#!/bin/bash

if [ $# -ne 3 ]
then
   echo "Usage: $0 <indir> <outdir> <prefix>"
   echo "       <indir> Directory to process e.g. ./data/h2s/interim/fmt"
   echo "       <outdir>  folder to write formatted files to e.g. ./data/h2s/interim/ft"
   echo"                 Files are written to sub-folders under this folder"
   echo "       <prefix>  is csv file prefix" 
   echo "                e.g. if csv files are how2sign_realigned_gls_[train|val|test].csv then pass \"how2sign_realigned_gls_\""
   exit 1
fi
# Change these values as needed.  
IN="$1"
OUT="$2"
DATASET="$3"
SUBDIRS=( "train" "dev" "test" )
CSV=".csv"

if [ ! -d "${IN}" ]
then
    echo "***ERROR***: Format folder $IN does not exist"
    exit 1
fi

for d in "${SUBDIRS[@]}"; do
    file="${IN}/${DATASET}${d}${CSV}"
    indir="${IN}/${d}"
    outdir="${OUT}/${d}"
    if [ ! -f $file ]
    then
       echo "**WARN**: File $file not found. Skipped..."
    elif [ ! -d $indir ]
    then
       echo "**WARN**: Folder $indir not found. Skipped..."    
    else
       python3 src/aslsignjoey/data_features.py "${file}" "${indir}" "${outdir}" 
    fi
done
