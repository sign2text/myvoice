#!/bin/bash
# Change these values as needed.  
RAW="./data/raw"
EXTRACT="./data/interim/extract"
SUBDIRS=( "train" "val" "test" )
DATASET="how2sign_realigned_"
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
