#!/bin/bash

if [ $# -ne 3 ]
then
   echo "Usage: $0 <indir> <outdir> <prefix>"
   echo "       <basedir> is starting directory e.g. ./data/h2s"
   echo "       <prefix>  is csv file prefix" 
   echo "                e.g. if csv files are how2sign_realigned_[train|val|test].csv then pass \"how2sign_realigned_\""
   echo "       <subdir>  is folder"
   exit 1
fi
# Change these values as needed.  
EXTRACT="$1"
FORMAT="$2"
DATASET="$3"
SUBDIRS=( "train" "val" "test" )
CSV=".csv"

sudo chown -R $USER $PWD

if [ ! -d "${EXTRACT}" ]
then
    echo "***ERROR***: Extract folder $EXTRACT does not exist"
    exit 1
fi

for d in "${SUBDIRS[@]}"; do
    file="${EXTRACT}/${DATASET}${d}${CSV}"
    extdir="${EXTRACT}/${d}"
    fmtdir="${FORMAT}/${d}"
    if [ ! -f $file ]
    then
       echo "**WARN**: File $file not found. Skipped..."
    elif [ ! -d $extdir ]
    then
       echo "**WARN**: Folder $extdir not found. Skipped..."    
    else
       python3 src/aslsignjoey/data_format.py "${file}" "${extdir}" "${fmtdir}"
    fi
done
