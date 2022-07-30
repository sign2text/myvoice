#!/bin/bash

if [ $# -ne 4 ]
then
   echo "Usage: $0 <indir> <outdir> <in_prefix> <out_prefix"
   echo "       <indir> Directory to process e.g. ./data/h2s/interim/ft"
   echo "       <outdir>  folder to write pickel to e.g. ./data/h2s "
s   echo "                All pickle files are written to this folder "
   echo "       <in_prefix>  is csv file prefix" 
   echo "                e.g. if csv files are how2sign_realigned_gls_[train|val|test].csv then pass \"how2sign_realigned_gls_\""
   echo "       <out_prefix>  is the prefix to use for output file" 
   echo "                e.g. \"h2s_realigned_gls_i3d_all\" to indicate realigned data with glosses using i3d features"
   exit 1
fi
# Change these values as needed.  
IN="$1"
OUT="$2"
DATASET="$3"
OUTF="$4"
SUBDIRS=( "train" "val" "test" )
CSV=".csv"

sudo chown -R $USER $PWD

if [ ! -d "${IN}" ]
then
    echo "***ERROR***: Extract folder $IN does not exist"
    exit 1
fi

for d in "${SUBDIRS[@]}"; do
    file="${IN}/${DATASET}${d}${CSV}"
    indir="${IN}/${d}"
    outfile="${OUT}/${OUTF}_${d}.pkl"
    if [ ! -f $file ]
    then
       echo "**WARN**: File $file not found. Skipped..."
    elif [ ! -d $indir ]
    then
       echo "**WARN**: Folder $indir not found. Skipped..."    
    else
       python3 src/aslsignjoey/data_prep.py "${file}" "${indir}" "${outfile}"
    fi
done
