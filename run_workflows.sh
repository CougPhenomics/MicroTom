#! /usr/bin/bash

outdir=output/vis
mkdir -p $outdir

python \
$CONDA_PREFIX/bin/plantcv-workflow.py \
--dir data/vis \
--workflow scripts/visworkflow.py \
--type png \
--json $outdir/vis.json \
--adaptor filename \
--meta plantbarcode,measurementlabel,timestamp,camera,frame \
--outdir $outdir \
--delimiter "(.{2})_(.+)_(\d{8}T\d{6})_(.+)_(\d+)" \
--timestampformat "%%Y%%m%%dT%%H%%M%%S" \
--cpu 12 \
--writeimg \
--create

python $CONDA_PREFIX/bin/plantcv-utils.py json2csv -j $outdir/vis.json -c $outdir/vis.csv

