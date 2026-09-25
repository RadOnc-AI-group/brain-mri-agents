#!/usr/bin/env bash
set -e

echo "=== Pulling BraTS Tumor Segmenters ==="
docker pull brainles/brats23_faking_it:latest
docker pull brainles/brats24_adult_glioma_faking_it:latest
docker pull brainles/brats23_met_nvauto:latest
docker pull brainles/brats24_meningioma_nic_vicorob

echo "=== Pulling SynthStrip (Skullstripping) ==="
docker pull freesurfer/synthstrip:1.8-gpu

echo "=== Pulling & Tagging SynthSeg ==="
docker pull ghcr.io/radonc-ai-group/brain-mri-agents/synthseg-robust-new:latest
docker tag ghcr.io/radonc-ai-group/brain-mri-agents/synthseg-robust-new:latest synthseg-robust-new:latest

echo "All segmentation containers pulled and ready."