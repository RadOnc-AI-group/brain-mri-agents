import json
import shutil
import asyncio
import subprocess

from pathlib import Path
from typing import List, Optional, Union, Dict

from agents import function_tool, RunContextWrapper

import pandas as pd
import nibabel as nib
import numpy as np

from util.data_types import Patient, TargetRegions
from util.context_manager import SessionContext

from skimage.measure import label
from radiomics import featureextractor  
from torchio.data.io import nib_to_sitk 

@function_tool
async def extract_radiomics_features_tumor_core(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> str:
    """
    Tool to extract radiomics features from the tumor core region of interest. Uses T1C scan as default and if not found, uses T1.
    Requires tumor segmentation and lesion instance map to be present in the context under 'seg' and 'lesion_map' keys.
    """

    print("Radiomics Feature Extraction for Tumor Core is called.")

    patient: Patient = wrapper.context.get(patient_name)
    if "seg" not in patient.loaded_images:
        raise ValueError("Segmentation not found in context.")
    if "lesion_map" not in patient.loaded_images:
        raise ValueError("Lesion map not found in context.")

    seg_image = patient.loaded_images["seg"]
    seg_arr = seg_image.get_fdata()

    lesion_map = patient.loaded_images["lesion_map"].get_fdata()

    if 'placeholder' in patient.t1c.lower():
        mri_path = patient.t1
    else:
        mri_path = patient.t1c

    tc_mask = np.isin(seg_arr, [1, 3]).astype(np.uint8)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableAllImageTypes()
    
    rows = []
    for inst in np.unique(lesion_map)[1:]:
        tc_mask_lesion = np.where(lesion_map == inst, tc_mask, 0).astype(bool)
        tc_mask_lesion = label(tc_mask_lesion)
        for i in np.unique(tc_mask_lesion)[1:]:
            tc_mask_inst = np.where(tc_mask_lesion == i, 1, 0).astype(np.uint8)
            # tc_mask_inst = nib.Nifti1Image(np.expand_dims(tc_mask_inst, axis=0), seg_image.affine, seg_image.header.copy())
            tc_mask_inst = nib_to_sitk(np.expand_dims(tc_mask_inst, axis=0), affine=seg_image.affine, force_3d=True)
            # wt_path = Path.cwd() / "tmp" / f"{patient.name}_wt_mask_inst{int(inst)}.nii.gz"
            # nib.save(wt_mask_inst, wt_path)

            # Extract features
            features = extractor.execute(str(mri_path), tc_mask_inst)

            row = {'lesion_id': f"lesion_{int(inst)}_tumor_core_{int(i)}"}
            for key, value in features.items():
                # key format: "<imageType>_<featureClass>_<featureName>"
                if key.startswith(('diagnostics_', 'general_')):
                    continue  # Skip diagnostics/general info
                # parts = key.split('_')
                # if len(parts) >= 3:
                #     feature_name = parts[-1]  # Extract only <featureName>
                #     row[feature_name] = value
                row[key] = value  
        
        rows.append(row)

        # # Clean up temporary file
        # wt_path.unlink()
        # wt_path.parent.rmdir()

    df = pd.DataFrame(rows)
    csv_path = wrapper.context.get("output_dir") / patient_name / f"{patient.name}_tumor_core_radiomics.csv"
    df.to_csv(csv_path, index=False)

    patient.radiomics.append(str(csv_path))
    wrapper.context.set(patient.name, patient)

    return f"Radiomics features for tumor core lesions extracted and saved as CSV to '{csv_path}'."

@function_tool
async def extract_radiomics_features_whole_tumor(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> str:
    """
    Tool to extract radiomics features from the whole tumor region of interest. For metastasis, it will consider tumor core (TC) bounds.
    Uses FLAIR scan as default from 'patient' and if not found, uses T2.
    Requires tumor segmentation and lesion instance map to be present in the context under 'seg' and 'lesion_map' keys.
    """

    print("Radiomics Feature Extraction for Whole Tumor is called.")

    patient: Patient = wrapper.context.get(patient_name)
    if "seg" not in patient.loaded_images:
        raise ValueError("Segmentation not found in context.")
    if "lesion_map" not in patient.loaded_images:
        raise ValueError("Lesion map not found in context.")

    seg_image = patient.loaded_images["seg"]
    seg_arr = seg_image.get_fdata()

    lesion_map = patient.loaded_images["lesion_map"].get_fdata()

    if 'placeholder' in patient.flair.lower():
        mri_path = patient.t2
    else:
        mri_path = patient.flair

    wt_mask = np.isin(seg_arr, [2]).astype(np.uint8)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableAllImageTypes()
    
    rows = []
    for inst in np.unique(lesion_map)[1:]:
        wt_mask_inst = np.where(lesion_map == inst, wt_mask, 0)
        # wt_mask_inst = nib.Nifti1Image(np.expand_dims(wt_mask_inst, axis=0), seg_image.affine, seg_image.header.copy())
        wt_mask_inst = nib_to_sitk(np.expand_dims(wt_mask_inst, axis=0), affine=seg_image.affine, force_3d=True)
        # wt_path = Path.cwd() / "tmp" / f"{patient.name}_wt_mask_inst{int(inst)}.nii.gz"
        # nib.save(wt_mask_inst, wt_path)

        # Extract features
        features = extractor.execute(str(mri_path), wt_mask_inst)

        row = {'lesion_id': f"lesion_{int(inst)}"}
        for key, value in features.items():
            # key format: "<imageType>_<featureClass>_<featureName>"
            if key.startswith(('diagnostics_', 'general_')):
                continue  # Skip diagnostics/general info
            # parts = key.split('_')
            # if len(parts) >= 3:
            #     feature_name = parts[-1]  # Extract only <featureName>
            #     row[feature_name] = value
            row[key] = value  
        
        rows.append(row)

        # # Clean up temporary file
        # wt_path.unlink()
        # wt_path.parent.rmdir()

    df = pd.DataFrame(rows)
    csv_path = wrapper.context.get("output_dir") / patient_name / f"{patient.name}_whole_tumor_radiomics.csv"
    df.to_csv(csv_path, index=False)

    patient.radiomics.append(str(csv_path))
    wrapper.context.set(patient.name, patient)

    return f"Radiomics features for whole tumor lesions extracted and saved as CSV to '{csv_path}'."


async def extract_radiomics_features_brain_regions(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> str:
    """
    Tool to extract radiomics features from all segmented brain regions (e.g., hippocampus, ventricles). 
    Uses T1 scan as default and if not found, uses T1C.
    Requires segmentation of brain regions to be present in the context under 'brain_regions'.
    """

    print("Radiomics Feature Extraction for Brain Regions is called.")

    synthseg_classes = json.load(open("data/synthseg_classes.json"))
    synthseg_id_to_name = {v:k for k,v in synthseg_classes.items()}

    patient: Patient = wrapper.context.get(patient_name)
    if "brain_regions" not in patient.loaded_images:
        raise ValueError("Brain regions segmentation not found in context.")

    seg_image = patient.loaded_images["brain_regions"]
    seg_arr = seg_image.get_fdata()

    if 'placeholder' in patient.t1.lower():
        mri_path = patient.t1c
    else:
        mri_path = patient.t1

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableAllImageTypes()
    
    rows = []
    for region_id in np.unique(seg_arr):
        if region_id == 0:
            continue  # Skip background
        region_mask = (seg_arr == region_id).astype(np.uint8)
        # region_mask_img = nib.Nifti1Image(region_mask, seg_image.affine, seg_image.header.copy())
        region_mask_img = nib_to_sitk(region_mask, affine=seg_image.affine, force_3d=True)
        # region_path = Path.cwd() / "tmp" / f"{patient.name}_region_{int(region_id)}.nii.gz"
        # nib.save(region_mask_img, region_path)

        # Extract features
        features = extractor.execute(str(mri_path), region_mask_img)

        row = {'region_name': synthseg_id_to_name.get(str(int(region_id)))}
        for key, value in features.items():
            # key format: "<imageType>_<featureClass>_<featureName>"
            if key.startswith(('diagnostics_', 'general_')):
                continue  # Skip diagnostics/general info
            # parts = key.split('_')
            # if len(parts) >= 3:
            #     feature_name = parts[-1]  # Extract only <featureName>
            #     row[feature_name] = value
            row[key] = value  
        
        rows.append(row)

        # # Clean up temporary file
        # region_path.unlink()
        # region_path.parent.rmdir()

    df = pd.DataFrame(rows)
    csv_path = wrapper.context.get("output_dir") / patient_name / f"{patient.name}_brain_regions_radiomics.csv"
    df.to_csv(csv_path, index=False)

    patient.radiomics.append(str(csv_path))
    wrapper.context.set(patient.name, patient)

    return f"Radiomics features for brain regions extracted and saved as CSV to '{csv_path}'."

def extract_radiomics_features_tumor_core_no_tool(patient: Patient) -> str:
    """
    Tool to extract radiomics features from the tumor core region of interest. Uses T1C scan as default and if not found, uses T1.
    Requires tumor segmentation and lesion instance map to be loaded in the context.
    """

    print("Radiomics Feature Extraction for Tumor Core is called.")

    seg_image = patient.loaded_images["seg"]
    seg_arr = seg_image.get_fdata()

    lesion_map = patient.loaded_images["lesion_map"]

    if 'placeholder' in patient.t1c.lower():
        mri_path = patient.t1
    else:
        mri_path = patient.t1c

    tc_mask = np.isin(seg_arr, [1, 3]).astype(np.uint8)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableAllImageTypes()
    
    rows = []
    for inst in np.unique(lesion_map)[1:]:
        tc_mask_lesion = np.where(lesion_map == inst, tc_mask, 0).astype(bool) # if glioma/meningioma, select only TC(s) inside WT; if metastasis, this is already TC
        tc_mask_lesion = label(tc_mask_lesion)
        for i in np.unique(tc_mask_lesion)[1:]:
            tc_mask_inst = np.where(tc_mask_lesion == i, 1, 0).astype(np.uint8)
            # tc_mask_inst = nib.Nifti1Image(np.expand_dims(tc_mask_inst, axis=0), seg_image.affine, seg_image.header.copy())
            tc_mask_inst = nib_to_sitk(np.expand_dims(tc_mask_inst, axis=0), affine=seg_image.affine, force_3d=True)
            # wt_path = Path.cwd() / "tmp" / f"{patient.name}_wt_mask_inst{int(inst)}.nii.gz"
            # nib.save(wt_mask_inst, wt_path)

            # Extract features
            features = extractor.execute(str(mri_path), tc_mask_inst)

            row = {'lesion_id': f"lesion_{int(inst)}_tumor_core_{int(i)}"}
            for key, value in features.items():
                # key format: "<imageType>_<featureClass>_<featureName>"
                if key.startswith(('diagnostics_', 'general_')):
                    continue  # Skip diagnostics/general info
                # parts = key.split('_')
                # if len(parts) >= 3:
                #     feature_name = parts[-1]  # Extract only <featureName>
                #     row[feature_name] = value
                row[key] = value  # keep full key for now
        
        rows.append(row)

        # # Clean up temporary file
        # wt_path.unlink()
        # wt_path.parent.rmdir()

    df = pd.DataFrame(rows)
    csv_path = f"{patient.name}_tumor_core_radiomics.csv"
    df.to_csv(csv_path, index=False)

    patient.radiomics.append(str(csv_path))

    return f"Radiomics features for tumor core lesions extracted and saved as CSV to '{csv_path}'."
