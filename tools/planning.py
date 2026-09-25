import json
import asyncio

from pathlib import Path
from typing import List, Optional, Union, Dict

from agents import function_tool, RunContextWrapper

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from util.data_types import TargetRegions, Patient
from util.context_manager import SessionContext
from util.planning_utils import process_multiple_labels

@function_tool
async def find_ctv_mets_postop(wrapper: RunContextWrapper[SessionContext], patient_name: str, expansion_mm: Union[float, List[float]]) -> str:
    """
    This tool contours the Clinical Target Volume (CTV) for post-operative brain metastases patients. 
    It takes the segmented resection cavity and expands it by a specified margin in millimeters along the dura mater (if contacting) to create the CTV. The tool returns the CTV as a NIfTI file path and saves the path in the patient context.
    The tool requires:
    - Post-operative metastasis segmentation that includes the resection cavity, loaded in the patient context.
    - Dura mater segmentation, loaded in the patient context.
    - Patient name
    - Expansion margin in millimeters
    """

    patient: Patient = wrapper.context.get(patient_name)
    if patient is None:
        raise ValueError(f"Patient {patient_name} not found in context.")
    if 'dura' not in patient.loaded_images:
        raise ValueError(f"Dura segmentation not found for patient {patient_name}. Please load the dura mater segmentation into the patient context before running this tool.")
    if 'seg' not in patient.loaded_images:
        raise ValueError(f"Post-operative segmentation not found for patient {patient_name}. Please load the post-operative segmentation into the patient context before running this tool.")
    
    dura = patient.loaded_images['dura']
    seg = patient.loaded_images['seg']

    spacing = seg.header.get_zooms()[:3]

    ctv = process_multiple_labels(seg.get_fdata(), dura.get_fdata(), spacing, expansion_mm)
    
    ctv = nib.Nifti1Image(ctv, seg.affine, seg.header) 


    out_dir = wrapper.context.get("output_dir") / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_file = str(out_dir / f"{patient.name}_ctv_mets_postop.nii.gz")

    nib.save(ctv, seg_file)

    return "File with the CTV contours saved at: " + seg_file




@function_tool
async def is_tumor_touching_dura(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> str:
    """
    This tool checks if each instance of the pre-operative tumor is in contact with the dura mater. It requires the lesion-wise enumerated tumor segmentation 'lesion_maps' and dura mater segmentation to be loaded in the patient context. 
    The tool returns a message indicating whether the lesions are touching the dura mater or not.
    """

    patient: Patient = wrapper.context.get(patient_name)
    if patient is None:
        raise ValueError(f"Patient {patient_name} not found in context.")
    if 'dura' not in patient.loaded_images:
        raise ValueError(f"Dura segmentation not found for patient {patient_name}. Please load the dura mater segmentation into the patient context before running this tool.")
    if 'lesion_map' not in patient.loaded_images:
        raise ValueError(f"Lesion map not found for patient {patient_name}. Please load the lesion map into the patient context before running this tool.")
    
    dura = patient.loaded_images['dura']
    seg = patient.loaded_images['lesion_map']

    dura_data = dura.get_fdata()
    seg_data = seg.get_fdata()

    touching_lesions = []
    for les in np.unique(seg_data).tolist()[1:]:  # Skip background label 0
        lesion_mask = seg_data == les
        touching = np.sum((lesion_mask) & (dura_data > 0))
        if touching > 10:
            touching_lesions.append(les)

    if touching_lesions:
        return f"Lesions {touching_lesions} are touching the dura mater."
    else:
        return "No lesions are touching the dura mater."
        







