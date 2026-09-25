import json
import asyncio

from pathlib import Path
from typing import List, Optional, Union, Dict

from agents import function_tool, RunContextWrapper

import nibabel as nib
import numpy as np
import pandas as pd
from panoptica import NaiveThresholdMatching, Metric, UnmatchedInstancePair, Panoptica_Evaluator, InputType

from util.data_types import PatientInput, TargetRegions, Patient
from util.context_manager import SessionContext


def check_volumetric_containment(arr_t0: np.ndarray, arr_t1: np.ndarray, containment_threshold: float = 0.8, 
                                 unmatched_t0: Optional[List[int]] = None, unmatched_t1: Optional[List[int]] = None) -> str:
    """
    Check if lesion volumes from one timepoint are substantially contained within lesions from another timepoint.
    Uses actual 3D volumetric overlap.
    Returns a message about potential fragmentation (one tumor splitting) or fusion (tumors merging).
    
    Args:
        arr_t0: Lesion map array for timepoint 0
        arr_t1: Lesion map array for timepoint 1
        containment_threshold: Minimum overlap ratio to consider containment (default 0.8)
        unmatched_t0: Optional list of unmatched lesion IDs from T0 to limit inspection
        unmatched_t1: Optional list of unmatched lesion IDs from T1 to limit inspection
    """
    instances_t0 = np.unique(arr_t0)
    instances_t0 = instances_t0[instances_t0 > 0]  # Exclude background
    instances_t1 = np.unique(arr_t1)
    instances_t1 = instances_t1[instances_t1 > 0]  # Exclude background
    
    # Determine which instances to check for containment
    check_t0 = instances_t0 if unmatched_t0 is None else np.array([i for i in instances_t0 if i in unmatched_t0])
    check_t1 = instances_t1 if unmatched_t1 is None else np.array([i for i in instances_t1 if i in unmatched_t1])
    
    if len(check_t0) == 0 and len(check_t1) == 0:
        return ""
    
    # Check containment: T1 lesions inside T0 lesions (fragmentation/descendants)
    t1_inside_t0 = {}  # t0_lesion_id -> [(t1_lesion_id, containment_ratio)]
    for t1_id in check_t1:  # Only check unmatched T1 lesions
        t1_mask = arr_t1 == t1_id
        t1_volume = np.sum(t1_mask)
        
        for t0_id in instances_t0:  # But check against ALL T0 lesions
            t0_mask = arr_t0 == t0_id
            intersection = np.sum(t1_mask & t0_mask)
            ratio = intersection / t1_volume if t1_volume > 0 else 0
            
            if ratio >= containment_threshold:
                t0_id_int = int(t0_id)
                if t0_id_int not in t1_inside_t0:
                    t1_inside_t0[t0_id_int] = []
                t1_inside_t0[t0_id_int].append((int(t1_id), ratio))
    
    # Check containment: T0 lesions inside T1 lesions (fusion/ancestors)
    t0_inside_t1 = {}  # t1_lesion_id -> [(t0_lesion_id, containment_ratio)]
    for t0_id in check_t0:  # Only check unmatched T0 lesions
        t0_mask = arr_t0 == t0_id
        t0_volume = np.sum(t0_mask)
        
        for t1_id in instances_t1:  # But check against ALL T1 lesions
            t1_mask = arr_t1 == t1_id
            intersection = np.sum(t0_mask & t1_mask)
            ratio = intersection / t0_volume if t0_volume > 0 else 0
            
            if ratio >= containment_threshold:
                t1_id_int = int(t1_id)
                if t1_id_int not in t0_inside_t1:
                    t0_inside_t1[t1_id_int] = []
                t0_inside_t1[t1_id_int].append((int(t0_id), ratio))
    
    # Format results
    messages = []
    
    if t1_inside_t0:
        for t0_lesion_id, matches in t1_inside_t0.items():
            t1_lesion_ids = [str(m[0]) for m in matches]
            ratios = [f"{m[1]:.1%}" for m in matches]
            messages.append(f"T1 lesion(s) {t1_lesion_ids} are {ratios} volumetrically contained in the region of T0 lesion [{t0_lesion_id}] → These lesion(s) are possible descendants of a split/shrinkage of that lesion and might not be absolute new lesions, needs human review.")
    
    if t0_inside_t1:
        for t1_lesion_id, matches in t0_inside_t1.items():
            t0_lesion_ids = [str(m[0]) for m in matches]
            ratios = [f"{m[1]:.1%}" for m in matches]
            messages.append(f"T0 lesion(s) {t0_lesion_ids} are {ratios} volumetrically contained in the region of T1 lesion [{t1_lesion_id}] → These lesion(s) possibly merged into the this lesion so it might not be an absolute new lesion, needs human review.")
    
    if messages:
        return "\n".join(messages)
    return ""


@function_tool
async def match_lesion_maps(wrapper: RunContextWrapper[SessionContext], name_timepoint0: str, name_timepoint1: str) -> str:
    """
    Match lesion instances between two timepoints of the same patient. 
    Matched lesions share the same instance ID in both timepoints.
    If timepoint T0 has unmatched instances, these can be vanished, absorbed or undersegmented lesions in T1.
    If timepoint T1 has unmatched instances, these are likely new lesions or false positives. 
    But also possible that a lesion has grown, shrunken or divided so much that it no longer passes the matching threshold with its counterpart in T0. Double-check the centroid and bounding box coordinates just in case.
    The matched lesion maps are stored back in the context under the same keys.
    IMPORTANT: This tool should be used before extracting tumor measurements, so that the measurements correspond to matched lesions.
    """
    timepoint_T0: Patient = wrapper.context.get(name_timepoint0)
    timepoint_T1: Patient = wrapper.context.get(name_timepoint1)
    if "lesion_map" not in timepoint_T0.loaded_images or "lesion_map" not in timepoint_T1.loaded_images:
        raise ValueError("Lesion maps not found in patient contexts.")

    image_T0 = timepoint_T0.loaded_images["lesion_map"]
    image_T1 = timepoint_T1.loaded_images["lesion_map"]

    arr_T0 = image_T0.get_fdata().astype(np.int32)
    arr_T1 = image_T1.get_fdata().astype(np.int32)
    
    if arr_T0.max() == 0 and arr_T1.max() == 0:
        msg = "No lesions found in either timepoint. Both lesion maps are empty."
    elif arr_T0.max() == 0:
        unmatched_T1 = np.unique(arr_T1).tolist()[1:]  # Exclude background
        msg = "No lesions found in T0. All instances in T1 are unmatched: {}".format(unmatched_T1)
    elif arr_T1.max() == 0:
        unmatched_T0 = np.unique(arr_T0).tolist()[1:]  # Exclude background
        msg = "No lesions found in T1. All instances in T0 are unmatched: {}".format(unmatched_T0)
    else:
        try:
            matcher = NaiveThresholdMatching(matching_threshold=0.25, matching_metric=Metric.IOU, allow_many_to_one=True)
            unmatched_input = UnmatchedInstancePair(prediction_arr=arr_T1, reference_arr=arr_T0)
            
            matched = matcher.match_instances(unmatched_input)
            arr_T0 = matched.reference_arr
            arr_T1 = matched.prediction_arr

            instances_T0 = np.unique(arr_T0)
            instances_T1 = np.unique(arr_T1)
            unmatched_T0 = np.setdiff1d(instances_T0, instances_T1).tolist()
            unmatched_T1 = np.setdiff1d(instances_T1, instances_T0).tolist()
        except Exception as e:
            print(e)
            return "Error during lesion matching: {}".format(e)
        # matched_instances = np.intersect1d(instances_T0, instances_T1) 
        # max_common = min(instances_T0.max(), instances_T1.max())
        # unmatched_T0 = instances_T0[instances_T0 > max_common].tolist()
        # unmatched_T1 = instances_T1[instances_T1 > max_common].tolist()

        image_T0 = nib.Nifti1Image(arr_T0, image_T0.affine, image_T0.header.copy())
        image_T1 = nib.Nifti1Image(arr_T1, image_T1.affine, image_T1.header.copy())

        msg = "Matched lesion maps stored in context under the same keys. Unmatched instances at T0: {}. Unmatched instances at T1: {}.".format(unmatched_T0, unmatched_T1)
        
        # Check for potential fragmentation/fusion using volumetric containment
        if unmatched_T0 != [] or unmatched_T1 != []:
            containment_msg = check_volumetric_containment(arr_T0, arr_T1, containment_threshold=0.5, 
                                                           unmatched_t0=unmatched_T0, unmatched_t1=unmatched_T1)
            if containment_msg:
                msg = msg + " However, " + containment_msg

    timepoint_T0.loaded_images["lesion_map"] = image_T0
    timepoint_T1.loaded_images["lesion_map"] = image_T1
    wrapper.context.set(timepoint_T0.name, timepoint_T0)
    wrapper.context.set(timepoint_T1.name, timepoint_T1)
    
    return msg


@function_tool
async def match_lesions_and_resection_cavity(wrapper: RunContextWrapper[SessionContext], patient_name_preop: str, patient_name_postop: str) -> str:
    """
    Match lesion instances from the pre-operative tumor segmentation with the resection cavity in the post-operative segmentation.
    Needs the pre-operative lesion map and post-operative segmentation to be loaded in the patient context under the respective patient names.
    """
    pass