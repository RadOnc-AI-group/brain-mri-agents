from curses import wrapper
import json
import shutil
import asyncio

from pathlib import Path
from typing import List, Optional, Union, Dict, Literal

from agents import function_tool, RunContextWrapper

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

from util.data_types import PatientInput, TargetRegions, Patient
from util.context_manager import SessionContext

# BRATS_CLASSES = json.load(open("./data/brats_classes.json"))

    # Args:
    #     image_name (str): Name of the segmentation mask variable in the global scope.
    #     target_regions (TargetRegions): Region names with label lists.
    
@function_tool
async def select_subregions(wrapper: RunContextWrapper[SessionContext], patient_name: str, target_regions: TargetRegions) -> str:
    """
    Extract specified regions from the segmentation mask, preserving original label values.
    Only use if you want to analyze specific subgroups from all segmented regions.
    Requires that the segmentation mask is already loaded.
    Saves the result in the context as [patient_name]['seg_selected_regions'].
    """
    patient: Patient = wrapper.context.get(patient_name)
    if "seg" not in patient.loaded_images:
        raise ValueError(f"Segmentation file is not loaded as 'seg' in patient context for '{patient_name}'.")
    seg = patient.loaded_images["seg"]
    array = seg.get_fdata()
    names = target_regions.names
    labels = target_regions.labels

    # ## old -- multi-channel mask
    # num_classes = len(names)
    # new_seg = np.expand_dims(np.zeros_like(array), 0).repeat(num_classes, axis=0)  # Initialize a multi-channel mask
    # for i, (key, labels) in enumerate(zip(names, labels), start=0):
    #     for j in labels:
    #         new_seg[i] = np.where(array == j, 1, new_seg[i])

    labels = [item for sublist in labels for item in sublist]  # flatten the list of lists
    new_seg = np.zeros_like(array)  # Initialize a single-channel mask
    where = np.isin(array, labels)
    new_seg = np.where(where, array, new_seg)   # keep original label values
    
    new_seg = nib.Nifti1Image(new_seg, affine=seg.affine, header=seg.header)

    # wrapper.context[patient.name].set(f"seg_selected_regions", new_seg)
    patient.loaded_images["seg_selected_regions"] = new_seg
    wrapper.context.set(patient_name, patient)

    return f"Selected regions saved in patient context under 'seg_selected_regions' for patient '{patient_name}'."


@function_tool
async def get_tumor_lesions_map(wrapper: RunContextWrapper[SessionContext], patient_name: str, tumor_type: Literal['glioma', 'meningioma', 'metastasis'] = 'glioma') -> str:
    """
    Identify and label individual tumor lesions. Uses 'whole tumor' labels to count separate instances for gliomas and meningiomas, and 'tumor core' for metastasis.
    Requires that the segmentation mask is already loaded.
    The result is a single-channel mask with enumerated labels for each lesion, saved globally as `lesion_map` into the patient context.
    """
    patient: Patient = wrapper.context.get(patient_name)
    if "seg" not in patient.loaded_images:
        raise ValueError(f"Segmentation file is not loaded as 'seg' in patient context for '{patient_name}'.")
    seg = patient.loaded_images["seg"]
    array = seg.get_fdata().astype(np.uint8)

    if tumor_type == 'glioma' or tumor_type == 'meningioma':
        include = [1, 2, 3]  # whole tumor for glioma and meningioma
    elif tumor_type == 'metastasis':
        include = [1, 3]  # tumor core for metastasis
    
    lesions = np.isin(array, include)
    lesions = label(lesions, connectivity=3)
    lesions = lesions.astype(np.int16)
    # if tumor_type == 'glioma' or tumor_type == 'meningioma':
    #     lesions = remove_small_objects(lesions.astype(np.int16), min_size=10)

    lesions = nib.Nifti1Image(lesions.astype(np.float32), affine=seg.affine, header=seg.header)
    patient.loaded_images["lesion_map"] = lesions
    wrapper.context.set(patient_name, patient)

    return f"Lesion map created and saved in patient context under 'lesion_map' for patient '{patient_name}'."

@function_tool
async def get_disconnected_resection_cavities(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> str:
    """
    Identify disconnected resection cavity regions in the post-operative segmentation. Requires that the post-operative segmentation is already loaded in the patient context under 'seg'.
    The result is a list of volumes for each disconnected cavity region, saved globally as `disconnected_cavity_volumes` into the patient context.
    """
    # patient: Patient = wrapper.context.get(patient_name)
    # if "seg" not in patient.loaded_images:
    #     raise ValueError(f"Segmentation file is not loaded as 'seg' in patient context for '{patient_name}'.")
    # seg = patient.loaded_images["seg"]
    # array = seg.get_fdata()
    # spacing = seg.header.get_zooms()[:3]
    # voxel_volume = np.prod(spacing)  # Volume of a single voxel in mm^3

    # cavity = label(array == 4, connectivity=3)
    # cavity_props = regionprops(cavity.astype(np.int16), spacing=spacing)
    # volumes = [prop.area * voxel_volume for prop in cavity_props if prop.area >= 10]  # list of volumes for disconnected cavity regions
    # patient.disconnected_cavity_volumes = volumes
    # wrapper.context.set(patient_name, patient)

    # return f"Disconnected resection cavity volumes computed and saved in patient context under 'disconnected_cavity_volumes' for patient '{patient_name}'."

    pass

@function_tool
async def select_one_lesion(wrapper: RunContextWrapper[SessionContext], patient_name: str, lesion_id: int) -> str:
    """
    Select a single lesion from the lesion map by its ID and use to filter "seg" to keep only that lesion.
    Requires that the seg and lesion map are already loaded.
    Saves the result in the context as [patient_name]['seg_selected_lesion'].
    """
    patient: Patient = wrapper.context.get(patient_name)
    if "lesion_map" not in patient.loaded_images:
        raise ValueError(f"Lesion map is not loaded in patient context for '{patient_name}'.")
    if "seg" not in patient.loaded_images:
        raise ValueError(f"Segmentation file is not loaded as 'seg' in patient context for '{patient_name}'.")
    
    lesion_map = patient.loaded_images["lesion_map"]
    array = lesion_map.get_fdata()
    seg = patient.loaded_images["seg"].get_fdata()  # Use the original segmentation to keep the same labels and values for the selected lesion

    selected_lesion = np.where(array == lesion_id, seg, 0).astype(np.float32)  # Filter out the selected lesion
    selected_lesion = nib.Nifti1Image(selected_lesion, affine=lesion_map.affine, header=lesion_map.header)

    patient.loaded_images[f"seg_lesion{lesion_id}"] = selected_lesion
    wrapper.context.set(patient_name, patient)

    return f"Selected lesion {lesion_id} saved in patient context under 'seg_lesion{lesion_id}' for patient '{patient_name}'."


# @function_tool
# async def get_target_volumes(wrapper: RunContextWrapper[SessionContext], image_name: str, target_regions: TargetRegions) -> dict[str, float]:
#     """
#     Compute the volume of all target channels in a segmentation mask with selected regions.
#     The output is a dictionary mapping region names to their volumes.

#     Args:
#         image_name (str): Name of the segmentation variable stored in the context.
#         target_regions (TargetRegions): The target regions to measure.
#     Returns:
#         dict[str, float]: Computed volumes in cubic millimeters.
#     """
#     if image_name not in wrapper.context:
#         raise ValueError(f"Segmentation file '{image_name}' is not loaded.")

#     seg = wrapper.context.get(image_name)
#     array = seg.get_fdata()
#     voxel_volume = np.prod(seg.header.get_zooms())  # Volume of a single voxel in mm^3
#     num_channels = array.shape[0]
#     names = target_regions.names
#     volumes = {}
#     for channel in range(num_channels):
#         total_volume = np.sum(array[channel] == 1) * voxel_volume  # Total volume of target voxels
#         volumes[names[channel]] = total_volume

#     return volumes

@function_tool
async def get_brain_region_volumes(wrapper: RunContextWrapper[SessionContext], patient_name: str, target_regions: TargetRegions) -> dict[str, float]:
    """
    The SynthSeg tool segmenting the anatomical brain regions outputs a CSV file with volumes of all regions in mm^3.
    Use this tool to read the volumes of desired target regions from that CSV file. This allows bypassing the need to load the full segmentation outputs.
    """
    patient: Patient = wrapper.context.get(patient_name)
    df = pd.read_csv(patient.brain_region_volumes)
    names = target_regions.names
    names = [name.lower() for name in names if name.lower() in df.columns]
    volumes = df.loc[0, names].to_dict()

    return volumes

# @function_tool
# async def get_tumor_volumes(wrapper: RunContextWrapper[SessionContext], image_name: str) -> dict[str, float]:
#     """
#     Brain tumor segmentation algorithms only output a segmentation mask.
#     This tool computes the cumulative volumes in mm^3 for all known tumor subclasses from the vocabulary.
#     """

#     if image_name not in wrapper.context:
#         raise ValueError(f"Segmentation file '{image_name}' is not loaded.")

#     seg = wrapper.context.get(image_name)
#     array = seg.get_fdata()
#     voxel_volume = np.prod(seg.header.get_zooms())  # Volume of a single voxel in mm^3
#     names_and_labels = json.load(open("./data/brats_classes.json"))

#     volumes = {}
#     for name, labels in names_and_labels.items():
#         # Compute the volume for each label
#         label_volumes = np.isin(array, labels).sum() * voxel_volume
#         volumes[name] = label_volumes

#     return volumes


@function_tool
async def get_tumor_measurements_per_lesion(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> Union[dict[str, float], str]:
    """
    Compute per-lesion measurements for gliomas and meningiomas using the segmentation and instance maps from the context.
    Lesions (whole tumors) and tumor cores with volume < 10 mm3 are ignored and flagged.
    Returns a dictionary of lesions -> measurements including:
    - whole_tumor_volume (mm^3)
    - edema_volume (mm^3)
    - gross_tumor_core_volume (mm^3)
    - bbox and centroid for tumor cores
    - enhancing and non-enhancing volumes (mm^3) for tumor cores
    Saves a JSON file '{image_name}_tumor_measurements.json' and returns the dict.
    """
    try:
        patient: Patient = wrapper.context.get(patient_name)
        if "seg" not in patient.loaded_images:
            raise ValueError(f"Segmentation file is not loaded as 'seg' in patient context for '{patient_name}'.")
        seg = patient.loaded_images["seg"]
        array = seg.get_fdata()
        spacing = seg.header.get_zooms()[:3]
        voxel_volume = np.prod(spacing)  # Volume of a single voxel in mm^3

        if "lesion_map" not in patient.loaded_images:
            raise ValueError(f"Lesion map is not loaded in patient context for '{patient_name}'.")
        lesion_map = patient.loaded_images["lesion_map"]
        lesion_map = lesion_map.get_fdata()

        # instance_ids = np.unique(lesion_map).tolist()
        measurements = {}

        cavity_vol = np.sum(array == 4) * voxel_volume
        if cavity_vol > 0:
            measurements["resection_cavity_volume"] = cavity_vol # need to match this to instances?

        ### TODO: need to remove small objects from lesion_map?
        lesion_map = remove_small_objects(lesion_map.astype(np.int16), min_size=10)
        props_wt = regionprops(lesion_map, spacing=spacing)
        
        
        for region in props_wt:
            instance_id = region.label
            wt_array = np.where(lesion_map == instance_id, array, 0)
            wt_vol = region.area
            edema_vol = (wt_array == 2).sum() * voxel_volume
            measurements[f"lesion_{instance_id}"] = {
                "whole_tumor_volume": wt_vol,
                "edema_volume": edema_vol,
                "gross_tumor_core_volume": (wt_vol - edema_vol),
                "bbox": region.bbox,
            }

            tc_array = np.where(np.isin(wt_array, [1, 3]), wt_array, 0)
            tc_instances = label(tc_array.astype(bool),connectivity=3)
            props_tc = regionprops(tc_instances, spacing=spacing)

            for tumor_region in props_tc:
                tc_id = tumor_region.label
                tc_vol = tumor_region.area
                centroid = tuple(map(int, tumor_region.centroid))
                if tc_vol < 10:
                    measurements[f"lesion_{instance_id}"][f"tumor_core_{tc_id}"] = {
                        "centroid": centroid,
                        "volume": "smaller_than_10",
                    }
                    continue  # skip very small lesions
                tc_masked = np.where(tc_instances == tc_id, tc_array, 0)
                et_vol = (tc_masked == 3).sum() * voxel_volume
                netc_vol = (tc_masked == 1).sum() * voxel_volume
                tc_max_diameter = round(tumor_region.feret_diameter_max, 2) # Maximum caliper distance
                measurements[f"lesion_{instance_id}"][f"tumor_core_{tc_id}"] = {
                    "volume": tc_vol,
                    "centroid": centroid,
                    "max_diameter": tc_max_diameter,
                    "bbox": tumor_region.bbox,
                    "non_enhancing_volume": netc_vol,
                    "enhancing_volume": et_vol,
                }
        out_file = wrapper.context.get("output_dir") / patient_name / f"{patient.name}_measurements.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        json.dump(measurements, open(out_file, "w"), indent=4)
        patient.measurements = str(out_file)
        wrapper.context.set(patient_name, patient)

        return measurements
    except Exception as e:
        print(f"Error in get_tumor_measurements_per_lesion for patient '{patient_name}': {e}")
        return e


@function_tool
async def get_metastasis_measurements_per_lesion(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> Union[dict[str, float], str]:
    """
    Compute per-lesion measurements for metastases using the segmentation and instance maps from the context.
    Lesions (tumor cores) with volume < 10 mm3 are ignored and flagged.
    Returns a dictionary of lesions -> measurements including:
    - total_edema_volume (mm^3)
    - gross_tumor_core_volume (mm^3)
    - bbox and centroid for tumor cores
    - enhancing and non-enhancing volumes (mm^3) for tumor cores
    Saves a JSON file '{image_name}_measurements.json' and returns the dict.
    """

    patient: Patient = wrapper.context.get(patient_name)
    if "seg" not in patient.loaded_images:
        raise ValueError(f"Segmentation file is not loaded as 'seg' in patient context for '{patient_name}'.")
    seg = patient.loaded_images["seg"]
    array = seg.get_fdata()
    spacing = seg.header.get_zooms()[:3]
    voxel_volume = np.prod(spacing)  # Volume of a single voxel in mm^3

    if "lesion_map" not in patient.loaded_images:
        raise ValueError(f"Lesion map is not loaded in patient context for '{patient_name}'.")
    lesion_map = patient.loaded_images["lesion_map"]
    lesion_map = lesion_map.get_fdata()

    # instance_ids = np.unique(lesion_map).tolist()
    # lesion_map = remove_small_objects(lesion_map.astype(np.int16), min_size=10)
    props = regionprops(lesion_map.astype(np.int16), spacing=spacing)

    edema = label(array == 2,connectivity=3)
    edema_props = regionprops(edema.astype(np.int16), spacing=spacing)
    measurements = {"disconnected_edema_volumes": [prop.area for prop in edema_props if prop.area >= 10]}  # list of edema volumes for disconnected edema regions
    wt = np.isin(array, [1,2,3])
    wt = label(wt,connectivity=3)
    wt_props = regionprops(wt.astype(np.int16), spacing=spacing)
    measurements["whole_tumor_volumes"] = []
    measurements['whole_tumor_bounding_boxes'] = []
    for prop in wt_props:
        if prop.area < 10:
            continue
        measurements["whole_tumor_volumes"].append(prop.area)
        measurements['whole_tumor_bounding_boxes'].append(prop.bbox)

    # measurements["whole_tumor_volumes"] = [prop.area for prop in wt_props if prop.area >=10]  # list of whole tumor volumes for disconnected whole tumor regions
    
    cavity_vol = np.sum(array == 4) * voxel_volume
    if cavity_vol > 0:
        measurements["resection_cavity_volume"] = cavity_vol

    for tumor_region in props:
        instance_id = tumor_region.label
        tc_array = np.where(lesion_map == instance_id, array, 0)
        centroid = tuple(map(int, tumor_region.centroid))
        tc_vol = tumor_region.area
        if tc_vol < 10:
            measurements[f"lesion_{instance_id}"] = {
            'tumor_core_1': {  # only one tumor core, for compatibility with other tools
                "centroid": centroid,
                "volume": "smaller_than_10",
                }
            }
            continue  # skip very small lesions
        et_vol = (tc_array == 3).sum() * voxel_volume
        netc_vol = (tc_array == 1).sum() * voxel_volume
        tc_max_diameter = round(tumor_region.feret_diameter_max, 2)  # Maximum caliper distance
        measurements[f"lesion_{instance_id}"] = {
            'tumor_core_1': {  # only one tumor core, for compatibility with other tools
                "bbox": tumor_region.bbox,
                "centroid": centroid,
                "volume": tc_vol,
                "max_diameter": tc_max_diameter,
                "non_enhancing_volume": netc_vol,
                "enhancing_volume": et_vol,
            }
        }
    out_file = wrapper.context.get("output_dir") / patient_name / f"{patient.name}_measurements.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    json.dump(measurements, open(out_file, "w"), indent=4)
    patient.measurements = str(out_file)
    wrapper.context.set(patient_name, patient)

    return measurements

@function_tool
async def get_tumor_locations(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> dict[str, float]:
    """
    After computing tumor measurements per lesion, this tool:
    Reads the tumor measurements per lesion from the previously computed JSON file.
    Matches the tumor core centroids to brain lobes based on their coordinates.
    Uses a predefined brain lobe atlas.
    """
    patient: Patient = wrapper.context.get(patient_name)
    if not Path(patient.measurements).exists():
        print(f"Measurements file '{patient.measurements}' not found.")
        # measurements = await get_glioma_measurements_per_lesion(wrapper, image_name, f"{image_name}_lesion_map")
    else:
        measurements = json.load(open(patient.measurements))

    atlas = nib.load("./data/BCI-DNI-sri24.nii.gz").get_fdata()
    atlas_labels = json.load(open("./data/brain_lobes.json"))
    atlas_labels = {v: k for k, v in atlas_labels.items()}  # invert dict

    locations = {}
    for instance, props in measurements.items():
        if not isinstance(props, dict):
            continue
        locations[instance] = {}
        for key, subprops in props.items():
            if key.startswith("tumor_core_"):
                if subprops["volume"] == "smaller_than_10":
                    continue
                x, y, z = subprops["centroid"]
                atlas_label = int(atlas[x, y, z])
                lobe_name = atlas_labels.get(atlas_label, "unknown")
                measurements[instance][key]["location"] = lobe_name
                locations[instance][key] = lobe_name


    json.dump(measurements, open(patient.measurements, "w"), indent=4)
    wrapper.context.set(patient_name, patient)

    return locations


@function_tool
async def read_rano_criteria(disease: Literal['metastasis','glioma']) -> str:
    """
    Retrieves RANO criteria from the file for evaluating brain tumor treatment response and longitudinal progression. 
    
    Criteria are based on post-segmentation volumetric measurements. 
    
    Required Workflow:
    1. Perform tumor segmentation on both timepoints first.
    2. Call this tool to retrieve the disease-specific RANO criteria.
    3. Do lesion matching, extract metrics and compare baseline vs. follow-up scans based on these criteria.
    """
    if disease == 'metastasis':
        return NotImplementedError("RANO criteria for metastasis are not yet implemented.")
    elif disease == 'glioma':
        rano = open('./data/rano-glioma.txt', 'r').read()

    return rano