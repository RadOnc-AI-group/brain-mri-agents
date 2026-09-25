import os
import sys
import subprocess

from loguru import logger
from pathlib import Path
from typing import List, Optional, Union, Dict

from agents import function_tool, RunContextWrapper

from brats import AdultGliomaPreTreatmentSegmenter, AdultGliomaPostTreatmentSegmenter, MetastasesSegmenter, MeningiomaSegmenter, PediatricSegmenter
from brats.constants import AdultGliomaPostTreatmentAlgorithms

from util.data_types import PatientInput, TargetRegions, Patient
from util.context_manager import SessionContext
from tools.register import register_one_image


# logger.level("ERROR")

# silence the brats loggers
logger.remove()
logger.add(
    sys.stderr,
    level="ERROR",
)

CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "all")


@function_tool
async def segment_glioma_preop(wrapper: RunContextWrapper[SessionContext], patient_name: str) -> str: 
# async def segment_glioma_preop(patient: Patient) -> Patient:
    """
    Segment glioma from pre-operative MRI scans (t1, t2, flair, t1C). Writes the result as a NIfTI file.
    Unless the user specifies otherwise, scans are assumed to be pre-op and this function should be used.
    This tool works in the SRI24 space with skullstripped images.
    """
    # print("Glioma Pre-op Segmentation tool is called.")
    if CUDA_DEVICES == "all":
        device = "1"
    else:
        device = CUDA_DEVICES.split(",")[0]  # use the first specified GPU for segmentation

    segmenter = AdultGliomaPreTreatmentSegmenter(cuda_devices=device)
    patient = wrapper.context.get(patient_name)

    out_dir = wrapper.context.get("output_dir") / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_file = str(out_dir / f"{patient.name}_glioma_seg_pre.nii.gz")
    
    segmenter.infer_single(
        t1n=patient.t1,
        t2w=patient.t2,
        t1c=patient.t1c,
        t2f=patient.flair,
        output_file=seg_file
    )

    # try:
    # seg = patient.t1.replace("t1n","seg-mav") ## debugging for now, remove later
    # shutil.copy(seg, seg_file)
    # except: 
    #     seg = patient.t1.replace("t1","seg") ## debugging for now, remove later
    #     shutil.copy(seg, seg_file)

    # wrapper.context[patient_name]['filenames'].seg = seg_file
    patient.seg = seg_file
    wrapper.context.set(patient_name, patient)

    return f"Segmentation completed. Segmentation file saved at: '{seg_file}'. Path stored in patient context under 'seg' attribute."



@function_tool
async def segment_glioma_postop(wrapper: RunContextWrapper[SessionContext],patient_name: str) -> str:
# async def segment_glioma_postop(patient: Patient) -> Patient:
    """
    Segment glioma from post-operative MRI scans (t1, t2, flair, t1c). Writes the result as a NIfTI file.
    The user should specify that the scans are post-op.
    This tool works in the MNI152 space with skullstripped images.
    """
    # print("Glioma Post-op Segmentation tool is called.")
    if CUDA_DEVICES == "all":
        device = "1"
    else:
        device = CUDA_DEVICES.split(",")[0]  # use the first specified GPU for segmentation
    
    segmenter = AdultGliomaPostTreatmentSegmenter(cuda_devices=device)
    patient = wrapper.context.get(patient_name)

    out_dir = wrapper.context.get("output_dir") / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_file = str(out_dir / f"{patient.name}_glioma_seg_post.nii.gz")

    segmenter.infer_single(
        t1n=patient.t1,
        t2w=patient.t2,
        t1c=patient.t1c,
        t2f=patient.flair,
        output_file=seg_file
    )
    # wrapper.context[patient_name]['filenames'].seg = seg_file
    patient.seg = seg_file
    wrapper.context.set(patient_name, patient)

    return f"Segmentation completed. Segmentation file saved at: '{seg_file}'. Path stored in patient context under 'seg' attribute."


@function_tool
async def segment_brain_metastases_preop(wrapper: RunContextWrapper[SessionContext],patient_name: str) -> str:
# async def segment_brain_metastases_preop(patient: Patient) -> Patient:
    """
    Segment brain metastases from pre-operative MRI scans (t1, t2, flair, t1c). Writes the result as a NIfTI file.
    This tool works in the SRI24 space with skullstripped images.
    """
    if CUDA_DEVICES == "all":
        device = "1"
    else:
        device = CUDA_DEVICES.split(",")[0]  # use the first specified GPU for segmentation
    

    patient = wrapper.context.get(patient_name)

    out_dir = wrapper.context.get("output_dir") / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_file = str(out_dir / f"{patient.name}_metastases_seg_pre.nii.gz")
    # print("Brain Metastases Pre-op Segmentation tool is called.")
    segmenter = MetastasesSegmenter(cuda_devices=device)

    segmenter.infer_single(
        t1n=patient.t1,
        t2w=patient.t2,
        t1c=patient.t1c,
        t2f=patient.flair,
        output_file=seg_file
    )

    # wrapper.context[patient_name]['filenames'].seg = seg_file
    patient.seg = seg_file
    wrapper.context.set(patient_name, patient)

    return f"Segmentation completed. Segmentation file saved at: '{seg_file}'. Path stored in patient context under 'seg' attribute."


@function_tool
async def segment_meningioma(wrapper: RunContextWrapper[SessionContext],patient_name: str) -> str:
# async def segment_meningioma(patient: Patient) -> Patient:
    """
    Segment meningioma from pre-therapy MRI scans (t1, t2, flair, t1c). Writes the result as a NIfTI file.
    This tool works in the SRI24 space with skullstripped images.
    """
    if CUDA_DEVICES == "all":
        device = "1"
    else:
        device = CUDA_DEVICES.split(",")[0]  # use the first specified GPU for segmentation
    

    patient = wrapper.context.get(patient_name)

    out_dir = wrapper.context.get("output_dir") / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_file = str(out_dir / f"{patient.name}_meningioma_seg.nii.gz")
    # print("Meningioma Segmentation tool is called.")
    segmenter = MeningiomaSegmenter(cuda_devices=device)

    segmenter.infer_single(
        t1n=patient.t1,
        t2w=patient.t2,
        t1c=patient.t1c,
        t2f=patient.flair,
        output_file=seg_file
    )
    # wrapper.context[patient_name]['filenames'].seg = seg_file
    patient.seg = seg_file
    wrapper.context.set(patient_name, patient)

    return f"Segmentation completed. Segmentation file saved at: '{seg_file}'. Path stored in patient context under 'seg' attribute."


@function_tool
async def segment_pediatric_brain_tumor(wrapper: RunContextWrapper[SessionContext],patient_name: str) -> str:
# async def segment_pediatric_brain_tumor(patient: Patient) -> Patient:
    """
    Segment pediatric brain tumors from pre-therapy MRI scans (t1, t2, flair, t1c). Writes the result as a NIfTI file.
    This tool works in the SRI24 space with skullstripped images.
    """
    if CUDA_DEVICES == "all":
        device = "1"
    else:
        device = CUDA_DEVICES.split(",")[0]  # use the first specified GPU for segmentation
    

    patient = wrapper.context.get(patient_name)
    out_dir = wrapper.context.get("output_dir") / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_file = str(out_dir / f"{patient.name}_pediatric_tumor_seg.nii.gz")
    # print("Pediatric Brain Tumor Segmentation tool is called.")
    segmenter = PediatricSegmenter(cuda_devices=device)

    segmenter.infer_single(
        t1n=patient.t1,
        t2w=patient.t2,
        t1c=patient.t1c,
        t2f=patient.flair,
        output_file=seg_file
    )
    # wrapper.context[patient_name]['filenames'].seg = seg_file

    patient.seg = seg_file
    wrapper.context.set(patient_name, patient)

    return f"Segmentation completed. Segmentation file saved at: '{seg_file}'. Path stored in patient context under 'seg' attribute."

def run_synthseg(input_path: str, output_path: str, csv_path: Optional[str] = None) -> None:
    """
    Run the SynthSeg segmentation command line tool.
    
    Args:
        path (str): Path to the input MRI scan.
        output_path (str): Path to save the output segmentation.
    """
    input_path = Path(input_path).resolve().expanduser()
    output_path = Path(output_path).resolve().expanduser()
    output_dir = output_path.parent
    output_file = output_path.name

    if CUDA_DEVICES == "all":
        gpu_flag = ["--gpus", "1"]
    else:
        # Docker requires the format "device=0,1"
        gpu_flag = ["--gpus", f'"device={CUDA_DEVICES}"']
    
    uid = os.getuid()
    gid = os.getgid()

    command = [
        "docker", "run", *gpu_flag, "--rm",
        "--user", f"{uid}:{gid}",
        "-v", f"{input_path}:/input/image.nii.gz",
        "-v", f"{output_dir}:/output",
        "synthseg-robust-new",
        "--i", "/input/image.nii.gz",
        "--o", f"/output/{output_file}",
    ]

    if csv_path:
        csv_path = Path(csv_path).resolve().expanduser().name # ensure csv_path is just the filename
        command.extend(["--vol", f"/output/{csv_path}"])
    
    subprocess.run(command, check=True)


@function_tool
async def segment_brain_regions(wrapper: RunContextWrapper[SessionContext],patient_name: str, image_to_select: str = 't1') -> tuple[str, str]:
# async def segment_brain_regions(patient: Patient, image_to_select: str = 't1') -> Patient:
    """
    Segment anatomical regions of the brain using SynthSeg. 
    It works for one scan only, the user should specify which modality to use in case of multiple paths.
    Use t1 if nothing specified, which is recommended for best results in general.
    This tool works in all image spaces (SRI24, MNI152, raw), as SynthSeg is robust to different spaces. So, prior registration is not required.
    """
    patient = wrapper.context.get(patient_name)
    modality = image_to_select.split('_')[0]  # e.g., 't1', 't2', 'flair', 't1c'
    # print(f"SynthSeg segmentation tool is called for modality: {modality}")
    path = getattr(patient, image_to_select)
    
    out_dir = wrapper.context.get("output_dir") / patient_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_file = str(out_dir / f"{patient.name}_{modality}_synthseg.nii.gz")
    csv_file = str(out_dir / f"{patient.name}_{modality}_synthseg_volumes.csv")

    
    run_synthseg(input_path=path, output_path=seg_file, csv_path=csv_file)

    # # print(f"Patient Input: {patient}")
    # # print(f"Image to select: {image_to_select}")

    # wrapper.context[patient_name]['filenames'].brain_regions = seg_file
    patient.brain_regions = seg_file
    patient.brain_region_volumes = csv_file
    wrapper.context.set(patient_name, patient)

    return f"Segmentation completed. Segmentation file saved at: '{seg_file}'. Path stored in patient context under 'brain_regions' attribute. And volumes CSV saved at: '{csv_file}', path stored under 'brain_region_volumes' attribute."
