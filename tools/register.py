import ants
import shutil

from pathlib import Path
from typing import List, Optional, Union, Dict, Literal

from agents import function_tool, RunContextWrapper, input_guardrail, GuardrailFunctionOutput

import nibabel as nib
import numpy as np

from util.data_types import Patient, TargetRegions
from util.context_manager import SessionContext


def check_if_registered_no_tool(patient: Patient, disease: str) -> str:
    """
    Check if the patient's scans are co-registered by comparing their shapes and affines.
    The reference is the t1 scan.
    Then compare t1 to MNI152 and SRI24 atlases to check if t1 is registered to either atlas space.
    """

    # skip everything if patient only has t1c -- dura and mets post-op use raw t1c only
    if "placeholder" in patient.t1.lower() and "placeholder" not in patient.t1c.lower() and "placeholder" in patient.t2.lower() and "placeholder" in patient.flair.lower():
        return f"Patient only has t1c scan. This is fine if we are doing dura or metastasis post-op segmentation, no need for registration."

    if "placeholder" in patient.t1.lower():
        raise ValueError("The t1 scan is required for checking registration.")
    
    t1: nib.Nifti1Image = nib.load(patient.t1)

    unregistered = []
    for field in ['t2', 'flair', 't1c']:
        pth = getattr(patient, field)
        if "placeholder" in pth.lower():
            continue
        image: nib.Nifti1Image = nib.load(pth)
        if not (image.shape[:3] == t1.shape[:3] and np.allclose(image.affine, t1.affine)):
            unregistered.append(field)
    
    # compare t1 to atlases
    sri: nib.Nifti1Image = nib.load(Path.cwd() / "data" / "brats_sri24.nii")
    mni: nib.Nifti1Image = nib.load(Path.cwd() / "data" / "brats_MNI152lin_T1_1mm.nii.gz")

    if disease.lower() == "mets_preop":
        sri = nib.load(Path.cwd() / "data" / "bratsmets_sri24.nii.gz")

    atlas_registered = []
    if (t1.shape[:3] == sri.shape[:3] and np.allclose(t1.affine, sri.affine) and np.allclose(nib.affines.voxel_sizes(t1.affine), nib.affines.voxel_sizes(sri.affine))):
        atlas_registered.append("SRI24")
    if (t1.shape[:3] == mni.shape[:3] and np.allclose(t1.affine, mni.affine) and np.allclose(nib.affines.voxel_sizes(t1.affine), nib.affines.voxel_sizes(mni.affine))):
        atlas_registered.append("MNI152")

    if not unregistered:
        if atlas_registered:
            return f"All MRI scans are co-registered to t1. Additionally, t1 is registered to atlas space: {atlas_registered[0]}."
        else:
            return f"All MRI scans are co-registered to t1, but t1 is not registered to any atlas space."
    else:
        if atlas_registered:
            return f"MRI scans {', '.join(unregistered)} are not co-registered to t1. But, t1 is registered to atlas space: {atlas_registered[0]}."
        else:
            return f"MRI scans {', '.join(unregistered)} are not co-registered to t1, and t1 is not registered to any atlas space."


### DOESN'T WORK -- LLM never sees the guardrail output ### 
@input_guardrail
async def check_if_registered_guardrail(wrapper: RunContextWrapper[SessionContext], agent, agent_input: str) -> str:
    """
    Check if the patient's scans are co-registered by comparing their shapes and affines.
    The reference is the t1 scan.
    Then compare t1 to MNI152 and SRI24 atlases to check if t1 is registered to either atlas space.
    """
    out_messages = []
    for patient_name in wrapper.context.list_keys():
        patient: Patient = wrapper.context.get(patient_name)
        message = check_if_registered_no_tool(patient, disease=patient.disease)
        out_messages.append(f"{patient_name}: {message}")
    
    out_messages = "\n".join(out_messages)
    out_messages = f"Registration check results for all patients in the context:\n{out_messages}"
    
    return GuardrailFunctionOutput(output_info=out_messages, tripwire_triggered=False)

@function_tool
async def check_if_registered(wrapper: RunContextWrapper[SessionContext], patient_name: str, disease: Literal['mets_preop','glioma_preop','glioma_postop','meningioma']) -> str:
    """
    Check if the patient's scans are co-registered by comparing their shapes and affines.
    The reference is the t1 scan.
    Then compare t1 to MNI152 and SRI24 atlases to check if t1 is registered to either atlas space.
    Feed the disease type with the 'disease' parameter. Options: ['mets_preop','glioma_preop','glioma_postop','meningioma']
    """
    patient: Patient = wrapper.context.get(patient_name)
    # skip everything if patient only has t1c -- dura and mets post-op use raw t1c only
    if "placeholder" in patient.t1.lower() and "placeholder" not in patient.t1c.lower() and "placeholder" in patient.t2.lower() and "placeholder" in patient.flair.lower():
        return f"Patient {patient_name} only has t1c scan. No need to check for registration."

    if "placeholder" in patient.t1.lower():
        raise ValueError("The t1 scan is required for checking registration.")
    
    t1: nib.Nifti1Image = nib.load(patient.t1)

    unregistered = []
    for field in ['t2', 'flair', 't1c']:
        pth = getattr(patient, field)
        if "placeholder" in pth.lower():
            continue
        image: nib.Nifti1Image = nib.load(pth)
        if not (image.shape[:3] == t1.shape[:3] and np.allclose(image.affine, t1.affine)):
            unregistered.append(field)
    
    # compare t1 to atlases
    sri: nib.Nifti1Image = nib.load(Path.cwd() / "data" / "brats_sri24.nii")
    mni: nib.Nifti1Image = nib.load(Path.cwd() / "data" / "brats_MNI152lin_T1_1mm.nii.gz")

    if disease.lower() == "mets_preop":
        sri = nib.load(Path.cwd() / "data" / "bratsmets_sri24.nii.gz")

    atlas_registered = []
    if (t1.shape[:3] == sri.shape[:3] and np.allclose(t1.affine, sri.affine) and np.allclose(nib.affines.voxel_sizes(t1.affine), nib.affines.voxel_sizes(sri.affine))):
        atlas_registered.append("SRI24")
    if (t1.shape[:3] == mni.shape[:3] and np.allclose(t1.affine, mni.affine) and np.allclose(nib.affines.voxel_sizes(t1.affine), nib.affines.voxel_sizes(mni.affine))):
        atlas_registered.append("MNI152")
        
    if not unregistered:
        if atlas_registered:
            return f"All scans are co-registered to t1 for patient {patient.name}. Additionally, t1 is registered to atlas space: {atlas_registered[0]}."
        else:
            return f"All scans are co-registered to t1 for patient {patient.name}, but t1 is not registered to any atlas space."
    else:
        if atlas_registered:
            return f"Scans {', '.join(unregistered)} are not co-registered to t1 for patient {patient.name}. But, t1 is registered to atlas space: {atlas_registered[0]}."
        else:
            return f"Scans {', '.join(unregistered)} are not co-registered to t1 for patient {patient.name}, and t1 is not registered to any atlas space."


def registration(source_path: str, target_path: str = None, skullstripped: bool = True, space: str = "SRI24", transform: str = "Rigid",):
    """
    Register the source image to the SRI24 or MNI152 atlas using ANTs.
    """
    if space is None:
        if target_path is None:
            raise ValueError("Either provide a valid space ('SRI24' or 'MNI152') or a target_path for registration.")
    elif "sri" in space.lower():
        if skullstripped:
            target_path = Path.cwd() / "data" / "brats_sri24_skullstripped.nii"
        else:
            target_path = Path.cwd() / "data" / "brats_sri24.nii"
        if "ras" in space.lower():
            target_path = Path(str(target_path).replace(".nii", ".nii.gz").replace('brats','bratsmets'))
    elif "mni" in space.lower():
        if skullstripped:
            raise ValueError("Skull-stripped MNI152 atlas is not available.")
        else:
            target_path = Path.cwd() / "data" / "brats_MNI152lin_T1_1mm.nii.gz"
    
    source = ants.image_read(source_path)
    target = ants.image_read(str(target_path))
    # if space is not None and "ras" in space.lower():
    #     target_path.unlink()

    # source_norm = ants.iMath(source, "Normalize")
    # target_norm = ants.iMath(target, "Normalize")
    
    reg = ants.registration(
        fixed=target,
        moving=source,
        type_of_transform=transform,
    )

    return reg, target

def register_one_image(source_path: str, target_path: str = None, path_to_register: str = None, write_path: str = None, skullstripped: bool = True, is_label: bool = False, transform: str = "Rigid", space: str = "SRI24") -> str:

    reg, target = registration(source_path, target_path=target_path, skullstripped=skullstripped, transform=transform, space=space)

    if path_to_register is not None: ## use source as registration reference, but use on different image than source. e.g., label maps
        source_path = path_to_register
        out = ants.apply_transforms(
            fixed=target,
            moving=ants.image_read(source_path),
            transformlist=reg['fwdtransforms'],
            interpolator='nearestNeighbor' if is_label else 'linear'
        )
    else:
        out = reg['warpedmovout']
    
    if write_path is not None:
        output_path = write_path
    elif space is not None:
        if "-ras" in space.lower():
            space  = space.replace("-ras", "")
        output_path = source_path.replace(".nii.gz", f"_{space.lower()}.nii.gz")
    else:
        output_path = source_path.replace(".nii.gz", f"_t1.nii.gz")
    ants.image_write(out, output_path)

    return output_path



@function_tool
async def co_register_patient(wrapper: RunContextWrapper[SessionContext], patient_name: str, skullstripped: bool = True) -> str:
    """
    Co-register all MRI scans of a patient to the t1 scan.
    """
    print(f"Co-registering patient {patient_name} to t1 space with skullstripped={skullstripped}")

    patient: Patient = wrapper.context.get(patient_name)
    if "placeholder" in patient.t1.lower():
        raise ValueError("The t1 scan is required for co-registration.")
    registered = []

    reg_pth = wrapper.context.get("output_dir") / patient.name / 'preprocessed' 
    reg_pth.mkdir(exist_ok=True, parents=True)

    for field in ['t2', 'flair', 't1c']:
        pth = getattr(patient, field)
        if "placeholder" in pth.lower():
            continue
        out_path = str(reg_pth / Path(pth).name)
        out_path = register_one_image(
            target_path=patient.t1,
            source_path=pth,
            write_path=out_path,
            skullstripped=skullstripped,
            space=None
        )
        patient.__setattr__(field, out_path)
        registered.append(field)

    wrapper.context.set(patient.name, patient)

    return f"Scans {', '.join(registered)} co-registered to t1 space for patient {patient.name}."


def co_register_patient_no_tool(patient: Patient, skullstripped: bool = True) -> str:
    """
    Co-register all MRI scans of a patient to the t1 scan.
    """
    if "placeholder" in patient.t1.lower():
        raise ValueError("The t1 scan is required for co-registration.")
    registered = []

    reg_pth = Path.cwd() / 'outputs' / patient.name / 'preprocessed' 
    reg_pth.mkdir(exist_ok=True, parents=True)

    shutil.copy(patient.t1, str(reg_pth / Path(patient.t1).name))
    patient.t1 = str(reg_pth / Path(patient.t1).name)

    for field in ['t2', 'flair', 't1c']:
        pth = getattr(patient, field)
        if "placeholder" in pth.lower():
            continue
        out_path = register_one_image(
            target_path=patient.t1,
            source_path=pth,
            skullstripped=skullstripped,
            space=None
        )
        out_path = str(reg_pth / Path(pth).name)
        patient.__setattr__(field, out_path)
        registered.append(field)

    return f"Scans {', '.join(registered)} co-registered to t1 space for patient {patient.name}."


@function_tool
async def atlas_register_patient(wrapper: RunContextWrapper[SessionContext], patient_name: str, skullstripped: bool = True, space: Literal['SRI24', 'MNI152'] = "SRI24", disease: Literal['mets_preop','glioma_preop','glioma_postop','meningioma'] = "mets_preop") -> str:
    """
    Register all input MRI scans of a patient to the specified atlas space. Supported spaces are "SRI24" and "MNI152".
    This tool can be used before segmentation tools, depending on the segmentation tool input requirements.
    Needs other MRIs to be co-registered to t1 space first.
    Feed the disease type with the 'disease' parameter. Options: ['mets_preop','glioma_preop','glioma_postop','meningioma']
    """
    print(f"Registering patient {patient_name} to {space} space with skullstripped={skullstripped}")

    patient: Patient = wrapper.context.get(patient_name)
    if "placeholder" in patient.t1.lower():
        raise ValueError("The t1 scan is required for registration.")
    if disease == "mets_preop":
        space += "-ras"
    reg, atlas = registration(patient.t1, skullstripped=skullstripped, space=space)
    t1 = reg['warpedmovout']
    reg_pth = wrapper.context.get("output_dir") / patient.name / 'preprocessed'  
    reg_pth.mkdir(exist_ok=True, parents=True)
    out_path = str(reg_pth / Path(patient.t1).name.replace(".nii.gz", "_sri.nii.gz"))
    ants.image_write(t1, out_path)
    patient.t1 = out_path

    registered = ['t1']

    for field in ['t2', 'flair', 't1c']:
        pth = getattr(patient, field)
        if "placeholder" in pth.lower():
            continue
        out = ants.apply_transforms(
            fixed=atlas,
            moving=ants.image_read(pth),
            transformlist=reg['fwdtransforms'],
            interpolator='linear',
        )
        out_path = str(reg_pth / Path(pth).name.replace(".nii.gz", "_sri.nii.gz"))
        patient.__setattr__(field, out_path)
        ants.image_write(out, out_path)
        registered.append(field)

    # for field in ['seg', 'dura', 'brain_regions']:
    #     pth = getattr(patient, field)
    #     if pth is not None:
    #         out = ants.apply_transforms(
    #             fixed=atlas,
    #             moving=ants.image_read(pth),
    #             transformlist=reg['fwdtransforms'],
    #             interpolator='nearestNeighbor'
    #         )
    #         out_path = pth.replace(".nii.gz", "_sri.nii.gz")
    #         patient.__setattr__(field, out_path)
    #         ants.image_write(out, out_path)
    #         registered.append(field)

    wrapper.context.set(patient.name, patient)

    return f"Scans {', '.join(registered)} registered to {space} space for patient {patient.name}."


def atlas_register_patient_no_tool(patient: Patient, skullstripped: bool = True, space: str = "SRI24") -> str:
    """
    Register all input MRI scans of a patient to the specified atlas space. Supported spaces are "SRI24" and "MNI152".
    This tool can be used before segmentation tools, depending on the segmentation tool input requirements.
    Needs other MRIs to be co-registered to t1 space first.
    """
    # patient: Patient = wrapper.context.get(patient_name)
    if "placeholder" in patient.t1.lower():
        raise ValueError("The t1 scan is required for registration.")
    reg, atlas = registration(patient.t1, skullstripped=skullstripped, space=space)
    t1 = reg['warpedmovout']
    reg_pth = Path.cwd() / 'outputs' / patient.name / 'preprocessed'  
    reg_pth.mkdir(exist_ok=True, parents=True)
    out_path = str(reg_pth / Path(patient.t1).name.replace(".nii.gz", "_sri.nii.gz"))
    ants.image_write(t1, out_path)
    patient.t1 = out_path

    registered = ['t1']

    for field in ['t2', 'flair', 't1c']:
        pth = getattr(patient, field)
        if "placeholder" in pth.lower():
            continue
        out = ants.apply_transforms(
            fixed=atlas,
            moving=ants.image_read(pth),
            transformlist=reg['fwdtransforms'],
            interpolator='linear'
        )
        out_path = str(reg_pth / Path(pth).name.replace(".nii.gz", "_sri.nii.gz"))
        patient.__setattr__(field, out_path)
        ants.image_write(out, out_path)
        registered.append(field)

    return f"Scans {', '.join(registered)} registered to {space} space for patient {patient.name}."


@function_tool
async def atlas_register_patient_outputs(wrapper: RunContextWrapper[SessionContext], patient_name: str, skullstripped: bool = True, space: str = "SRI24") -> str:
    """
    Register all output segmentation maps of a patient to the specified atlas space. Supported spaces are "SRI24" and "MNI152".
    """
    patient: Patient = wrapper.context.get(patient_name)
    if "placeholder" in patient.t1.lower():
        src = patient.t1c
        used = "t1c"
    else:
        src = patient.t1
        used = "t1"
    reg, atlas = registration(src, skullstripped=skullstripped, space=space)
    img = reg['warpedmovout']
    reg_pth = wrapper.context.get("output_dir") / patient.name 
    reg_pth.mkdir(exist_ok=True, parents=True)
    out_path = str(reg_pth / Path(src).name.replace(".nii.gz", "_sri.nii.gz"))
    ants.image_write(img, out_path)
    patient[used] = out_path

    registered = []

    for field in ['seg', 'dura', 'brain_regions']:
        pth = getattr(patient, field)
        if pth is not None:
            out = ants.apply_transforms(
                fixed=atlas,
                moving=ants.image_read(pth),
                transformlist=reg['fwdtransforms'],
                interpolator='nearestNeighbor'
            )
            
            out_path = str(reg_pth / Path(pth).name.replace(".nii.gz", "_sri.nii.gz"))
            patient.__setattr__(field, out_path)
            ants.image_write(out, out_path)
            registered.append(field)

    wrapper.context.set(patient.name, patient)

    return f"Scans {', '.join(registered)} registered to {space} space for patient {patient.name}."