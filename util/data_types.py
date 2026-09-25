
import json
import numpy as np
import nibabel as nib

from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal, Union
from agents import function_tool, RunContextWrapper, input_guardrail, GuardrailFunctionOutput

from util.context_manager import SessionContext

base = Path(__file__).parent.resolve()
BRATS_CLASSES = json.load(open(base / "../data/brats_classes.json"))
BRAIN_REGIONS = json.load(open(base / "../data/synthseg_classes.json"))


class SubAgentResponse(BaseModel):
    message: str = Field(..., description="A very short summary of the outcomes.")
    success: bool = Field(..., description="Whether the agent's operations were successful.")
    # used_tools: List[str] = Field(default_factory=list, description="List of tools used during this operation.")

class SegmentationResponse(SubAgentResponse):
    segmentation_paths: Optional[list[str]] = Field(default=None, description="Path to the generated segmentation file(s).")
    brain_region_volume_files: Optional[list[str]] = Field(default=None, description="Path to the CSV file(s) containing brain region volumes, if applicable.")

class PreprocessingConfig(BaseModel):
    patient_name: str
    perform_skullstripping: bool = Field(default=False, description="Whether to perform skullstripping. If False, accept the MRI scans as already skullstripped. If True, perform skullstripping as part of preprocessing.")
    perform_co_registration: bool = Field(default=False, description="Whether to perform co-registration of MRI scans to the T1 modality.")
    perform_atlas_registration: Literal[None,"MNI152", "SRI24"] = Field(default=None, description="Whether to perform registration to a standard atlas space, and which atlas to use. If None, no atlas registration will be performed.")

class PreprocessingRequest(BaseModel):
    request_configs: list[PreprocessingConfig] = Field(
        ..., 
        description="Preprocessing configuration for each patient"
    )

# class PreprocessingRequest(BaseModel):
#     patient_names: str = Field(..., description="Unique identifier for the patient in the context to preprocess")
#     skullstripping: bool = Field(default=False, description="Whether to perform skullstripping. If False, accept the MRI scans as already skullstripped.")
#     co_registration: bool = Field(default=False, description="Whether to perform co-registration of MRI scans to the T1 modality.")
#     atlas_registration: Optional[Literal["MNI152", "SRI24"]] = Field(default=None, description="Whether to perform registration to a standard atlas space, and which atlas to use. If None, no atlas registration will be performed.")

class SegmentationConfig(BaseModel):
    patient_name: str = Field(..., description="Unique identifier for the patient in the context to segment")
    target_type: Literal["glioma", "brain_metastases", "meningioma", "pediatric_brain_tumor", "dura", "brain_regions"] = Field(..., description="The type of target to segment, which determines the appropriate segmentation tool to use.")
    operation_status: Literal["pre-operative", "post-operative"] = Field(..., description="Whether the segmentation is for pre-operative or post-operative scans.")
    inputs_skullstripped: bool = Field(default=True, description="Whether the input scans are already skullstripped.")
    # target_structures: Optional[list[str]] = None  # e.g. ["tumor_core", "whole_tumor", "edema"] for gliomas, or specific brain regions

class SegmentationRequest(BaseModel):
    request_configs: list[SegmentationConfig] = Field(
        ..., 
        description="Segmentation configuration for each patient"
    )

def create_placeholder_image(
    lookup_path: str,
    target_path: Union[str, Path] = (Path.cwd() / "placeholder.nii.gz")
) -> None:
    """
    Create and save a placeholder image filled with zeros, in the same shape as the lookup image.
    """
    lookup = nib.load(lookup_path)
    placeholder = np.zeros(lookup.shape, dtype=np.float32)
    placeholder = nib.Nifti1Image(placeholder, lookup.affine, lookup.header)
    nib.save(placeholder, target_path)



class PatientInput(BaseModel):
    """
    PatientInput is a data class to represent the input for a patient.
    It includes the paths for the MRI scans, as given by the user.

    Args:
        name (str): Patient call tag or name. Default is patient0_t0 for patient 0 at timepoint 0.
        t1 (str): Path to the T1 MRI scan.
        t2 (str): Path to the T2 MRI scan.
        flair (str): Path to the FLAIR MRI scan.
        t1c (str): Path to the contrast-enhanced T1 (i.e., T1CE or T1C) MRI scan.
    """
    name: Optional[str] = Field("patient0_t0", description="Patient identifier or name")
    t1: Optional[str] = Field(None, description="Path to T1 MRI scan")
    t2: Optional[str] = Field(None, description="Path to T2 MRI scan")
    flair: Optional[str] = Field(None, description="Path to FLAIR MRI scan")
    t1c: Optional[str] = Field(None, description="Path to contrast-enhanced T1 (i.e., T1CE or T1C) MRI scan")
    

class Patient(PatientInput):
    """
    Extension of PatientInput to include output file paths for various results.
    """
    # disease: Optional[str] = Field(None, description="Disease type, e.g., glioma, metastasis")
    t1c_w_skull: Optional[str] = Field(None, description="Path to contrast-enhanced T1 (i.e., T1CE or T1C) MRI scan with skull, if available.")
    seg: Optional[str] = Field(None, description="Path for the segmentation result")
    dura: Optional[str] = Field(None, description="Path for the dura segmentation result")
    brain_regions: Optional[str] = Field(None, description="Path for the brain region segmentation result")
    brain_region_volumes: Optional[str] = Field(None, description="Path for the brain region volumes CSV file")
    measurements: Optional[str] = Field(None, description="Path for the measurements JSON file")
    radiomics: List[str] = Field(default_factory=list, description="Paths for the radiomics CSV files")
    loaded_images: dict = Field(default_factory=dict, description="Dict to store loaded images by their keys")


# @function_tool
def check_input_paths(
    patient_input: PatientInput,
) -> tuple[PatientInput, str]:
    """
    Ensure all MRI paths in PatientInput are set. If any are missing, create placeholder images with matching shape.
    This should be called every time after a PatientInput instance is created or modified.
    """
    first_not_none = next((v for k,v in patient_input.model_dump().items() if v is not None and k != "name"), None)

    placeholder_path = Path.cwd() / "placeholder.nii.gz"
    create_placeholder_image(lookup_path=first_not_none, target_path=placeholder_path)

    placeholders = []
    available = []
    for field in patient_input.model_fields_set:
        if field == "name":
            continue
        path = getattr(patient_input, field)
        if path is None or path == "":
            # print(f"Warning: {field} path is not provided. Using placeholder image instead.")
            setattr(patient_input, field, str(placeholder_path))
            placeholders.append(field)
        elif not Path(path).exists():
            # print(f"Warning: {field} path is provided but it does not exist. Using placeholder image instead.")
            setattr(patient_input, field, str(placeholder_path))
            placeholders.append(field)
        else:
            available.append(field)

    # if 'seg' in placeholders:
    #     message_suffix = "Segmentation file is not available."
    #     placeholders.remove('seg')
    # else:
    #     message_suffix = f"Segmentation file is provided: {patient_input.seg}. Saved at key 'seg' in patient context."

    if placeholders:
        message = f"Available inputs: {', '.join(available)}. Using placeholder images for missing ones: {', '.join(placeholders)}"
    else:
        message = "All inputs (t1, t1c, t2, flair) are available."

    # message += (" " + message_suffix)

    return patient_input, message

# @function_tool
def instantiate_patient(
        t1: Optional[str] = None,
        t2: Optional[str] = None,
        flair: Optional[str] = None,
        t1c: Optional[str] = None,
        name: Optional[str] = None,
        seg: Optional[str] = None,
        dura: Optional[str] = None,
        brain_regions: Optional[str] = None,
        brain_region_volumes: Optional[str] = None,
        measurements: Optional[str] = None,
        radiomics: Optional[List[str]] = [],
        t1c_w_skull: Optional[str] = None
) -> tuple[Patient, str]:
    """
    Create a Patient instance from given MRI scan paths. 
    Ensure all MRI paths in PatientInput are set. If any are missing, create placeholder images with matching shape.
    """
    patient = PatientInput(
        name=name,
        t1=t1,
        t2=t2,
        flair=flair,
        t1c=t1c,
    )
    patient, message = check_input_paths(patient)
    patient = Patient(**patient.model_dump(),seg=seg, dura=dura, brain_regions=brain_regions, brain_region_volumes=brain_region_volumes,
                      measurements=measurements, radiomics=radiomics, loaded_images={}, t1c_w_skull=t1c_w_skull)
    
    if t1c_w_skull:
        message += " Additionally, a t1c scan with skull is provided, which can be used for the corresponding segmentation tools if needed."
    provided = []
    for field in ['seg', 'dura', 'brain_regions', 'brain_region_volumes', 'measurements', 'radiomics']:
        val = getattr(patient, field)
        if val:
            provided.append(field)
    if provided:
        message += f" Some outputs are already provided under keys: {', '.join(provided)}."
    else:
        message += " No outputs are pre-provided."
    message = f"Patient name in context: {patient.name}. " + message
    return patient, message


## doesn't work as expected, guardrails do not override the input 
# @input_guardrail
# def instantiate_patient_guardrail(wrapper: RunContextWrapper[SessionContext], agent, llm_input: list) -> str:
#     """
#     Guardrail function to instantiate PatientInput from LLM input and refactor the input.
#     Reads all the patient data from the LLM input JSON string, creates a PatientInput instance, saves it in context.
#     """
#     prompt = llm_input[0] # expecting the prompt sentence as the first item
#     tripwire = False

#     for patient_data in llm_input[1:]:
#         try:
#             patient_dict = json.loads(patient_data)
#         # patient, message = instantiate_patient_to_dict(
#         #     name=patient_dict.get("name"),
#         #     t1=patient_dict.get("t1",),
#         #     t2=patient_dict.get("t2"),
#         #     flair=patient_dict.get("flair"),
#         #     t1c=patient_dict.get("t1c"),
#         #     seg=patient_dict.get("seg")
#         # )
#             patient, message = instantiate_patient(**patient_dict)
#             wrapper.context.set(patient.name, patient)
#             prompt += ("\n" + message)
#         except Exception as e:
#             tripwire = True
#             prompt += f"\nError in creating PatientInput from provided data: {e}"
    
#     return GuardrailFunctionOutput(output_info=prompt, tripwire_triggered=tripwire)



def instantiate_patient_to_dict(
        name: Optional[str] = None,
        t1: Optional[str] = None,
        t2: Optional[str] = None,
        flair: Optional[str] = None,
        t1c: Optional[str] = None,
        seg: Optional[str] = None
) -> dict:
    """
    Check and prepare patient input deterministically to pass to the agent, and not make it use the tool.
    Ensure all MRI paths in PatientInput are set. If any are missing, create placeholder images with matching shape.
    """
    patient = PatientInput(
        name=name,
        t1=t1,
        t2=t2,
        flair=flair,
        t1c=t1c,
        seg=seg
    )
    patient, message = check_input_paths(patient)
    patient = patient.model_dump()
    # del patient["name"]

    return patient

@function_tool
async def load_image(wrapper: RunContextWrapper[SessionContext], patient_name: str, image_to_load: Literal['t1', 't1c', 't2', 'flair', 'seg', 'dura', 'brain_regions']) -> str:
    """
    Load a NIfTI image from stored paths and save it into patient context.
    Args:
        wrapper (RunContextWrapper[SessionContext]): The context wrapper to access global context.
        patient_name (str): The name of the patient in context.
        image_to_load (str): The attribute name of the Patient object to load. available: 't1', 't1c', 't2', 'flair', 'seg', 'dura', 'brain_regions'
    """
    patient = wrapper.context.get(patient_name)
    image_path = getattr(patient, image_to_load)
    image = nib.load(image_path)
    patient.loaded_images[image_to_load] = image
    wrapper.context.set(patient_name, patient)

    return f"Image from path '{image_path}' loaded and stored in patient context with the provided key '{image_to_load}'."

@function_tool
async def read_brain_regions_vocabulary() -> dict:
    """
    Get the vocabulary of all available brain region names and their segmentation labels as a dict.
    """
    regions_vocabulary = BRAIN_REGIONS.copy()  # Create a copy to avoid modifying the original
    # regions_vocabulary.update(BRATS_CLASSES)  # Extend with BRATS classes
    
    return regions_vocabulary
    # return list(regions_vocabulary.keys())


class TargetRegions(BaseModel):
    """
    TargetRegions is a data class to represent the target regions to extract after segmentation.
    It should be constructed based on the regions vocabulary.

    Args:
        names (List[str]): List of region names to extract.
        labels (List[List[int]]): Corresponding list of lists of integer labels for each region.
        
    """
    names: List[str]
    labels: List[List[int]]



# @function_tool
# async def create_target_region_dict(dict_string: str):
#     """
#     Create a dictionary of target regions from a JSON string and save it globally as 'target_regions_dict'.

#     Args:
#         dict_string (str): A dict as a valid JSON string with region names as keys and label lists as values.
#     """
#     try:
#         target_regions = json.loads(dict_string)
#         context['target_regions_dict'] = target_regions
#     except Exception as e:
#         raise ValueError(f"Error in creating region dict")




if __name__ == "__main__":

    # Example usage
    patient_input = PatientInput(
        t1="path/to/t1.mri",
        t2="path/to/t2.mri",
        flair="path/to/flair.mri",
        t1c=None
    )
    print(patient_input)

    target_regions = TargetRegions(
        names=['tumor core', 'whole tumor'],
        labels=[[1, 3], [1, 2, 3]]
    )

    print(target_regions)