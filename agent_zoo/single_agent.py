import os
import json
import asyncio

from rich.console import Console
from rich.markdown import Markdown

from agents import Agent, ModelSettings

from util.data_types import PatientInput, Patient, check_input_paths, load_image, instantiate_patient, instantiate_patient_to_dict, read_brain_regions_vocabulary
from util.helpers import create_openai_model
from tools.segment import segment_glioma_preop, segment_glioma_postop, segment_brain_metastases_preop, segment_meningioma, segment_brain_regions, segment_pediatric_brain_tumor
from tools.volumetry import select_subregions, get_tumor_lesions_map, get_brain_region_volumes, get_tumor_locations, get_tumor_measurements_per_lesion, get_metastasis_measurements_per_lesion, read_rano_criteria
from tools.matching import match_lesion_maps
from tools.missing_modality import generate_missing_modality
from tools.radiomics import extract_radiomics_features_tumor_core, extract_radiomics_features_whole_tumor
from tools.register import atlas_register_patient, co_register_patient, check_if_registered
from tools.skullstripping import skullstrip_patient

BRATS_CLASSES = json.load(open("./data/brats_classes.json"))
BRAIN_REGIONS = json.load(open("./data/synthseg_classes.json"))

model = os.environ["MODEL_NAME"]
temperature = float(os.environ.get("TEMPERATURE", 0.7))

openai_model = create_openai_model(model)

system_prompt = f"""You are an expert AI assistant for neuroradiology. You can answer questions about brain anatomy, and provide insights into brain diseases. 
Use the given tools to assist your workflow. Do not hesitate to use them due to computational cost or timing concerns.
Use the appropriate segmentation tool based on patient's disease type and pre/post-operative status.
In case any intermediate output (e.g., segmentation) is pre-provided for a patient, skip re-computing it and use the output directly.
Each segmentation tool has specific input requirements. Some can handle raw inputs, but some require co-registration and registration to standard atlas spaces like SRI24 or MNI152.
Always use 'check_if_registered' tool to control if MRI scans of a patient are co-registered, and if they are aligned to standard brain atlas spaces like MNI152 or SRI24.
If they are not, you can either co-register them to t1 and/or register everything to the desired atlas space when needed. 
You can also perform skullstripping. However, if no information is given in the input regarding skullstripping status, accept the MRI scans as already skullstripped.s
Make sure to load the segmentation outputs into context with 'load_image' before performing further analysis. Segmentation tools work directly on Patient data structure, so loading is not required for their inputs. 
Analysis tools can perform volumetric measurements, lesion mapping on segmentations, and radiomics feature extraction on brain MRI scans based on the segmentations.
Avoid too much verbosity in your responses, but be informative and precise.
If there are any output files computed along the way, provide their paths in your final response.

The vocabulary of available targets and their labels are strictly to be adhered to.
Tumor classes: {str(BRATS_CLASSES)}
For brain regions, use the 'read_brain_regions_vocabulary' tool to get the full dict when needed.
"""

single_agent_w_regcheck = Agent(
    name="Assistant",
    instructions=system_prompt,
    model=openai_model,
    tools=[
        check_if_registered,
        co_register_patient,
        atlas_register_patient,
        skullstrip_patient,
        segment_glioma_preop,
        segment_glioma_postop,
        segment_brain_metastases_preop,
        segment_meningioma,
        segment_pediatric_brain_tumor,
        segment_brain_regions,
        load_image,
        read_brain_regions_vocabulary,
        select_subregions,
        get_tumor_lesions_map,
        get_brain_region_volumes,
        get_tumor_measurements_per_lesion,
        get_metastasis_measurements_per_lesion,
        get_tumor_locations,
        match_lesion_maps,
        extract_radiomics_features_tumor_core,
        extract_radiomics_features_whole_tumor,
        read_rano_criteria,
        # run_your_code
    ],
    model_settings=ModelSettings(temperature=temperature,) if model not in ["gpt-5-mini"] else ModelSettings(),
)

single_agent = single_agent_w_regcheck

