"""
The analysis agent is the entry point, then invokes preprocessing, segmentation agents as needed. The agents are presented as tools to each other. 
"""
import os
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field
from agents import Agent, ModelSettings, FileSearchTool

from tools.register import atlas_register_patient, co_register_patient, check_if_registered, check_if_registered_no_tool
from tools.skullstripping import skullstrip_patient
from tools.segment import segment_glioma_preop, segment_glioma_postop, segment_brain_metastases_preop, segment_meningioma, segment_brain_regions, segment_pediatric_brain_tumor
from tools.volumetry import select_subregions, get_tumor_lesions_map, get_brain_region_volumes, get_tumor_locations, get_tumor_measurements_per_lesion, get_metastasis_measurements_per_lesion, read_rano_criteria
from tools.matching import match_lesion_maps
from tools.missing_modality import generate_missing_modality
from tools.radiomics import extract_radiomics_features_tumor_core, extract_radiomics_features_whole_tumor
from util.data_types import SubAgentResponse, SegmentationResponse, PreprocessingRequest, SegmentationRequest, load_image, read_brain_regions_vocabulary
from util.helpers import create_openai_model

BRATS_CLASSES = json.load(open("./data/brats_classes.json"))
BRAIN_REGIONS = json.load(open("./data/synthseg_classes.json"))

model = os.environ["MODEL_NAME"]
temperature = float(os.environ.get("TEMPERATURE", 0.7))
openai_model = create_openai_model(model)

system_prompt = """
You are an expert medical image processing assistant specialized in brain MRI preprocessing tasks. You have deep knowledge of neuroimaging standards and best practices.
Use the given tools to co-register MRI scans to t1, and/or to register everything to standard atlas spaces like MNI152 or SRI24. 
You can also perform skullstripping.
The segmentation agent may delegate to you to ensure the input scans meet the segmentation tool requirements.
Follow the instruct
Be brief, concise and precise in your responses, avoid too much verbosity.
"""
preprocessing_agent = Agent(
    name="preprocessing_assistant",
    instructions=system_prompt,
    model=openai_model,
    tools=[
        atlas_register_patient,
        co_register_patient,
        skullstrip_patient,
        # check_if_registered,
        # run_your_code
    ],
    model_settings=ModelSettings(temperature=temperature,) if model not in ["gpt-5-mini"] else ModelSettings(),
    # input
    # output_type=SubAgentResponse,

    # input_guardrails=[check_if_registered_guardrail],
)

preprocessing_agent_as_tool = preprocessing_agent.as_tool(
    tool_name="preprocessing_assistant",
    tool_description="Expert agent for brain MRI preprocessing tasks like skullstripping and image registration.",
    # parameters=PreprocessingRequest,
    # include_input_schema=True
)

system_prompt = """
You are a segmentation expert for brain MRIs. You are given different segmentation networks specialized for various brain tumor types and brain structures as tools.
Use the appropriate tool based on patient's disease type and pre/post-operative status. 
Each segmentation tool has specific input requirements. Some can handle raw inputs, but some require co-registration and registration to standard atlas spaces like SRI24 or MNI152.
Use the 'check_if_registered' tool to control if the input scans meet the requirements. If no information is given in the input regarding skullstripping status, accept the MRI scans as already skullstripped.
If the requirements are not met, call the preprocessing agent to ensure they are fulfilled before segmentation. 
Be brief, concise and precise in your responses, avoid too much verbosity.
"""

segmentation_agent_w_regcheck = Agent(
    name="segmentation_assistant",
    instructions=system_prompt,
    model=openai_model,
    tools=[
        check_if_registered,
        segment_glioma_preop,
        segment_glioma_postop,
        segment_brain_metastases_preop, 
        segment_meningioma,
        segment_brain_regions,
        segment_pediatric_brain_tumor,
        preprocessing_agent_as_tool,
        # run_your_code
    ],
    model_settings=ModelSettings(temperature=temperature,) if model not in ["gpt-5-mini"] else ModelSettings(),
    # output_type=SegmentationResponse,
    # input_guardrails=[check_if_registered_guardrail],
)

segmentation_agent_w_regcheck_as_tool = segmentation_agent_w_regcheck.as_tool(
    tool_name="segmentation_assistant",
    tool_description="Expert agent for brain MRI segmentation tasks using specialized networks for different tumor types and brain structures.",
    # parameters=SegmentationRequest,
    # include_input_schema=True
)


# The preprocessing agent can handle skullstripping and image registration tasks, ensuring that MRI scans are co-registered and aligned to a standard atlas space.
# The segmentation agent can perform brain tumor and structure segmentation using specialized networks.

system_prompt = f"""
You are an expert medical image analysis assistant specialized in brain MRI tasks. You have deep knowledge of neuroimaging standards and best practices.
Use the given analysis tools to perform volumetric measurements, lesion mapping, and radiomics feature extraction on brain MRI scans and their segmentations.
There are two specialized agents available for preprocessing and segmentation tasks.
The preprocessing agent can handle skullstripping and image registration tasks, ensuring that MRI scans are co-registered and aligned to standard atlas spaces.
The segmentation agent can perform segmentation of brain tumors and major brain structures using various specialized networks.
Call these agents as tools when their expertise is needed. Do not hesitate to use agents or tools due to computational cost or timing concerns.
You can start directly with the segmentation_assistant. It will hand over to preprocessing_assistant if any preprocessing steps are needed.
Make sure to load the segmentation outputs into context with 'load_image' before performing further analysis. Segmentation tools work directly on Patient data structure, so loading is not required for their inputs. 
In case any intermediate output is pre-provided for a patient, skip re-computing it and use the output directly.
If no information is given in the input regarding skullstripping status, accept the MRI scans as already skullstripped.
Avoid too much verbosity in your responses, but be informative and precise.
If there are any output files computed along the way, provide their paths in your final response.

The vocabulary of available targets and their labels are strictly to be adhered to.
Tumor classes: {str(BRATS_CLASSES)}
For brain regions, use the 'read_brain_regions_vocabulary' tool to get the full dict when needed.
"""
# Always pass the prior patient knowledge in a clear and structured manner along to other agents when calling them. Be sure to include all relevant information available in the input (e.g. registration status, any pre-existing segmentations).

analysing_orchestrator_w_regcheck = Agent(
    name="analysis_orchestrator",
    instructions=system_prompt,
    model=openai_model,
    tools=[
        load_image,
        read_brain_regions_vocabulary,
        # select_subregions,
        get_tumor_lesions_map,
        get_brain_region_volumes,
        get_tumor_locations,
        get_tumor_measurements_per_lesion,
        get_metastasis_measurements_per_lesion,
        match_lesion_maps,
        extract_radiomics_features_tumor_core,
        extract_radiomics_features_whole_tumor,
        preprocessing_agent_as_tool,
        segmentation_agent_w_regcheck_as_tool,
        read_rano_criteria
        # run_your_code
    ],
    model_settings=ModelSettings(temperature=temperature,) if model not in ["gpt-5-mini"] else ModelSettings(),
)

analysing_orchestrator = analysing_orchestrator_w_regcheck