"""
The analysis agent is the entry point, then invokes preprocessing, segmentation agents as needed. All agents speak to each other via handoffs. 
"""
import os
import json

from agents import Agent, ModelSettings

from tools.register import atlas_register_patient, co_register_patient, check_if_registered, check_if_registered_no_tool
from tools.skullstripping import skullstrip_patient
from tools.segment import segment_glioma_preop, segment_glioma_postop, segment_brain_metastases_preop, segment_meningioma, segment_brain_regions, segment_pediatric_brain_tumor
from tools.volumetry import select_subregions, get_tumor_lesions_map, get_brain_region_volumes, get_tumor_locations, get_tumor_measurements_per_lesion, get_metastasis_measurements_per_lesion, read_rano_criteria
from tools.matching import match_lesion_maps
from tools.missing_modality import generate_missing_modality
from tools.radiomics import extract_radiomics_features_tumor_core, extract_radiomics_features_whole_tumor
from util.data_types import PatientInput, Patient, check_input_paths, load_image, instantiate_patient, read_brain_regions_vocabulary
from util.helpers import create_openai_model, session_input_callback

BRATS_CLASSES = json.load(open("./data/brats_classes.json"))
BRAIN_REGIONS = json.load(open("./data/synthseg_classes.json"))

model= os.environ["MODEL_NAME"]
temperature = float(os.environ.get("TEMPERATURE", 0.7))
openai_model = create_openai_model(model)

### put back if you re-activate check_if_registered tool ###
# Always use 'check_if_registered' tool first to control if MRI scans of a patient are co-registered, and if they are aligned to standard brain atlas spaces like MNI152 or SRI24.
# If they are not, you can either co-register them to t1 and/or register everything to the desired atlas space.

system_prompt = """
You are an expert medical image processing assistant specialized in brain MRI preprocessing tasks. You have deep knowledge of neuroimaging standards and best practices.
You ONLY do preprocessing tasks. Use the given tools to co-register MRI scans to t1, and/or to register everything to standard atlas spaces like MNI152 or SRI24. 
The segmentation agent may delegate to you to ensure the input scans meet the segmentation tool requirements.
You can also perform skullstripping. However, if no information is given in the input regarding skullstripping status, accept the MRI scans as already skullstripped.
After finishing preprocessing, hand off back to the agent that has handed off to you.
Be brief, concise and precise in your responses, avoid too much verbosity.
Segmentation tool requirements:
- Glioma pre-op: Skullstripped, co-registered to t1, registered to SRI24
- Glioma post-op: Skullstripped, co-registered to t1, registered to MNI152
- Brain metastases pre-op: Skullstripped, co-registered to t1, registered to SRI24
- Brain metastases post-op: Unskullstripped, no registration required
- Meningioma: Skullstripped, co-registered to t1, registered to SRI24
- Pediatric brain tumor: Skullstripped, co-registered to t1, registered to SRI24
- Brain regions: Skullstripped, no registration required
- Dura: Unskullstripped, no registration required
"""
# unlike as_tools API, agent has to know the seg requirements, because handoffs API doesn't pass messages between agents -- seg agent can't tell what's required

preprocessing_agent_w_regcheck = Agent(
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
    # input_guardrails=[check_if_registered_guardrail],
)

# Ask the user about these details if not provided.
# for further analysis tasks if needed, or finish the task


system_prompt = """
You are a segmentation expert for brain MRIs. You are given different segmentation networks specialized for various brain tumor types and brain structures as tools.
You ONLY do segmentation tasks.
Use the appropriate tool based on patient's disease type and pre/post-operative status. 
Each segmentation tool has specific input requirements. Some can handle raw inputs, but some require co-registration and registration to standard atlas spaces like SRI24 or MNI152.
Use the 'check_if_registered' tool to control if the input scans meet the requirements. However, if no information is given in the input regarding skullstripping status, accept the MRI scans as already skullstripped.
If not, hand off to the preprocessing_assistant to ensure the requirements are met before segmentation. 
Once segmentation is complete, hand off to the analysis_orchestrator agent for further analysis.
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
    ],
    model_settings=ModelSettings(temperature=temperature,) if model not in ["gpt-5-mini"] else ModelSettings(),
    # input_guardrails=[check_if_registered_guardrail],
)

system_prompt = f"""
You are an expert medical image analysis assistant specialized in brain MRI tasks. You have deep knowledge of neuroimaging standards and best practices.
Use the given analysis tools to perform volumetric measurements, lesion mapping, and radiomics feature extraction on brain MRI scans and their segmentations.
There are two specialized agents available for preprocessing and segmentation tasks.
The preprocessing agent can handle skullstripping and image registration tasks, ensuring that MRI scans are co-registered and aligned to standard atlas spaces.
The segmentation agent can perform segmentation of brain tumors and major brain structures using various specialized networks.
Hand over to these agents when their expertise is required. Do not hesitate to call agents or use tools due to computational cost or timing concerns.
You can start directly with the segmentation_assistant. It will hand over to preprocessing_assistant if any preprocessing steps are needed.
Make sure to load the segmentation outputs into context with 'load_image' before performing further analysis. Segmentation tools work directly on Patient data structure, so loading is not required for their inputs. 
In case any intermediate output (e.g., segmentation) is pre-provided for a patient, skip re-computing it and use the output directly.
Avoid too much verbosity in your responses, but be informative and precise.
If there are any output files computed along the way, provide their paths in your final response.

The vocabulary of available targets and their labels are strictly to be adhered to.
Tumor classes: {str(BRATS_CLASSES)}
For brain regions, use the 'read_brain_regions_vocabulary' tool to get the full dict when needed.
"""

analysing_orchestrator_w_regcheck = Agent(
    name="analysis_orchestrator",
    instructions=system_prompt,
    model=openai_model,
    tools=[
        load_image,
        read_brain_regions_vocabulary,
        select_subregions,
        get_tumor_lesions_map,
        get_brain_region_volumes,
        get_tumor_locations,
        get_tumor_measurements_per_lesion,
        get_metastasis_measurements_per_lesion,
        match_lesion_maps,
        extract_radiomics_features_tumor_core,
        extract_radiomics_features_whole_tumor,
        read_rano_criteria
        # run_your_code
    ],
    handoffs=[preprocessing_agent_w_regcheck, segmentation_agent_w_regcheck],
    model_settings=ModelSettings(temperature=temperature,) if model not in ["gpt-5-mini"] else ModelSettings(),
    # input_guardrails=[instantiate_patient_guardrail],
)

segmentation_agent_w_regcheck.handoffs = [preprocessing_agent_w_regcheck, analysing_orchestrator_w_regcheck]
preprocessing_agent_w_regcheck.handoffs = [segmentation_agent_w_regcheck, analysing_orchestrator_w_regcheck]

analysing_orchestrator = analysing_orchestrator_w_regcheck

