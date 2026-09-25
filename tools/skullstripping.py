import os
import subprocess
from pathlib import Path
from typing import List, Optional, Union, Dict

from agents import function_tool, RunContextWrapper, input_guardrail, GuardrailFunctionOutput

import nibabel as nib
import numpy as np

from util.data_types import Patient, TargetRegions
from util.context_manager import SessionContext

CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "all")

def run_synthstrip(input_path: Union[str, Path], output_path: Union[str, Path]) -> None:

    input_path = Path(input_path).resolve().expanduser()
    output_path = Path(output_path).resolve().expanduser()
    output_dir = output_path.parent
    output_name = output_path.name

    # subprocess.run(["./synhtsrip-docker", "-i", "/home/cerdur/data/shared/agent-project/single-timepoint/glioma/preop/aslbbb05/t1.nii.gz", "-o", "./t1_skullstripped.nii.gz" ])

    if CUDA_DEVICES == "all":
        gpu_flag = ["--gpus", "all"]
    else:
        # Docker requires the format "device=0,1"
        gpu_flag = ["--gpus", f"device={CUDA_DEVICES}"]

    uid = os.getuid()
    gid = os.getgid()

    command = [
        "docker", "run", "--rm",
        "--user", f"{uid}:{gid}",
        *gpu_flag,
        "-v" , f"{input_path}:/input/image.nii.gz",
        "-v", f"{output_dir}:/output",
        "freesurfer/synthstrip:1.8-gpu",
        "-i", "/input/image.nii.gz",
        "-o", f"/output/{output_name}"
    ]

    subprocess.run(command)


@function_tool
async def skullstrip_patient(wrapper: RunContextWrapper[SessionContext],patient_name: str) -> str:
    """
    Tool to skullstrip all MRI images of a patient using SynthStrip.
    """
    patient: Patient = wrapper.context.get(patient_name)

    out_dir = wrapper.context.get("output_dir") / patient.name / 'preprocessed' 
    out_dir.mkdir(parents=True, exist_ok=True) 

    processed = []
    for field in ['t1', 't2', 'flair', 't1c']:
        pth = getattr(patient, field)
        if "placeholder" in pth.lower():
            continue
        pth = Path(pth)
        out = out_dir / pth.name.replace('.nii.gz', '_skullstripped.nii.gz')
        run_synthstrip(pth, out)
        setattr(patient, field, str(out))
        processed.append(field)

    wrapper.context.set(patient.name, patient)

    return f"Scans {', '.join(processed)} are skullstripped and their paths updated in patient {patient.name}."



    


