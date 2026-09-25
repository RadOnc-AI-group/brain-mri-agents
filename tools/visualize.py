
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from pathlib import Path
from typing import Optional, Union, List
from agents import function_tool
from pydantic import BaseModel, Field
from skimage.measure import label, regionprops

from util.data_types import Patient







@function_tool
async def visualize_result_with_overlay(
    image_name: str,
    patient_input: Patient,
    image_to_select: Optional[str] = None,
    slice_index: Optional[int] = None,
) -> None: 
    """
    Tool to visualize the segmentation result overlaid on the original MRI scans.
    """

    pass