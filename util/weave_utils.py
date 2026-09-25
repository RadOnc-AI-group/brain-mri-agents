import os
import re
import json
import time
import weave
import locale
import asyncio
import difflib
import pandas as pd
from typing import Any
from pathlib import Path
from copy import deepcopy
from collections import Counter
from difflib import SequenceMatcher
# from weave.scorers import SummarizationScorer
from agents import Runner, Agent, SQLiteSession, RunHooks, RunContextWrapper, Tool, ModelSettings
from pydantic import PrivateAttr, Field, BaseModel


from tools.register import check_if_registered_no_tool
from util.data_types import instantiate_patient
from util.context_manager import SessionContext
from util.helpers import create_openai_model

LLM_PRICES = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "cached": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4.1": {"input": 2.00 / 1_000_000, "cached": 0.50 / 1_000_000, "output": 8.00 / 1_000_000},
    "gpt-4.1-mini": {"input": 0.40 / 1_000_000, "cached": 0.1 / 1_000_000, "output": 1.60 / 1_000_000},
    "gpt-5": {"input": 1.25 / 1_000_000, "cached": 0.125 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-5.1": {"input": 1.25 / 1_000_000, "cached": 0.125 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-5.2": {"input": 1.75 / 1_000_000, "cached": 0.175 / 1_000_000, "output": 14.00 / 1_000_000},
    "gpt-5-mini": {"input": 0.25 / 1_000_000, "cached": 0.025 / 1_000_000, "output": 2.00 / 1_000_000},
    "gpt-5-pro": {"input": 15.00 / 1_000_000, "cached": 15.00 / 1_000_000, "output": 120.00 / 1_000_000},
    "gpt-5.4": {"input": 2.50 / 1_000_000, "cached": 0.25 / 1_000_000, "output": 15.00 / 1_000_000},
    "gpt-5.5": {"input": 5. / 1_000_000, "cached": 0.5 / 1_000_000, "output": 30.00 / 1_000_000},
    "o3": {"input": 2.00 / 1_000_000, "cached": 0.50 / 1_000_000, "output": 8.00 / 1_000_000},
    "o3-mini": {"input": 1.10 / 1_000_000, "cached": 0.55 / 1_000_000, "output": 4.40 / 1_000_000},
    "o4-mini": {"input": 1.10 / 1_000_000, "cached": 0.275 / 1_000_000, "output": 4.40 / 1_000_000},
    "o4-mini-deep-research": {"input": 2.00 / 1_000_000, "cached": 0.5 / 1_000_000, "output": 8.00 / 1_000_000},
    "gemini-2.5-flash": {"input": 0.3 / 1_000_000, "cached": 0.03 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-2.5-flash-lite": {"input": 0.10 / 1_000_000, "cached": 0.01 / 1_000_000, "output": 0.40 / 1_000_000},
    "gemini-2.5-pro": {"input": 1.25 / 1_000_000, "cached": 0.125 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-3-flash-preview": {"input": 0.5 / 1_000_000, "cached": 0.05 / 1_000_000, "output": 3.00 / 1_000_000},
    "gemini-3-pro-preview": {"input": 2.00 / 1_000_000, "cached": 0.20 / 1_000_000, "output": 12.00/ 1_000_000},
    "gemini-3.1-pro-preview": {"input": 2.00 / 1_000_000, "cached": 0.20 / 1_000_000, "output": 12.00/ 1_000_000},
    "gemini-3.5-flash": {"input": 1.5 / 1_000_000, "cached": 0.15 / 1_000_000, "output": 9.00 / 1_000_000},
    "claude-sonnet-4-20250514": {"input": 3. / 1_000_000, "cached": 0.3 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-sonnet-4-5-20250929": {"input": 3. / 1_000_000, "cached": 0.3 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3. / 1_000_000, "cached": 0.3 / 1_000_000, "output": 15.00 / 1_000_000},
    "google/gemma-4-31b": {"input": 0. / 1_000_000, "cached": 0. / 1_000_000, "output": 0. / 1_000_000},
    "google/gemma-4-e4b": {"input": 0. / 1_000_000, "cached": 0. / 1_000_000, "output": 0. / 1_000_000},
    "qwen/qwen3.5-9b": {"input": 0. / 1_000_000, "cached": 0. / 1_000_000, "output": 0. / 1_000_000},
    "qwen/qwen3.6-35b-a3b": {"input": 0. / 1_000_000, "cached": 0. / 1_000_000, "output": 0. / 1_000_000},

}

LLM_RATE_LIMITS = {
    "gpt-5": {"requests_per_minute": 500, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 500_000},
    "gpt-5.1": {"requests_per_minute": 500, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 500_000}, # not sure about numbers but there is basically no limit
    "gpt-5.2": {"requests_per_minute": 500, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 500_000},
    "gpt-5.4": {"requests_per_minute": 500, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 500_000},
    "gpt-5.5": {"requests_per_minute": 500, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 500_000},
    "gpt-5-mini": {"requests_per_minute": 500, "input_tokens_per_minute": 4_000_000, "output_tokens_per_minute": 500_000},
    "gemini-2.5-pro": {"requests_per_minute": 150, "input_tokens_per_minute": 1_200_000, "output_tokens_per_minute": 65_789},
    "gemini-2.5-flash": {"requests_per_minute": 150, "input_tokens_per_minute": 1_200_000, "output_tokens_per_minute": 65_789},
    "gemini-3-pro-preview": {"requests_per_minute": 25, "input_tokens_per_minute": 1_000_000, "output_tokens_per_minute": 65_000},
    "gemini-3.1-pro-preview": {"requests_per_minute": 25, "input_tokens_per_minute": 1_000_000, "output_tokens_per_minute": 65_000},
    "gemini-3.5-flash": {"requests_per_minute": 1000, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 65_000},
    "claude-sonnet-4-20250514": {"requests_per_minute": 1_000, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 400_000},
    "claude-sonnet-4-5-20250929": {"requests_per_minute": 1_000, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 400_000},
    "claude-sonnet-4-6": {"requests_per_minute": 1_000, "input_tokens_per_minute": 2_000_000, "output_tokens_per_minute": 400_000},
    "google/gemma-4-31b": {"requests_per_minute": 500, "input_tokens_per_minute": 262144, "output_tokens_per_minute": 50_000},
    "google/gemma-4-e4b": {"requests_per_minute": 500, "input_tokens_per_minute": 131072, "output_tokens_per_minute": 50_000},
    "qwen/qwen3.5-9b": {"requests_per_minute": 500, "input_tokens_per_minute": 262144, "output_tokens_per_minute": 50_000},
    "qwen/qwen3.6-35b-a3b": {"requests_per_minute": 500, "input_tokens_per_minute": 262144, "output_tokens_per_minute": 50_000},
}


def _selectively_deduplicate(tool_list: list) -> list:
    """
    Remove duplicates for handoffs only. It shouldn't be relevant if the agent processes timeponts sequentially or in a batch. We will see the inefficiency of this in cost-related metrics.
    """
    new_list = []
    seen = set()
    for t in tool_list:
        if "_assistant" in t or "_orchestrator" in t:
            seen.add(t)
            if t not in new_list:
                new_list.append(t)
        else:
            new_list.append(t)
            
    return new_list

@weave.op
def control_tool_usage(gt_tools: list, output: dict, task_type: str = None) -> dict:
    if isinstance(gt_tools, weave.ObjectRef):
        gt_tools = gt_tools.get()
    stats = output.get("stats", {})
    used_tools = stats.get("executed_tools", [])

    full_trace = [t["name"] for t in used_tools] 
    errorless_trace = [t["name"] for t in used_tools if not t.get("is_error", False)]
    if task_type == "handoffs":
        full_trace = _selectively_deduplicate(full_trace)
        errorless_trace = _selectively_deduplicate(errorless_trace)
    num_used = len(full_trace)
    num_errors = stats.get("error_count", 0)
    num_correct = len(errorless_trace)
    llm_responses = stats.get("llm_responses", [])

    gt_counter = Counter(gt_tools)
    trace_counter = Counter(errorless_trace)

    # for tool in errorless_trace:
    #     if "_assistant" in tool or "_orchestrator" in tool:
    #         trace_counter[tool] = 1  # remove the duplicates of handoff tools, as long as they are called at least once, it should be fine -- we see the inefficiency of multiple handoffs in cost-related metrics

    matches_unordered = sum((gt_counter & trace_counter).values())
    num_missing = sum((gt_counter - trace_counter).values()) # False Negatives
    num_extra = sum((trace_counter - gt_counter).values())   # False Positives
    
    matcher = difflib.SequenceMatcher(None, gt_tools, errorless_trace)
    # Sum of the lengths of all matching blocks
    ordered_matches = sum(triple.size for triple in matcher.get_matching_blocks())
    
    # Denominators
    len_gt = max(1, len(gt_tools))
    len_trace = max(1, num_correct)

    error_rate = num_errors / max(1, num_used)
    # Unordered Metrics (Did it select the right tools?)
    unordered_prec = matches_unordered / len_trace  # of all tool calls made, how many were expected?
    unordered_recall = matches_unordered / len_gt    # of all expected tool calls, how many were made?
    unordered_f1 = 2 * (unordered_prec * unordered_recall) / max(1, (unordered_prec + unordered_recall))
    jaccard_index = matches_unordered / max(1, (len_gt + num_extra))  # Intersection over Union

    # Ordered Metrics (Did it do them in the right order?)
    similarity = matcher.ratio() # Standard difflib similarity
    ordered_prec = ordered_matches / len_trace  
    ordered_recall = ordered_matches / len_gt
    
    return {
        "total_tool_call_count": num_used,
        "matched_tool_call_count": matches_unordered,
        "missed_tool_call_count": num_missing,
        "extra_tool_call_count": num_extra,
        "extra_tool_call_rate": num_extra / max(1, num_correct),
        "errorless_tool_call_count": num_correct,
        "errorless_tool_call_rate": num_correct / max(1, num_used),
        "error_rate": error_rate,
        "unordered_precision": unordered_prec,
        "unordered_recall": unordered_recall,
        "unordered_f1": unordered_f1,
        "jaccard_index": jaccard_index,
        "ordered_precision": ordered_prec,
        "ordered_recall": ordered_recall,
        "ordered_f1": similarity,
    }

## unnecessary -- weave returns them inside "output.stats" already
@weave.op
def extract_costs(output: dict) -> dict:
    stats = output.get("stats", {})
    (
        num_tool_calls,
        llm_request_count,
        # input_tokens, ## try the automatic weave extraction first, if not working, use the manual function
        # cached_tokens,
        # output_tokens,
        
     ) =  stats["tool_call_count"], stats["llm_request_count"], # stats["total_input_tokens"], stats["total_output_tokens"]
    
    
    return {
        "total_tool_calls": num_tool_calls,
        # "total input tokens": input_tokens,
        # "total output tokens": output_tokens,
        "total_llm_requests": llm_request_count,
    }

@weave.op
def control_rano(llm_scorer_input: dict, output: dict) -> dict:
    if isinstance(llm_scorer_input, weave.ObjectRef):
            llm_scorer_input = llm_scorer_input.get()

    gt = llm_scorer_input[-1]    ## in form "\n<expected answers>\n{list(contents)}\n</expected answers>" parse back to list
    gt = re.search(r"<expected answers>\n(.*?)\n</expected answers>", gt, re.DOTALL)
    if gt:
        gt = gt.group(1).split(': ')[1].strip("']")
        # gt = gt[0] if isinstance(gt, list) and len(gt) > 0 else gt

    lookup = {"SD": "Stable Disease", "PD": "Progressive Disease", "PR": "Partial Response", "CR": "Complete Response"}
    second_gt = lookup.get(gt,gt)    
    response = output.get("response")

    if gt in response or second_gt in response:
        return {"correct_response_acc": 1}
    else:
        return {"correct_response_acc": 0}


class JudgeOutput(BaseModel):
    total_queried: int = Field(..., description="Total number of aspects asked about in the query.")
    total_responded: int = Field(..., description="Number of aspects mentioned in the response.")
    total_responded_correctly: int = Field(..., description="Number of aspects for which the number/measurement/value was correctly given in the response.")
    num_hallucinations: int = Field(..., description="Number of hallucinated aspects in the response (i.e., aspects that were not asked about in the query).")

class ResponseJudgeLLM(weave.Scorer):
    model_id: str = os.environ.get("LLM_JUDGE", "gpt-5-mini")
    system_prompt: str = """
    You are evaluating the response of an agentic pipeline. The pipeline just receives the initial <query> and produces a final <response> using various tools.
    You will receive the <query> along with a list of <expected answers>, and the pipeline's <response>. The answers are formatted as a list, and are ordered by timepoint->lesion->measurement->value.
    In some cases, this format in <expected answers> is broken and multiple answers are combined into a single string, but you can still figure out the individual answers based on the content. 
    Count each aspect that is asked about in the <query>. An aspect can be a specific measurement (e.g., tumor core volume of lesion 1 at timepoint 0), or a more general aspect (e.g., whether there is a new lesion at timepoint 1 compared to timepoint 0). Then check if each aspect is included in the <response>, and whether it is answered correctly.
    The <response> may not be structured and can come in a free-text format. So, you need to understand the content of the <response> carefully.
    Do not pay attention to typos or small wording differences, as long as the meaning is correct.
    In some cases, the lesion indexing between the <expected answers> and the <response> may differ (e.g., lesion 1 in expected answers corresponds to lesion 2 in response). You need to figure this out based on the volumes or other context. Do not mark answers as incorrect just because lesion indices mismatch, only if there is no clear correspondence.
    Also ignore the enumaration of timepoints. Sometimes it starts from 0, sometimes from 1, but the order is always preserved. Just pay attention to the context.
    For numeric values, a small margin of error of +/- 10% is allowed, but not on 'counts' (e.g., number of lesions).
    Sometimes the agent responds large values with , as a thousands separator (e.g., 22,295). You should interpret this as 22295.0.
    In the output, provide the following: 
     - the total number of unique aspects asked about in the query.
     - the number of aspects mentioned in the response, independent of whether the numbers/values/answers were correct. this also includes aspects that were hallucinated (i.e., not asked about in the query).
     - the number of aspects for which the numbers/values/answers were correctly given in the response
     - the number of hallucinated aspects in the response (i.e., aspects that were not asked about in the query). output file paths and useful notices are not considered hallucinations.
     
    Example 1:
    <query>
    "How many lesions does this glioma patient have, and what is the cumulative total tumor core volume across all lesions?"
    </query>
    <expected answers>
    ["num total lesions: 1", "cumulative total tumor core volume (all lesions): 16134.0 mm3"]
    </expected answers>
    <response>
    "This glioma patient has 1 lesion. The cumulative total tumor core volume across all lesions is 16134.0 mm3. The total edema volume is 10293.2 mm3."
    </response>
    total_queried: 2
    total_responded: 2
    total_responded_correctly: 2
    num_hallucinations: 1 (edema volume is not asked about in the query)

    Example 2:
    <query>
    "Give volumes for left and right cerebellum cortex and white matter."
    </query>
    <expected answers>
    ['left cerebellum cortex: 64325.51', 'left cerebellum white matter: 19968.639', 'right cerebellum cortex: 61957.84', 'right cerebellum white matter: 17896.9']
    </expected answers>
    <response>
    "The volumes are as follows: left cerebellum cortex: 64325.51, left cerebellum white matter: 19968.639, right cerebellum white matter: 16000."
    </response>
    total_queried: 4
    total_responded: 3 (one target missing)
    total_responded_correctly: 2 (one number incorrect)
    num_hallucinations: 0
    """
    
    def __init__(self, **data):
        super().__init__(**data)
        model = create_openai_model(self.model_id)
        self._agent = Agent(
            name="judge_agent",
            model=model,
            instructions=self.system_prompt,
            output_type=JudgeOutput,
            # model_settings=ModelSettings(temperature=0.0),
        )

    @weave.op
    async def score(self, llm_scorer_input: list, output: dict) -> dict:
        if isinstance(llm_scorer_input, weave.ObjectRef):
            llm_scorer_input = llm_scorer_input.get()
        
        response = output.get("response")
        # total_queried = float(llm_scorer_input[-1])
        prompt = f"{''.join(llm_scorer_input)}\n<response>\n{response}\n</response>"
        result = await Runner.run(
            self._agent,
            prompt,    
        )
        output = result.final_output.model_dump()
        total_queried = output['total_queried']
        include_rate = min(1., output["total_responded"] / max(1., total_queried))
        correct_response_rate = output["total_responded_correctly"] / max(1., total_queried)
        correct_response_precision = min(1.,output["total_responded_correctly"] / max(1., output["total_responded"]))
        hallucination_rate = output["num_hallucinations"] / max(1., output["total_responded"])
        quality_score = 0.3 * include_rate + 0.5 * correct_response_rate + 0.2 * (1 - hallucination_rate)
        output["response_inclusion_rate"] = include_rate
        output["correct_response_acc"] = min(1., correct_response_rate)
        output["correct_response_acc_raw"] = correct_response_rate
        output["correct_response_precision"] = correct_response_precision
        output["hallucination_rate"] = hallucination_rate
        output["quality_score"] = max(0., quality_score)

        return output


# ============================================================================
# HYBRID RESPONSE JUDGE: LLM Parsing + Deterministic Validation
# ============================================================================
#
# APPROACH:
# 1. LLM parses ground truth list → structured QA pairs (handles complex context)
# 2. LLM parses free-text response → structured QA pairs (handles natural language)
# 3. Deterministic code matches pairs and validates correctness (reproducible!)
#
# WHY THIS ADDRESSES REVIEWER CONCERNS:
# - Correctness checking is fully deterministic and auditable
# - Unit conversion, tolerance, and matching logic is transparent
# - LLM only does what it's good at: parsing complex/contextual text
# - Results are reproducible: same parsed pairs → same validation outcome
#
# HANDLES EDGE CASES:
# - Missing units in ground truth (LLM infers from context: volumes→mm3, diameters→mm, changes→%)
# - Complex multi-QA items: "lesion 1: enhancing: 284 mm3, non-enhancing: 11 mm3"
#   (LLM preserves context when splitting)
# - Text answers: "location: frontal lobe" (categorical matching)
# - Unit conversion: cm3 ↔ mm3, cm ↔ mm (deterministic conversion table)
# - Percentage values: "volume increase: 20%" (treated as numeric with tolerance)
# - Coordinate data: "bbox: (129, 79, 47, 174, 144, 117)" (element-wise comparison with tolerance)
# - Centroids: "centroid: (150, 110, 80)" (3D coordinate triplets)
# - Unitless counts: "number of lesions: 3" (exact matching, no tolerance)
# - Fuzzy aspect matching: handles minor wording differences
# - Tolerance: 0-1 normalized scale (0.1 = 10% tolerance)
# ============================================================================

class ParsedQA(BaseModel):
    """Represents a single parsed question-answer pair from the response."""
    aspect: str = Field(..., description="The specific measurement or aspect being answered (e.g., 'tumor core volume of lesion 1 at timepoint 0')")
    value: float | str | list[float] | None = Field(..., description="The numerical value, categorical answer, or coordinate list (for bbox/centroid) provided")
    unit: str | None = Field(default=None, description="The unit of measurement if applicable (e.g., 'mm', 'cm3', 'coordinates')")
    # raw_text: str = Field(..., description="The original text snippet from the response for this Q&A")

class JudgeParseOutput(BaseModel):
    """Output from LLM that only parses the response, without checking correctness."""
    parsed_qas: list[ParsedQA] = Field(..., description="List of all question-answer pairs extracted from the response")
    # total_aspects_found: int = Field(..., description="Total number of aspects addressed in the response")

# Unit conversion utilities
UNIT_CONVERSIONS = {
    # Volume conversions (all to mm3 as base unit)
    "mm3": 1.0,
    "mm³": 1.0,
    "cm3": 1000.0,
    "cm³": 1000.0,
    "ml": 1000.0,
    "cc": 1000.0,
    "l": 1_000_000.0,
    # Length conversions (all to mm as base unit)
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    # Percentage (no conversion, base unit is percent)
    "percent": 1.0,
    "%": 1.0,
    "coordinates": 1.0,  # Special marker for coordinate data, no conversion
    # "count" : 1.0,
    
}

def normalize_unit(unit: str | None) -> str | None:
    """Normalize unit string to canonical form."""
    if not unit:
        return None
    unit = unit.strip().lower()
    # Handle unicode superscripts
    unit = unit.replace("³", "3")
    # Normalize percentage symbols
    if unit == "%":
        unit = "percent"
    return unit

def safe_numeric_convert(value: any) -> float | list | str | None:
    """
    Safely convert a value to numeric format, handling edge cases.
    
    Handles:
    - Strings with commas: '135,295' -> 135295.0
    - Already numeric values: pass through
    - Lists/arrays: return as-is
    - Non-numeric strings: return as-is (for categorical matching)
    
    Returns the converted value or original if conversion fails.
    """
    # If it's already a list, return as-is (coordinates)
    if isinstance(value, list):
        return value
    
    # If it's already numeric, return as-is
    if isinstance(value, (int, float)):
        return float(value)
    
    # Try to convert string to float
    if isinstance(value, str):
        # Remove commas and whitespace
        cleaned = value.replace(',', '').replace("−", "-").strip()
        try:
            return float(cleaned)
        except ValueError:
            # Not a numeric string, return original
            return value
    
    # For any other type, return as-is
    return value


def convert_to_base_unit(value: float | str | list, unit: str | None) -> tuple[float | list | str, str]:
    """
    Convert a value to base unit (mm3 for volumes, mm for lengths, percent for percentages).
    Returns (converted_value, base_unit)
    If unit is None, returns value as-is with 'unitless' marker.
    """
    # First, safely convert the value to appropriate type
    converted_value = safe_numeric_convert(value)
    
    if unit is None:
        # Return as unitless - let matching handle this case
        return converted_value, "unitless"
    
    norm_unit = normalize_unit(unit)
    if norm_unit in UNIT_CONVERSIONS:
        # Determine unit type
        if norm_unit == "percent":
            # Percentage unit (ensure it's numeric)
            if not isinstance(converted_value, (int, float)):
                converted_value = safe_numeric_convert(value)
            return converted_value, "percent"
        elif '3' in norm_unit or '³' in norm_unit or norm_unit in ['ml', 'cc', 'l']:
            # Volume unit (ensure it's numeric)
            if not isinstance(converted_value, (int, float)):
                converted_value = safe_numeric_convert(value)
            return float(converted_value) * UNIT_CONVERSIONS[norm_unit] , "mm3"
        elif norm_unit == "coordinates":
            # Coordinate data (should be a list, no conversion needed)
            return converted_value, "coordinates"
        else:
            # Length unit (ensure it's numeric)
            if not isinstance(converted_value, (int, float)):
                converted_value = safe_numeric_convert(value)
            return float(converted_value) * UNIT_CONVERSIONS[norm_unit] , "mm"
    
    # Unknown unit, return value as-is
    return converted_value, norm_unit or "unitless"


def normalize_aspect_for_matching(aspect: str) -> str:
    """
    Normalize aspect text for robust matching.

    Lesion numbering is not stable in some longitudinal pipelines (e.g., lesion 2 in
    GT can be lesion 3 in the response after matching/reindexing). We intentionally
    remove lesion indices while keeping other context (like timepoints and modality
    terms) so semantically equivalent aspects can still match.
    """
    normalized = aspect.lower().strip()
    normalized = normalized.replace("−", "-")

    # Remove explicit lesion indices but keep the lesion token itself.
    normalized = re.sub(r"\blesion\s*[-_#:]?\s*\d+\b", "lesion", normalized)

    # Normalize whitespace/punctuation noise.
    normalized = re.sub(r"[_,:;]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def value_similarity_score(parsed_value: float | str | list,
                           expected_value: float | str | list,
                           parsed_unit: str,
                           expected_unit: str) -> float:
    """Return a [0,1] compatibility score to disambiguate near-equal aspect names."""
    # Different base units are generally incompatible.
    if parsed_unit != expected_unit:
        return 0.0

    # Coordinate data: score by per-element deviation.
    if isinstance(parsed_value, list) and isinstance(expected_value, list):
        if len(parsed_value) != len(expected_value):
            return 0.0
        if not parsed_value:
            return 1.0
        diffs = []
        for p, e in zip(parsed_value, expected_value):
            try:
                diffs.append(abs(float(p) - float(e)))
            except (TypeError, ValueError):
                return 1.0 if str(p).strip().lower() == str(e).strip().lower() else 0.0
        avg_diff = sum(diffs) / len(diffs)
        # 5 mm average error maps to score 0.
        return max(0.0, 1.0 - (avg_diff / 5.0))

    # Numeric data: use relative error when possible.
    if isinstance(parsed_value, (int, float)) and isinstance(expected_value, (int, float)):
        if parsed_unit == "unitless" and expected_unit == "unitless":
            return 1.0 if abs(parsed_value - expected_value) < 1e-6 else 0.0
        rel_err = abs(parsed_value - expected_value) / (abs(expected_value) + 1e-6)
        return max(0.0, 1.0 - rel_err)

    # Text/categorical data.
    return SequenceMatcher(None, str(parsed_value).lower().strip(), str(expected_value).lower().strip()).ratio()

def fuzzy_match_aspect(aspect1: str, aspect2: str,) -> float:
    """
    Check if two aspect strings are similar enough to be considered a match.
    Uses case-insensitive comparison and allows for minor wording differences.
    """
    
    # Raw normalize
    a1 = aspect1.lower().strip()
    a2 = aspect2.lower().strip()
    # Lesion-index-invariant normalize
    n1 = normalize_aspect_for_matching(aspect1)
    n2 = normalize_aspect_for_matching(aspect2)
    
    # Exact match
    if a1 == a2 or n1 == n2:
        return 1.0
    
    # Fuzzy match
    raw_ratio = SequenceMatcher(None, a1, a2).ratio()
    normalized_ratio = SequenceMatcher(None, n1, n2).ratio()
    return max(raw_ratio, normalized_ratio)

def find_matching_expected_qa(parsed_aspect: str, parsed_base_val: float | str | list, parsed_base_unit: str,
                              expected_qas: list[dict], matched_indices: set, threshold: float = 0.7) -> tuple[int | None, dict | None]:
    """
    Find the expected QA that matches the parsed QA based on aspect similarity.
    Returns the best match or None if no good match is found.
    """
    best_match = None
    best_score = 0.0
    idx = None

    for i, expected in enumerate(expected_qas):
        # Skip if this GT QA has already been matched
        if i in matched_indices:
            continue
            
        # Check aspect similarity
        score = fuzzy_match_aspect(parsed_aspect, expected["aspect"])
        if score > threshold:
            # Convert expected to base units for unit compatibility check
            expected_base_val, expected_base_unit = convert_to_base_unit(expected["value"], expected["unit"])
            
            # Units must be compatible (same base unit) or both be non-numeric
            units_compatible = (
                parsed_base_unit == expected_base_unit or
                (not isinstance(parsed_base_val, (int, float, list)) and not isinstance(expected_base_val, (int, float, list)))
            )
            
            if units_compatible:
                # Tie-break with value compatibility to avoid cross-lesion swaps
                # when lesion indices differ but multiple aspects are textually similar.
                val_score = value_similarity_score(
                    parsed_base_val,
                    expected_base_val,
                    parsed_base_unit,
                    expected_base_unit,
                )
                combined_score = (0.75 * score) + (0.25 * val_score)
                if combined_score > best_score:
                    best_score = combined_score
                expected["value"] = expected_base_val 
                expected["unit"] = expected_base_unit
                best_match = expected
                idx = i
    
    return idx,best_match # if best_score >= threshold else None


def compare_converted_values(parsed_base: float, parsed_base_unit: str,
                             expected_base: float, expected_base_unit: str,
                             tolerance_pct: float = 0.1) -> bool:
    """
    Compare already-converted base values 
    
    This function expects values that have already been converted to base units
    via convert_to_base_unit().

    Returns:
        True if values match within tolerance, False otherwise
    """
    # If both are unitless (counts), require exact match or very small tolerance
    if parsed_base_unit == "unitless" and expected_base_unit == "unitless":
        # Allow tiny floating point errors but essentially exact match
        return abs(parsed_base - expected_base) < 1e-6
    
    # If units don't match after conversion to base units, they're incompatible
    if parsed_base_unit != expected_base_unit:
        return False
    
    # Compare with tolerance (for mm3, mm, percent with their respective tolerances)
    frac_diff = abs(parsed_base - expected_base) / abs(expected_base + 1e-6)  # Avoid division by zero
    return frac_diff <= tolerance_pct


def check_coordinate_correctness(parsed_coords: list, expected_coords: list, tolerance_mm: int = 5) -> bool:
    """
    Check if coordinate data (bounding box, centroid) is correct.
    Compares element-wise with tolerance.
    
    Args:
        parsed_coords: List of coordinates from parsed response (e.g., [129, 79, 47, 174, 144, 117])
        expected_coords: List of expected coordinates
        tolerance_mm: Allowed difference in millimeters per coordinate
    """
    # Must have same length
    if len(parsed_coords) != len(expected_coords):
        return False
    
    # Check each coordinate element-wise
    for parsed_val, expected_val in zip(parsed_coords, expected_coords):
        try:
            p_num = float(parsed_val)
            e_num = float(expected_val)
            
            # # Handle zero case
            # if e_num == 0:
            #     if abs(p_num) > 1e-6:
            #         return False
            # else:
            # Apply tolerance
            diff = abs(p_num - e_num)
            if diff > tolerance_mm:
                return False
        except (ValueError, TypeError):
            # If conversion fails, do exact string match
            if str(parsed_val).strip() != str(expected_val).strip():
                return False
    
    return True

async def check_categorical_correctness(parsed_value: str | float, expected_value: str | float,
                                  parsed_aspect: str, expected_aspect: str,
                                  model) -> bool:
    """
    Check if a categorical/string answer matches the expected answer.
    Handles common variations in text answers.
    """
    # Normalize both values
    parsed_str = str(parsed_value).lower().strip()
    expected_str = str(expected_value).lower().strip()
    
    # Try exact match first
    if parsed_str == expected_str:
        return True
    
    # For counts, check if they're equal as numbers
    try:
        parsed_num = float(parsed_str)
        expected_num = float(expected_str)
        return abs(parsed_num - expected_num) < 1e-6
    except (ValueError, TypeError):
        pass
    
    if SequenceMatcher(None, parsed_str, expected_str).ratio() >= 0.7:
        return True
    
    # Fall back to LLM judge for ambiguous text comparisons
    # This handles cases like "same" vs "same lesion" that the parsers might produce
    answer_check_prompt = f"""
    You are checking if a parsed answer from a response matches the expected answer for a specific aspect.
    Do not pay attention to minor wording differences or typos, as long as the meaning is correct. Consider the context of the aspect being asked about.
    If there are "or" statements or multiple parts in the expected answer, the parsed answer can match any one of them to be considered correct.
    The query aspect is: "{expected_aspect}" and the expected answer is: "{expected_str}".
    The parsed aspect from the response is: "{parsed_aspect}" and the corresponding answer is: "{parsed_str}".
    Do the answers match? Should we consider this as a correct case?
    """

    class CorrectnessOutput(BaseModel):
        is_correct: bool = Field(..., description="Whether the parsed answer matches the expected answer for the aspect")

    answer_check_agent = Agent(
            name="answer_check_agent",
            model=model,
            output_type=CorrectnessOutput,
            # model_settings=ModelSettings(temperature=0.0),
        )

    result = await Runner.run(answer_check_agent, input=answer_check_prompt)

    return result.final_output.is_correct



class HybridResponseJudge(weave.Scorer):
    """
    Hybrid approach: 
    1. LLM parses BOTH the ground truth list AND the response into structured QA pairs
       (handles complex contextual parsing like "lesion 1: enhancing: 284 mm3, non-enhancing: 11 mm3")
    2. Deterministic code validates correctness by matching and comparing the parsed pairs
       (handles unit conversion, numerical tolerance, text matching)
    
    This addresses reviewer concerns by making the correctness checking
    fully reproducible and transparent, while leveraging LLM's strength in parsing.
    """
    model_id: str = os.environ.get("LLM_JUDGE", "gpt-5.1")
    tolerance_pct: float = 0.1  # Fractional tolerance for numerical values (0-1 scale, 0.1 = 10%)
    fuzzy_threshold: float = 0.98  # Threshold for fuzzy aspect matching
    count_tolerance_pct: float = 0.0  # No tolerance for counts
    gt_cache_file: str | Path | None = None  # Path to JSON file for caching parsed ground truth QAs
    _gt_cache: dict = PrivateAttr(default_factory=dict)  # In-memory cache of parsed GT QAs
    
    gt_parser_prompt: str = """
    You are parsing a list of expected question-answer pairs from a ground truth dataset.
    You will receive the <query> for which the question-answer pairs are expected. Use the query to better understand the context of the expected answers.
    
    Each item in the list may contain one or more QA pairs. Your job is to extract ALL individual QA pairs
    while preserving the context (which lesion, which timepoint, etc.).
    
    For each QA pair, extract:
    1. The complete aspect/measurement including all context (e.g., "lesion 1 enhancing tumor volume at timepoint 0")
    2. The value (numerical or categorical)
    3. The unit if specified - INFER from context if missing (see guidelines below)
    
    CRITICAL INSTRUCTIONS:
    - Preserve ALL context when splitting multi-QA items (lesion number, timepoint, etc.)
    - If units are missing, INFER the most likely unit from context:
      * Tumor/lesion volumes: typically mm3 (range: 100-100,000 mm3 = 0.1-100 cm3)
      * Tumor/lesion diameters: typically mm (range: 5-100 mm = 0.5-10 cm)
      * Brain region volumes: typically mm3 (range: 1,000-100,000 mm3)
      * Percentage values (growth, change, increase/decrease): use "percent" or "%" unit
      * Coordinate data (bbox, centroid): use "coordinates" as unit, value is a list
      * If value is < 500 and aspect mentions "diameter" or "size": likely mm
      * If value is > 500 and aspect mentions "volume": likely mm3
      * If aspect mentions "increase", "decrease", "growth", "change" and value is small: likely percent
    - Only set unit to null if truly ambiguous or non-numeric (and not coordinates)
    - For coordinate data (bounding boxes, centroids):
      * Extract as a list/array of numbers: e.g., [129, 79, 47, 174, 144, 117] for bbox
      * Bounding boxes typically have 6 values (x1, y1, z1, x2, y2, z2)
      * Centroids typically have 3 values (x, y, z)
      * Use "coordinates" as the unit
    - For "decrease/change" aspects, include the sign of the number (e.g., "1294 mm3 decrease" → value: -1294, unit: "mm3")
    - If the unit is numeric (mm3, mm), DO NOT put non-numeric text in the value field. (e.g., "approximately 16,000 mm3" → value: 16000, unit: "mm3") 
    - Each QA must be atomic and fully qualified
    - Handle complex items like "lesion 1: enhancing tumor volume: 284.0 mm3, non-enhancing tumor volume: 11.0 mm3" → Extract as two separate QAs, both for "lesion 1"
    
    Example Input 1 (with explicit units):
    ["lesion 1: enhancing tumor volume: 284.0 mm3, non-enhancing tumor volume: 11.0 mm3, location: frontal lobe"]
    
    Example Output 1:
    {
        "parsed_qas": [
            {
                "aspect": "lesion 1 enhancing tumor volume",
                "value": 284.0,
                "unit": "mm3",
            },
            {
                "aspect": "lesion 1 non-enhancing tumor volume",
                "value": 11.0,
                "unit": "mm3",
            },
            {
                "aspect": "lesion 1 location",
                "value": "frontal lobe",
                "unit": null,
            }
        ],
    }
    
    Example Input 2 (missing units - infer from context & coordinate data):
    ["lesion 1: tumor core volume: 4059, maximum diameter: 23.5, volume increase: 15",
     "lesion 1: whole lesion bbox: (129, 79, 47, 174, 144, 117)", "lesion 1: centroid: (150, 110, 80)"]
    
    Example Output 2:
    {
        "parsed_qas": [
            {
                "aspect": "lesion 1 tumor core volume",
                "value": 4059.0,
                "unit": "mm3",
            },
            {
                "aspect": "lesion 1 maximum diameter",
                "value": 23.5,
                "unit": "mm",
            },
            {
                "aspect": "lesion 1 volume increase",
                "value": 15.0,
                "unit": "percent",
            },
            {
                "aspect": "lesion 1 whole lesion bbox",
                "value": [129, 79, 47, 174, 144, 117],
                "unit": "coordinates",
            },
            {
                "aspect": "lesion 1 centroid",
                "value": [150, 110, 80],
                "unit": "coordinates",
            }
        ],
    }
    """
    
    response_parser_prompt: str = """
    You are a parser that extracts question-answer pairs from a free-text response.
    
    You will receive:
    1. A <query>
    2. A <response> (free-text answer to the query)
    3. A list of <expected_aspects> from ground truth (what we're looking for)
    
    Your job: For each expected aspect, find if and how it was addressed in the <response>.
    
    IMPORTANT INSTRUCTIONS:
    - For each expected aspect, search for the corresponding information in the response
    - Prefer the closest semantic aspect from the expected list, but do NOT force lesion index identity (e.g., lesion 2 in expected can map to lesion 3 in response if measurements/context match)
    - Extract the value and unit as stated in the response (may differ from GT)
    - Preserve the exact numerical values as stated in the response
    - Extract units exactly as written (mm3, cm3, ml, %, percent, etc.)
    - For percentages, use "percent" or "%" as the unit
    - For coordinate data (bounding boxes, centroids):
      * Extract as a list of numbers: [129, 79, 47, 174, 144, 117] or [150, 110, 80]
      * Use "coordinates" as the unit
      * Parse from formats like "(129, 79, 47, 174, 144, 117)" or "[129, 79, 47, 174, 144, 117]"
    - For counts (e.g., "number of lesions"), the unit should be null
    - If an expected aspect is NOT mentioned in the response, skip it (don't include it)
    - If the response mentions additional aspects not in the expected list, include them too. These are important to count hallucinations
    - For "decrease/change" aspects, include the sign of the number (e.g., "1294 mm3 decrease" → value: -1294, unit: "mm3")
    - If the unit is numeric (mm3, mm), DO NOT put non-numeric text in the value field. (e.g., "approximately 16,000 mm3" → value: 16000, unit: "mm3") 
      * If answer says value is unknown or not specified, keep the unit but set the value to 0.0 (e.g., "couldn't find the tumor volume" → value: 0.0, unit: "mm3")
    - Ignore file names in the response
    - Return parsed QAs in the same order as the expected aspects 
    - Do NOT check correctness - just parse what is stated in the response
    - For text answers (e.g., "location: frontal lobe"), extract the text value
    
    Example 1:
    <query>
    "How many lesions does this glioma patient have? What are their tumor core volumes and centroid coordinates?"
    </query>
    <expected_aspects>
    ["number of lesions", "cumulative total tumor core volume across all lesions"]
    </expected_aspects>
    <response>
    "This glioma patient has 1 lesion. Its tumor core volume is 16.134 cm3 and its centroid is at (150, 110, 80)."
    </response>
    
    Output:
    {
        "parsed_qas": [
            {
                "aspect": "number of lesions",
                "value": 1,
                "unit": null,
            },
            {
                "aspect": "lesion 1 tumor core volume",
                "value": 16.134,
                "unit": "cm3",
            },
            {
                "aspect": "lesion 1 centroid",
                "value": [150, 110, 80],
                "unit": "coordinates",
            }
        ],
    }
    
    Example 2 (with percentage and missing aspect):
    <query>
    "How much did the tumor grow between timepoints?"
    </query>
    <expected_aspects>
    ["tumor volume increase", "tumor diameter increase"]
    </expected_aspects>
    <response>
    "The tumor volume increased by 20% between the two timepoints."
    </response>
    
    Output:
    {
        "parsed_qas": [
            {
                "aspect": "tumor volume increase",
                "value": 20.0,
                "unit": "percent",
            }
        ],
    }
    
    """
    
    def __init__(self, **data):
        super().__init__(**data)

        model = create_openai_model(self.model_id)
        
        # Agent for parsing ground truth
        self._gt_parser = Agent(
            name="gt_parser_agent",
            model=model,
            instructions=self.gt_parser_prompt,
            output_type=JudgeParseOutput,
            # model_settings=ModelSettings(temperature=0.0),
        )
        
        # Agent for parsing response
        self._response_parser = Agent(
            name="response_parser_agent",
            model=model,
            instructions=self.response_parser_prompt,
            output_type=JudgeParseOutput,
            # model_settings=ModelSettings(temperature=0.0),
        )        
        # Load GT cache if file exists
        if self.gt_cache_file and Path(self.gt_cache_file).exists():
            with open(self.gt_cache_file, 'r') as f:
                self._gt_cache = json.load(f)
                print(f"Loaded GT cache with {len(self._gt_cache)} entries from {self.gt_cache_file}")
        elif self.gt_cache_file:
            self.gt_cache_file = Path(self.gt_cache_file)
            self.gt_cache_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._gt_cache = {}
    
    @weave.op
    async def score(self, sample_id: str, llm_scorer_input: list, output: dict) -> dict:
        """
        Score a response by:
        1. Using LLM to parse the ground truth list into structured QA pairs (cached for consistency)
        2. Using LLM to parse the response into structured QA pairs
        3. Using deterministic code to match and check correctness
        """
        if isinstance(llm_scorer_input, weave.ObjectRef):
            llm_scorer_input = llm_scorer_input.get()
        if isinstance(sample_id, weave.ObjectRef):
            sample_id = sample_id.get()
        
        # # Extract sample_id from llm_scorer_input (last element)
        # sample_id = llm_scorer_input[-1] if len(llm_scorer_input) > 0 and isinstance(llm_scorer_input[-1], str) and not llm_scorer_input[-1].startswith("\n") else None
        
        response = output.get("response")
        
        # Extract query and expected answers from llm_scorer_input
        query_section = llm_scorer_input[0]  # Contains <query>...</query>
        expected_answers_raw = llm_scorer_input[1] if len(llm_scorer_input) > 1 else ""
        
        # Extract expected answers string
        match = re.search(r'<expected answers>\s*\n(.*?)\n</expected answers>', 
                                   expected_answers_raw, re.DOTALL)
        if match:
            expected_str = match.group(1).strip()
        else:
            expected_str = "[]"
        
        # Check if GT is cached for this sample (if sample_id is available)
        if sample_id and sample_id in self._gt_cache:
            # Load from cache (convert dicts back to ParsedQA objects)
            expected_qas = [ParsedQA(**qa) for qa in self._gt_cache[sample_id]]
        else:
            # Parse GT with LLM and cache it
            gt_prompt = f"{query_section}\n\nThe ground truth list for the query to parse:\n{expected_str}"
            gt_result = await Runner.run(
                self._gt_parser,
                gt_prompt,
            )
            expected_qas = gt_result.final_output.parsed_qas
            
            # Save to cache (only if sample_id is available)
            if sample_id:
                self._gt_cache[sample_id] = [qa.model_dump() for qa in expected_qas]
                
                # Write cache to file if path is configured
                if self.gt_cache_file:
                    cache_path = str(self.gt_cache_file)
                    with open(cache_path, 'w') as f:
                        json.dump(self._gt_cache, f, indent=2)
        
        total_expected = len(expected_qas)
        
        # Extract expected aspects to condition the response parser
        expected_aspects = [eq.aspect for eq in expected_qas]
        
        # Use LLM to parse the response (conditioned on expected aspects)
        response_prompt = f"{query_section}\n<expected_aspects>\n{expected_aspects}\n</expected_aspects>\n<response>\n{response}\n</response>"
        response_result = await Runner.run(
            self._response_parser,
            response_prompt,
        )
        
        parsed_qas = response_result.final_output.parsed_qas
        
        # Deterministic matching and validation
        total_included = 0
        total_responded_correctly = 0
        matched_expected_indices = []  # List of (parsed_idx, expected_idx) tuples
        matched_gt_indices = set()  # Set of GT indices that have been matched (for unique matching)
        
        for j,parsed_qa in enumerate(parsed_qas):
            # Convert parsed value to base units once (before matching)
            base_value_ans, base_unit_ans = convert_to_base_unit(parsed_qa.value, parsed_qa.unit)
            
            # Find matching expected QA (with unique matching: each GT QA matched at most once)
            # Returns already-converted expected values to avoid redundant conversion
            idx, matching_expected = find_matching_expected_qa(
                parsed_qa.aspect,
                base_value_ans,
                base_unit_ans,
                [{"aspect": eq.aspect, "value": eq.value, "unit": eq.unit} for eq in expected_qas],
                matched_gt_indices,
                threshold=self.fuzzy_threshold
            )
            
            if matching_expected:
                total_included += 1
                matched_expected_indices.append((j, idx))
                matched_gt_indices.add(idx)  # Mark this GT QA as matched

                base_value_exp = matching_expected["value"]
                base_unit_exp = matching_expected["unit"]
                
                # Check correctness deterministically (using already-converted values)
                is_correct = False
                
                # Route to appropriate comparison function based on converted types
                # 1. Check if this is coordinate data (lists)
                if base_unit_exp == "coordinates": #(isinstance(base_value_ans, list) and isinstance(base_value_exp, list)) or base_unit_exp == "coordinates":
                    # Coordinate/list comparison (bbox, centroid)
                    is_correct = check_coordinate_correctness(
                        base_value_ans,
                        base_value_exp,
                        tolerance_mm=5
                    )
                
                # 2. Check if both are numeric after conversion
                elif isinstance(base_value_ans, (int, float)) and isinstance(base_value_exp, (int, float)):
                    # Numerical comparison
                    
                    # will not use tolerance if unitless
                    is_correct = compare_converted_values(
                        base_value_ans, base_unit_ans,
                        base_value_exp, base_unit_exp,
                        tolerance_pct=self.tolerance_pct
                    )
                
                # 3. Fall back to categorical comparison for strings and mixed types
                else:
                    # Categorical comparison (handles text answers)
                    is_correct = await check_categorical_correctness(
                        parsed_qa.value, matching_expected["value"],
                        parsed_qa.aspect, matching_expected["aspect"],
                        model=self.model_id
                    )
                
                if is_correct:
                    total_responded_correctly += 1
                else:
                    stop=1

        # Calculate metrics
        include_rate = min(1.0, total_included / max(1.0, total_expected))
        correct_response_acc = min(1.0, total_responded_correctly / max(1.0, total_expected))
        correct_response_precision = min(1.0, total_responded_correctly / max(1.0, total_included)) if total_included > 0 else 0.0
        
        
        num_hallucinations = len(parsed_qas) - total_included
        hallucination_rate = num_hallucinations / max(1.0, len(parsed_qas))

        # Calculate quality score (weighted combination)
        # Quality considers: coverage (30%), correctness (50%), no hallucinations (20%)
        quality_score = max(0.0, 0.3 * include_rate + 0.5 * correct_response_acc + 0.2 * (1 - hallucination_rate))
        
        return {
            "total_queried": total_expected,
            "total_included": total_included,
            "total_responded_correctly": total_responded_correctly,
            # "num_parsed_qas": len(parsed_qas),
            "hallucination_rate": hallucination_rate,
            "response_inclusion_rate": include_rate,
            "correct_response_acc": correct_response_acc,
            # "correct_response_acc_raw": total_responded_correctly / max(1.0, total_expected),
            "correct_response_precision": correct_response_precision,
            "quality_score": quality_score,
            # Store parsed data for debugging
            # "parsed_qas_debug": [qa.model_dump() for qa in parsed_qas],
            # "expected_qas_debug": [qa.model_dump() for qa in expected_qas],
        }


# EXTRACTION_SYS_PROMPT = """
# Given a <text>, extract all the unique entities from the text with or without repetition. Repetition is required when the same entity is mentioned multiple times in different contexts (e.g., different lesions or timepoints).
# """

# EXTRACTION_PROMPT = """
# Extract all the unique entities from the following <text>:
# <text>
# {text}
# </text>
# """
# EVAL_SYS_PROMPT="""
# You are evaluating the response of an agentic pipeline. The pipeline just receives the initial <query> and produces a final <response> using various tools.
# You will receive the <query> along with a set of raw information from the ground truth database of measurements inside the <input>, and the pipeline's response.
# Evaluate the quality of the <response>, in which you consider whether the pipeline has achieved the desired outcome and correctly presented all the relevant information in a concise and informative manner.
# The judgement should be based on the entities that are relevant to the query, which may be explicitly mentioned in the <input> or inferred from it.
# The entities should be counted separately for multiple lesions and/or timepoints.

# # Evaluation considerations
# - Does the <response> contain all the key information relevant to the <query> in the <input>?
# - Is the <response> concise and informative?
# - Does the <response> correctly present the relevant entities without numerical or factual errors?
# - Does the <response> contain information or assertions that are not present in the <input>?

# # Scoring Rubric

# `1.0`: The <response> contains all of the key information and entities extracted from the <input>, is concise and informative, and doesn't contain any information or assertions that are not relevant to the <query>. All the answers are numerically and factually correct. 
# '0.75': The <response> reached the desired outcome of the <query>. It contains most of the key information and entities extracted from the <input>, is somewhat concise and informative, and doesn't contain any information or assertions that are not relevant to the <query>. Most answers are numerically and factually correct.
# `0.5`: The <response> contains some of the key information and entities extracted from the <input>, but misses some of the entities. It is somewhat concise and informative.

# `0.0`: The <response> failed to reach a reasonable outcome and misses most or all of the key information in the <input>, or contains information or assertions that are not relevant to the <input>.
# """
# EVAL_PROMPT="""
# Evaluate the quality of the following <response> given the <input>:

# <input>
# {input}
# </input>

# <response>
# {summary}
# </response>
# """


# # we need a wrapper class to override the score method
# class SummarizationScorer(SummarizationScorer):
#     model_id: str = os.environ.get("LLM_JUDGE", "o3")
#     extraction_system_prompt: str = EXTRACTION_SYS_PROMPT
#     extraction_prompt: str = EXTRACTION_PROMPT
#     summarization_evaluation_system_prompt: str = EVAL_SYS_PROMPT
#     summarization_evaluation_prompt: str = EVAL_PROMPT
#     column_map: dict = {
#         "input": "llm_scorer_input",
#     }
#     @weave.op
#     async def score(self, input: str, output: dict) -> dict:
#         response = output.get("response", "")
        # return await super().score(input=input, output=response)



@weave.op
def preprocess_data(sample: dict, framework="single", task=1, reg_check=False) -> dict:
    """
    framework: "single" | "as_tools" | "handoffs"
    """
    prompt = sample["prompt"]
    match framework:
        case "single":
            gt_tools = sample.get("expected_tools", [])
        case "as_tools":
            gt_tools = sample.get("expected_tools_multi", [])
        case "handoffs":
            gt_tools = sample.get("expected_tools_w_handoffs", [])
        case "orchestrator":
            gt_tools = sample.get("expected_tools_w_orchestrator", [])
        case _:
            raise ValueError(f"Unknown framework: {framework}")

    

    patients = []   
    for d in sample["data"]:
        patient, msg = instantiate_patient(**d)
        if reg_check and ("tumor" in sample["id"] or "combined" in sample["id"]):

            if "pre-op" in prompt.lower():
                timepoint = "preop"
            elif "post-op" in prompt.lower():
                timepoint = "postop"
            if "metastasis" in prompt.lower():
                disease = "mets"
            elif "glioma" in prompt.lower():
                disease = "glioma"

            disease_str = disease + "_" + timepoint

            msg += (" " + check_if_registered_no_tool(patient,disease=disease_str))
        
        if "aslbb" in sample["id"] or 'sfb' in sample['id']:
            msg += " The input scans are not skullstripped."
        # else: 
        #     msg += " The input scans are skullstripped."
        patients.append(patient)
        prompt += f"\n{msg}"
    
    all_information = [f"<query>\n{prompt}\n</query>"]
   
    # if task > 1:
    #     measurement_files = sample.get("tumor_measurements_file")
    #     if isinstance(measurement_files, list):
    #         measurements = str({f"Timepoint {str(i+1)}": json.load(open(m, "r")) for i, m in enumerate(measurement_files)})
    #     else:
    #         measurements = str(json.load(open(measurement_files, "r")))

    #     synthseg_files = sample.get("synthseg_volumes_file")
    #     if isinstance(synthseg_files, list):
    #         synthseg_volumes = str({f"Timepoint {str(i+1)}": pd.read_csv(m).drop(columns="subject").iloc[0].to_dict() for i, m in enumerate(synthseg_files)})    
    #     else:
    #         synthseg_volumes = str(pd.read_csv(synthseg_files).drop(columns="subject").iloc[0].to_dict())
    #     all_information += "\n Ground truth database:"

    #     if "tumor" in sample or "combined" in sample:
    #         all_information += f"\nTumor measurements: {measurements}"
    #     if "brain_region" in sample or "combined" in sample:
    #         all_information += f"\nBrain region volumes: {synthseg_volumes}"

    contents = sample.get("expected_contents")
    if contents:
        num_contents = len(contents)
        all_information.append(f"\n<expected answers>\n{list(contents)}\n</expected answers>")
        # all_information.append(num_contents)
    
    # # Add sample_id to llm_scorer_input for GT caching
    # all_information.append(sample["id"])
    
    return {
        "sample_id": sample["id"],
        "prompt": prompt,
        "gt_tools": gt_tools,
        "patients": patients,
        "llm_scorer_input": all_information, ## named input to match the Weave SummarizationScorer expected keys
    }

class MonitorUsageHook(RunHooks):
    
    def __init__(self, requests_per_minute=25, input_tokens_per_minute=30_000, output_tokens_per_minute=8_000, **kwargs):
        super().__init__()
        self.tpm_input = input_tokens_per_minute
        self.tpm_output = output_tokens_per_minute
        self.rpm = requests_per_minute  
        self.wait_time = 45
        self.last_seen_requests = 0
        self.last_seen_inputs = 0
        self.last_seen_outputs = 0
        self.usage_history = []

    def _get_last_minute_usage(self):
        now = time.time()
        self.usage_history = [u for u in self.usage_history if now - u[0] < 60]

        return sum(u[1] for u in self.usage_history), sum(u[2] for u in self.usage_history), sum(u[3] for u in self.usage_history) #num_requests, num_input_tokens, num_output_tokens
    
    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, output: Any) -> None:
        requests = context.usage.requests - self.last_seen_requests
        input_tokens = context.usage.input_tokens - self.last_seen_inputs
        output_tokens = context.usage.output_tokens - self.last_seen_outputs
        
        self.last_seen_inputs = context.usage.input_tokens
        self.last_seen_outputs = context.usage.output_tokens
        self.last_seen_requests = context.usage.requests

        # if self.last_seen_requests >  5:
        #     aa = 5
        self.usage_history.append((time.time(), requests, input_tokens, output_tokens))

    async def on_tool_start(self, context, agent, tool):
        num_requests, num_input_tokens, num_output_tokens = self._get_last_minute_usage()

        if num_requests >= (self.rpm-1) or num_input_tokens >= (self.tpm_input-5000) or num_output_tokens >= (self.tpm_output-1000):  # add some buffer to avoid hitting the limit
            print(f"Rate limit approaching: {num_requests} requests, {num_input_tokens} input tokens, {num_output_tokens} output tokens in the last minute. Waiting for {self.wait_time} seconds...")
            await asyncio.sleep(self.wait_time)




class AgentWeaveWrapper(weave.Model):
    _agent = PrivateAttr()
    _stats = PrivateAttr()
    _usage_hook = PrivateAttr()
    agent_type: str = None
    llm: str = None
    output_log_file: str = None
    output_dir: str = None

    def __init__(self, agent: Agent,agent_type: str = None, output_log_file: str = None, output_dir: str = None, **kwargs):
        super().__init__(**kwargs)
        self._agent = agent
        self.agent_type = agent_type
        self.llm = agent.model.model
        self._usage_hook = MonitorUsageHook(**LLM_RATE_LIMITS.get(self.llm, {}))
        self.output_log_file = output_log_file
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "outputs" / self.llm / self.agent_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reset_stats()

        if self.llm == "claude-sonnet-4-0":
            self.llm = "claude-sonnet-4-20250514"
        elif self.llm == "claude-sonnet-4-5":
            self.llm = "claude-sonnet-4-5-20250929"

    @weave.op
    async def predict(self, sample_id: str, prompt: str, patients: list, llm_scorer_input: str, gt_tools: list) -> dict:
        ## need to put llm_scorer_input and expected_tools here, otherwise weave backhand removes them and they are not available for the scorer

        session = SQLiteSession(session_id=sample_id)
        session_context = SessionContext()
        session_context["output_dir"] = self.output_dir

        for patient in patients:
            session_context[patient.name] = patient

        i = 0
        while i < 2:
            try:
                # asyncio.sleep(30)
                result = await Runner.run(
                    self._agent,
                    prompt,
                    session=session,
                    context=session_context,
                    max_turns=40, # we need especially more with handoffs API (each handoff is a new turn)
                    hooks=self._usage_hook) 
            
                if result and result.final_output != "":
                    break

            except Exception as e:
                error_msg = str(e).lower()
                if 'ratelimit' not in error_msg and 'rate limit' not in error_msg and 'resource' not in error_msg and '429' not in error_msg and 'quota' not in error_msg:
                    raise e
                else:
                    await asyncio.sleep(60)  # wait for 60 seconds before retrying 
            i += 1
            
        
        # Give Weave backend 2s to populate 'summary' on the last child node
        await asyncio.sleep(2)
        call = weave.get_current_call()
        self._analyze_trace_tree(call)
        stats = self._stats
        self._reset_stats()

        out = {"response": result.final_output, "stats": stats}
        save_out = deepcopy(out)
        save_out["case_id"] = sample_id

        if self.output_log_file:
            # Read existing data if file exists
            if os.path.exists(self.output_log_file):
                with open(self.output_log_file, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = {}
            else:
                existing_data = {}
                  
            existing_data[sample_id] = save_out
        
            # Write back to file
            with open(self.output_log_file, 'w') as f:
                json.dump(existing_data, f, indent=4)

        return out


    def _analyze_trace_tree(self,call):
        
        is_llm_call = call.op_name and (
            "openai.responses.create" in call.op_name or 
            "chat.completions.create" in call.op_name
        )

        is_handoff_op = call.op_name and "openai_agent_handoff" in call.op_name

        if is_llm_call:
            # get costs
            if hasattr(call, "summary") and call.summary:
                if "usage" in call.summary:
                    usage = call.summary["usage"]
                    key = [x for x in usage.keys() if self.llm in x][0]  # e.g., "gpt-4o"
                    # Safely add, defaulting to 0 if key is missing
                    if "completions" in call.op_name:
                        input_tokens = usage[key].get("prompt_tokens", 0)
                        output_tokens = usage[key].get("completion_tokens", 0)
                        cached_tokens = usage[key].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    else:
                        input_tokens = usage[key].get("input_tokens", 0)
                        output_tokens = usage[key].get("output_tokens", 0)
                        cached_tokens = usage[key].get("input_tokens_details", {}).get("cached_tokens", 0) ## no cached tokens for completions API ?
                    self._stats["total_input_tokens"] += input_tokens
                    # self._stats["input_tokens_cached"] += cached_tokens
                    # self._stats["input_tokens_new"] += (input_tokens - cached_tokens)
                    self._stats["total_output_tokens"] += output_tokens
                    self._stats["total_tokens"] += (input_tokens + output_tokens)
                    self._stats["estimated_cost"] += self._get_costs(input_tokens, cached_tokens, output_tokens)
                    self._stats["llm_request_count"] += usage[key].get("requests", 0)

        if is_handoff_op:
            info = call.output.get("metadata", None)
            if info and isinstance(info, dict):
                from_agent = info.get("from_agent", "unknown")
                to_agent = info.get("to_agent", "unknown")
                self._stats["executed_tools"].append({
                    "name": f"{from_agent}_to_{to_agent}",
                    "is_handoff": True,
                    "is_error": False
                })

        # inspect tool calls
        if call.op_name and "openai_agent_function" in call.op_name:
            inputs = dict(call.inputs) if call.inputs else {}
            tool_name = inputs.get("name", "unknown")
            self._stats["tool_call_count"] += 1

            # Detect Error
            is_error = False
            # Check for Python exceptions captured by Weave
            if hasattr(call, "exception") and call.exception:
                is_error = True
            # Check for explicit error keys in output
            elif hasattr(call, "output"):
                out = call.output
                if isinstance(out, dict) and out.get("error"):
                    is_error = True
            if is_error:
                self._stats["error_count"] += 1
            # Detect Handoff
            # Heuristic: if name contains 'assistant' or 'agent', it is a nested agent call
            is_handoff = "assistant" in tool_name or "agent" in tool_name

            self._stats["executed_tools"].append({
                "name": tool_name,
                # "arguments": inputs,
                "is_handoff": is_handoff,
                "is_error": is_error
            })

            if "assistant" in tool_name or "agent" in tool_name:
                if hasattr(call, "output") and isinstance(call.output, dict):
                    # Direct String Access (Matches Child[2])
                    output_val = call.output.get("output")
                    
                    if isinstance(output_val, str) and output_val.strip():
                        # Save this! This is the intermediate update.
                        self._stats["llm_responses"].append(f"{tool_name}: {output_val}")
            

        # --- RECURSE ---
        if hasattr(call, "children"):
            for child in call.children():
                self._analyze_trace_tree(child)

    
    def _reset_stats(self):
        self._stats = {
            "total_input_tokens": 0,
            # "input_tokens_cached": 0,
            # "input_tokens_new": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "llm_request_count": 0,
            "tool_call_count": 0,
            "error_count": 0,
            "estimated_cost": 0.0,
            "executed_tools": [],
            "llm_responses": [], # Contains all intermediate LLM responses
        }


    def _get_costs(self, input_tokens, cached_tokens, output_tokens):
        # new_tokens = input_tokens - cached_tokens
        cost = (input_tokens * LLM_PRICES[self.llm]["input"]) + \
               (cached_tokens * LLM_PRICES[self.llm]["cached"]) + \
               (output_tokens * LLM_PRICES[self.llm]["output"])
        return round(cost, 6)
    

class ReplayRun(weave.Model):
    _stats = PrivateAttr()
    llm: str = None
    run_trace: dict = None

    def __init__(self, run_trace_file: str, llm: str, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.run_trace = json.load(open(run_trace_file, "r"))
        # self._reset_stats()
        

    @weave.op
    async def predict(self, sample_id: str, prompt: str, patients: list, llm_scorer_input: str, gt_tools: list) -> dict:
        ## need to put llm_scorer_input and expected_tools here, otherwise weave backhand removes them and they are not available for the scorer
        
        trace = self.run_trace.get(sample_id)
        # if trace:
        stats = trace["stats"]
        response = trace["response"]

        out = {"response": response, "stats": stats}
        # save_out = deepcopy(out)
        # save_out["case_id"] = sample_id

        return out

# def _analyze_trace_tree(node):
#     """
#     Traverses a Weave trace tree to extract token usage, cost, 
#     tool executions, and handoffs in a single pass.
#     """
    
#     # Calculate Est. Cost (Optional: Example using GPT-4o pricing)
#     # Input: $2.50 / 1M, Output: $10.00 / 1M
#     cost = (stats["total_input_tokens"] / 1_000_000 * 2.50) + \
#            (stats["total_output_tokens"] / 1_000_000 * 10.00)
#     stats["estimated_cost"] = round(cost, 6)
    
#     return stats

# def _print_node_structure(node, depth=0):
#     indent = "  " * depth
    
#     # Determine node type (simplified)
#     node_type = node.op_name if hasattr(node, "op_name") else "Unknown"
    
#     print(f"{indent}▶ Node: {node_type}")
    
#     # INSPECT INPUTS
#     if hasattr(node, "inputs") and isinstance(node.inputs, dict):
#         print(f"{indent}  - Inputs Keys: {list(node.inputs.keys())}")
#         # If this is a tool execution, the arguments are often here:
#         if "messages" not in node.inputs: # Filter out massive chat history
#             print(f"{indent}  - Input Values (Preview): {str(node.inputs)[:100]}")

#     # INSPECT OUTPUT
#     if hasattr(node, "output"):
#         out = node.output
#         if isinstance(out, dict):
#             print(f"{indent}  - Output Keys: {list(out.keys())}")
            
#             # CHECK "output" KEY specifically
#             if "output" in out:
#                 inner = out["output"]
#                 print(f"{indent}    - output['output'] Type: {type(inner)}")
#                 if isinstance(inner, dict):
#                     print(f"{indent}    - output['output'] Keys: {list(inner.keys())}")
#                 elif isinstance(inner, str):
#                     print(f"{indent}    - output['output'] String (Preview): {inner[:50]}...")
#                 elif hasattr(inner, '__dict__'):
#                     print(f"{indent}    - output['output'] Attributes: {dir(inner)}")
#         else:
#             print(f"{indent}  - Output Type: {type(out)}")

#     # RECURSE<
#     if hasattr(node, "children"):
#         for child in node.children():
#             _print_node_structure(child, depth + 1)

