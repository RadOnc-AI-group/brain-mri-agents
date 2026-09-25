# Brain MRI Agents: Usage Guide

<!-- [← Back to Overview](README.md) • [Usage Guide](USAGE.md) • [Datasets](dataset/README.md)

--- -->

## 🧠 Overview

This guide provides detailed instructions on running interactive neuroimaging analysis sessions (`run_single.py`), configuring custom patient input scans (`patient_files.json`), and executing batch benchmark evaluations (`weave_evaluation.py`).

---

## 🤖 Interactive Single-Session (`run_single.py`)

Run an interactive multi-agent session on your MRI scans:

```bash
python run_single.py --patient-files-json patient_files.json --agent-type as_tools --llm gpt-5.4
```

### 1. Input Configuration: `patient_files.json`

`run_single.py` reads scan paths and imaging metadata from a JSON configuration file (default: `patient_files.json`). The file maps unique patient identifiers (`<patient_name>`) to their scan paths and properties. Fill the file with the desired image paths from your work directory.

#### Schema

```json
{
  "patient_001_01": {
    "t1": "patient_001_t1.nii.gz",
    "t1c": "patient_001_t1c.nii.gz",
    "t2": "patient_001_t2.nii.gz",
    "flair": "patient_001_flair.nii.gz",
    "skullstripped": true
  }
}
```

#### Fields Description

| Field | Type | Required | Description |
|---|---|---|---|
| `<patient_name>` | `string` (key) | Yes | Unique identifier for the patient or timepoint in the session context (e.g., `patient_001_01`, `patient_stp0`). |
| `t1` | `string` | Yes | Path to the T1-weighted MRI scan (`.nii` or `.nii.gz`). |
| `t1c` | `string` | Optional* | Path to the contrast-enhanced T1 (T1CE/T1C) MRI scan (`.nii` or `.nii.gz`). Required for optimal performance. |
| `t2` | `string` | Optional* | Path to the T2-weighted MRI scan (`.nii` or `.nii.gz`). Required for optimal performance. |
| `flair` | `string` | Optional* | Path to the FLAIR MRI scan (`.nii` or `.nii.gz`). Required for optimal performance. |
| `skullstripped` | `boolean` | Yes | `true` if the scans are already skullstripped, `false` if they include skull/extracranial tissue. |
| `seg` | `string` | Optional | Path to pre-existing tumor/lesion segmentation mask, if you want to skip segmentation. |
| `brain_regions`| `string` | Optional | Path to pre-existing anatomical brain region segmentation, if you want to skip segmentation. |

*\*Note: If any optional MRI modality is omitted or not found, placeholder images filled with zeros are created automatically. This will reduce segmentation performance.*

#### Multiple Patients or Longitudinal Timepoints

You can define multiple patients or longitudinal timepoints in `patient_files.json`. All entries will be loaded into the agent's context simultaneously:

```json
{
  "patient_001_tp0": {
    "t1": "patient_001_tp0_t1.nii.gz",
    "t1c": "patient_001_tp0_t1c.nii.gz",
    "t2": "patient_001_tp0_t2.nii.gz",
    "flair": "patient_001_tp0_flair.nii.gz",
    "skullstripped": true
  },
  "patient_001_tp1": ...
}
```

---

### 2. CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--patient-files-json` | `str` | `patient_files.json` | Path to JSON file containing patient scan paths. |
| `--agent-type` | `str` | `single` | Architecture to use: `single`, `as_tools`, or `handoffs`. |
| `--prompt` | `str` | `None` | Initial clinical question/prompt (if omitted, you will be prompted interactively). |
| `--llm` | `str` | `gpt-5.4` | LLM model name. |
| `--temperature` | `float` | `0.0` | Sampling temperature for LLM responses. |
| `--no-weave` | `flag` | `False` | Disable Weights & Biases / Weave telemetry. |

#### Agent Architectures (`--agent-type`):
- `single`: Single monolithic agent equipped directly with all image-processing, segmentation, and analysis tools.
- `as_tools`: Analysis agent as orchestrator that invokes specialized sub-agents (preprocessing, segmentation) as tools.
- `handoffs`: Collaborative multi-agent workflow using dynamic agent-to-agent handoffs.

---

### 3. Interactive Terminal Commands

During an interactive session, type your clinical question or command at the `👤 You:` prompt:

| Command | Description |
|---|---|
| `prompt` | Redisplay the initial prompt and patient context. |
| `context` | List currently loaded patient scans, paths, and intermediate outputs. |
| `reset` | Reset conversation history. |
| `clear` | Clear the terminal screen. |
| `help` | Show available commands. |
| `exit` / `quit` / `q` | End the session and persist execution logs to `eval_outputs/`. |

---

## 📊 Batch Benchmark Evaluation (`weave_evaluation.py`)

To evaluate agents systematically across standard benchmark datasets (e.g., BraTS, UPenn-GBM, UCSF-PDGM):

```bash
python weave_evaluation.py --task 2 --agent-type as_tools --llm claude-sonnet-4-6 --temperature 0.0
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--task` | `int` | `2` | Evaluation task identifier (`2`: single-timepoint analysis, `3`: longitudinal analysis). |
| `--agent-type` | `str` | `as_tools` | Architecture to evaluate (`single`, `as_tools`, `handoffs`). |
| `--llm` | `str` | `claude-sonnet-4-6` | LLM model name. |
| `--temperature` | `float` | `0.0` | Sampling temperature for agent responses. |
| `--filter-cases` | `str` | `None` | Path to a text file containing specific case names to evaluate (one per line). |

Evaluation traces, tool usages, costs, and judge assessments are logged directly to [Weights & Biases Weave](https://wandb.ai/site/weave) and inside `eval_outputs/`.

See [dataset/README.md](dataset/README.md) for details on the evaluation cohorts and case manifests.
