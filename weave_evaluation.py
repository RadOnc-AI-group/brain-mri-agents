import os
import argparse
import weave
import json
import asyncio
from datetime import datetime
from pathlib import Path

os.environ["LLM_JUDGE"] = "gpt-5" 
from util.weave_utils import AgentWeaveWrapper, ResponseJudgeLLM, preprocess_data, control_tool_usage, extract_costs

# RERUN = True

def setup(task, agent_type, llm_model, temperature=0.0, filter_cases_file=None):
    os.environ["MODEL_NAME"] = llm_model
    os.environ["TEMPERATURE"] = str(temperature)  
    from agent_zoo import (
        analysing_orchestrator_w_regcheck,
        orchestrator_w_regcheck,
        single_agent_w_regcheck,
        analysing_orchestrator_w_regcheck_handoffs,
    )

    if agent_type == "single":
        agent = single_agent_w_regcheck
    elif agent_type == "orchestrator":
        agent = orchestrator_w_regcheck
    elif agent_type == "as_tools":
        agent = analysing_orchestrator_w_regcheck
    elif agent_type == "handoffs":
        agent = analysing_orchestrator_w_regcheck_handoffs
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    json_path = f"./dataset/task{task}_cases.json"

    data = json.load(open(json_path, "r"))

    if filter_cases_file is not None:
        with open(filter_cases_file, "r") as f:
            only_use_cases = [line.strip() for line in f.readlines()]
        data = {k:v for k,v in data.items() if k in only_use_cases}

    data = list(data.values())

    day = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H-%M-%S")

    if agent_type == "as_tools":
        agent_str = "as-tools"
    else:
        agent_str = agent_type

    out_path = Path(f"./eval_outputs/{day}/task{task}_agent_{agent_str}_{llm_model}_{time}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    init_out = {}
    json.dump(init_out, open(out_path, "w"), indent=4)

    return agent, data, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Agent LLM for Brainz")
    parser.add_argument("--agent-type", type=str, default="as_tools", help="Type of agent to use", choices=["single", "orchestrator", "as_tools", "handoffs"])
    parser.add_argument("--task", type=int, default=2, help="Task type")
    parser.add_argument("--filter-cases", type=str, default=None, help="Path to txt file containing case names to only evaluate on (one case name per line)")
    parser.add_argument("--llm", type=str, default="claude-sonnet-4-6", help="LLM model to use") #gpt-5.1 # claude-sonnet-4
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for agent LLM responses")
    args = parser.parse_args()

    # Determine Weave/W&B project and entity from environment / api_keys.yaml
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project_base = os.environ.get("WANDB_PROJECT", "").strip() or f"brain-agent-task{args.task}"
    project_name = f"{entity}/{project_base}" if entity else project_base

    client = weave.init(project_name)
    # call = client.get_call("019b4b87-e994-73f9-9ec3-744894c54bc1",include_costs=True)

    agent, data, out_path = setup(args.task, args.agent_type, args.llm, args.temperature, filter_cases_file=args.filter_cases) 
    
    name = out_path.stem[6:] + "_" + out_path.parts[-2]  # remove 'taskX_' prefix and add day suffix

    # if 'claude' in args.llm.lower() or 'gemini-3-pro-preview' in args.llm.lower():
    os.environ['WEAVE_PARALLELISM'] = '1'

    model = AgentWeaveWrapper(agent=agent, agent_type=args.agent_type, output_log_file=out_path)
    evaluation = weave.Evaluation(
        dataset=data, 
        scorers=[lambda gt_tools,output:control_tool_usage(gt_tools=gt_tools, output=output, task_type=args.agent_type),] if args.filter_cases is None else [],  # don't evaluate reruns, do alltogether later
        preprocess_model_input=lambda x: preprocess_data(x, framework=args.agent_type, task=args.task), 
        evaluation_name=name,
    )
