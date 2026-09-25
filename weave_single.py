import os
import argparse
import json
import weave
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from datetime import datetime
from pathlib import Path
from typing import Union, List, Tuple

from agents import Agent, Runner, SQLiteSession, ItemHelpers
from util.data_types import instantiate_patient, Patient
from util.context_manager import SessionContext
from util.helpers import setup_api_keys


def setup(agent_type: str, llm_model: str, temperature: float = 0.0, task: int = None) -> Tuple[Agent, Path]:
    """Sets up the environment variables and instantiates the chosen agent."""
    os.environ["MODEL_NAME"] = llm_model
    os.environ["TEMPERATURE"] = str(temperature)

    from agent_zoo import (
        analysing_orchestrator_w_regcheck,
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

    day = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H-%M-%S")
    agent_str = "as-tools" if agent_type == "as_tools" else agent_type

    task_prefix = f"task{task}_" if task is not None else ""
    out_path = Path(f"./eval_outputs/{day}/{task_prefix}single_agent_{agent_str}_{llm_model}_{time}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    init_out = {}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(init_out, f, indent=4)

    return agent, out_path


def preprocess_patient(patient_name: str, patient_info: dict) -> Tuple[Patient, str]:
    """
    Preprocesses a single patient dictionary entry into a Patient object
    and returns the patient together with its contextual description string.
    """
    skullstripped = patient_info.get("skullstripped", True)

    allowed_keys = {
        "t1", "t2", "flair", "t1c", "seg", "dura",
        "brain_regions", "brain_region_volumes",
        "measurements", "radiomics", "t1c_w_skull"
    }
    kwargs = {k: v for k, v in patient_info.items() if k in allowed_keys}

    patient, msg = instantiate_patient(name=patient_name, **kwargs)

    if skullstripped is False:
        msg += " The input scans are not skullstripped."
    else:
        msg += " The input scans are skullstripped."

    return patient, msg


def load_and_preprocess_patient_files(json_path: Union[str, Path]) -> Tuple[List[Patient], List[str]]:
    """
    Reads patient_files.json and instantiates/preprocesses all patient entries.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Patient files configuration not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty dictionary in '{path}' with patient names as keys.")

    patients = []
    messages = []
    for patient_name, patient_info in data.items():
        if not isinstance(patient_info, dict):
            raise ValueError(f"Expected a dictionary for patient '{patient_name}', got {type(patient_info)}.")
        patient, msg = preprocess_patient(patient_name, patient_info)
        patients.append(patient)
        messages.append(msg)

    return patients, messages


def execute_agent_turn(agent, input_text: str, session_context: SessionContext, session: SQLiteSession, console: Console):
    """Executes a single turn of the agent, prints streaming/item updates, and returns the next active agent."""
    agent_name = getattr(agent, "name", "agent")
    console.print(f"\n[bold yellow]🤖 Running {agent_name}...[/bold yellow]\n")

    try:
        response = Runner.run_sync(
            agent,
            input=input_text,
            context=session_context,
            session=session,
        )
    except Exception as e:
        console.print(f"[bold red]❌ Error during agent execution:[/bold red] {e}")
        return agent, None

    turn_tools = []
    messages = []

    for item in response.new_items:
        if item.type == "tool_call_item":
            tool_name = getattr(item.raw_item, "name", "unknown_tool")
            turn_tools.append(tool_name)
            console.print(f"  [bold yellow]🔧 Tool called:[/bold yellow] [cyan]{tool_name}[/cyan]")
        elif item.type == "handoff_call_item":
            target_name = getattr(item.raw_item, "name", "sub-agent")
            console.print(f"  [bold magenta]🔀 Handoff to:[/bold magenta] [magenta]{target_name}[/magenta]")
        elif item.type == "message_output_item":
            msg_text = ItemHelpers.text_message_output(item)
            messages.append(msg_text)
            console.print()
            active_name = getattr(response.last_agent, "name", "Agent")
            console.print(Panel(Markdown(msg_text), title=f"[bold green]Agent Response ({active_name})[/bold green]", border_style="green"))

    if not messages and getattr(response, "final_output", None):
        out_text = str(response.final_output)
        messages.append(out_text)
        console.print()
        console.print(Panel(Markdown(out_text), title="[bold green]Agent Response[/bold green]", border_style="green"))

    combined_response = "\n\n".join(messages)
    turn_summary = {
        "user": input_text,
        "agent": combined_response,
        "tools": turn_tools,
        "timestamp": datetime.now().isoformat(),
    }
    return response.last_agent, turn_summary


if __name__ == "__main__":
    setup_api_keys()

    parser = argparse.ArgumentParser(description="Interactive Brain MRI Agent Terminal")
    parser.add_argument("--patient-files-json", type=str, default="patient_files.json", help="Path to JSON file containing patient MRI file paths (default: patient_files.json)")
    parser.add_argument("--agent-type", type=str, default="single", help="Type of agent to use", choices=["single", "as_tools", "handoffs"])
    parser.add_argument("--prompt", type=str, default=None, help="Initial clinical prompt/question (if omitted, you will be prompted interactively)")
    parser.add_argument("--llm", type=str, default="gpt-5.4-mini", help="LLM model to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for agent LLM responses")
    parser.add_argument("--task", type=int, default=None, help="Optional task identifier")
    parser.add_argument("--no-weave", action="store_true", help="Disable Weave tracking for this run")
    args = parser.parse_args()

    console = Console()

    # Initialize Weave tracking if enabled
    if not args.no_weave:
        entity = os.environ.get("WANDB_ENTITY", "").strip()
        default_proj = f"brain-agent-task{args.task}" if args.task is not None else "brain-mri-agents"
        project_base = os.environ.get("WANDB_PROJECT", "").strip() or default_proj
        project_name = f"{entity}/{project_base}" if entity else project_base
        client = weave.init(project_name)
    else:
        client = None
        project_name = None

    # Load agent
    agent, out_path = setup(args.agent_type, args.llm, args.temperature, task=args.task)

    # Load and preprocess all patient entries
    try:
        patients, patient_messages = load_and_preprocess_patient_files(args.patient_files_json)
    except Exception as e:
        console.print(f"[bold red]❌ Error loading patient files from '{args.patient_files_json}':[/bold red] {e}")
        exit(1)

    # Initialize session context and register patients
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session = SQLiteSession(session_id)
    session_context = SessionContext()
    session_context["output_dir"] = Path.cwd() / "outputs" / args.llm / args.agent_type

    for patient in patients:
        session_context[patient.name] = patient

    # Display session header
    weave_status = f"[bold]Weave:[/bold] {project_name}" if not args.no_weave else "[bold]Weave:[/bold] [dim]Disabled[/dim]"
    header_info = (
        f"[bold]Agent:[/bold] {args.agent_type}  |  "
        f"[bold]LLM:[/bold] {args.llm}  |  "
        f"[bold]Patients Loaded:[/bold] {len(patients)}  |  "
        f"{weave_status}"
    )
    console.print(Panel(header_info, title="[bold cyan]🧠 Brain MRI Agent Session[/bold cyan]", border_style="cyan"))

    # Display loaded patient details
    patients_summary = "\n".join([f"• {msg}" for msg in patient_messages])
    console.print(Panel(patients_summary, title="[bold green]📁 Loaded Patients Context[/bold green]", border_style="green"))

    current_agent = agent
    history = []

    # Obtain initial prompt
    if args.prompt:
        user_prompt = args.prompt.strip()
    else:
        console.print("\n[bold green]Enter your initial clinical question / instructions for the agent:[/bold green]")
        try:
            user_prompt = Prompt.ask("[bold cyan]Prompt[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session aborted by user.[/bold yellow]")
            exit(0)

    # Preprocess patient description block and combine with initial user prompt
    patient_context_str = "\n".join(patient_messages)
    if user_prompt:
        initial_turn_input = f"{user_prompt}\n{patient_context_str}"
    else:
        initial_turn_input = patient_context_str

    console.print(Panel(initial_turn_input, title="[bold cyan]Initial Query & Patient Context[/bold cyan]", border_style="cyan"))

    # Execute initial agent turn
    current_agent, turn_info = execute_agent_turn(current_agent, initial_turn_input, session_context, session, console)
    if turn_info:
        history.append(turn_info)

    # Enter interactive terminal loop for follow-up turns
    console.print("\n" + "═" * 60)
    console.print("[bold green]💬 Interactive Mode Active[/bold green]")
    console.print("[dim]Type your message and press Enter.[/dim]")
    console.print("[dim]Commands: 'exit'/'quit'/'q' to end, 'clear' to clear screen, 'reset' to reset session, 'prompt' to view initial prompt, 'context' to view patients, 'help' for help.[/dim]")
    console.print("═" * 60 + "\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]👤 You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold yellow]Session ended by user.[/bold yellow]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in {"exit", "quit", "q"}:
            console.print("[bold yellow]Exiting interactive session. Goodbye![/bold yellow]")
            break

        if cmd == "clear":
            console.clear()
            continue

        if cmd in {"help", "?"}:
            console.print(
                "[bold]Available Commands:[/bold]\n"
                "  [cyan]exit[/cyan], [cyan]quit[/cyan], [cyan]q[/cyan] : Exit session\n"
                "  [cyan]clear[/cyan]           : Clear terminal screen\n"
                "  [cyan]reset[/cyan]           : Reset conversation history\n"
                "  [cyan]prompt[/cyan]          : Redisplay initial case prompt\n"
                "  [cyan]context[/cyan]         : List loaded patient context\n"
                "  [cyan]help[/cyan]            : Show this help message"
            )
            continue

        if cmd == "prompt":
            console.print(Panel(initial_turn_input, title="Initial Query & Patient Context", border_style="cyan"))
            continue

        if cmd == "context":
            patients_info = "\n".join([f"• [bold]{p.name}[/bold]: {session_context.get(p.name, p)}" for p in patients])
            console.print(Panel(patients_info or "No patients loaded.", title="Loaded Patient Context", border_style="blue"))
            continue

        if cmd == "reset":
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session = SQLiteSession(session_id)
            current_agent = agent
            console.print("[bold yellow]Conversation history has been reset.[/bold yellow]")
            continue

        current_agent, turn_info = execute_agent_turn(current_agent, user_input, session_context, session, console)
        if turn_info:
            history.append(turn_info)

    # Save interaction history if any turns were executed
    if history:
        out_record = {
            "patient_names": [p.name for p in patients],
            "patient_files": str(args.patient_files_json),
            "agent_type": args.agent_type,
            "llm": args.llm,
            "initial_prompt": user_prompt,
            "history": history,
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_record, f, indent=4, default=str)
            console.print(f"[dim]Saved session history to [underline]{out_path}[/underline][/dim]")
        except Exception as e:
            console.print(f"[dim red]Failed to save session history: {e}[/dim red]")
