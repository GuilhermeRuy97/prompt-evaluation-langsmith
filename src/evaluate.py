"""
    COMPLETE script to evaluate optimized prompts.

    This script:
    1. Loads evaluation dataset from .jsonl file (datasets/bug_to_user_story.jsonl)
    2. Creates/updates dataset in LangSmith
    3. Pulls optimized prompts from LangSmith Hub (single source of truth)
    4. Executes prompts against the dataset
    5. Calculates 5 metrics (Helpfulness, Correctness, F1-Score, Clarity, Precision)
    6. Publishes results to LangSmith dashboard
    7. Displays summary in terminal

    Supports multiple LLM providers:
    - OpenAI (gpt-5.2)
    - Google Gemini (gemini-1.5-flash, gemini-1.5-pro)

    Configure the provider in the .env file through the LLM_PROVIDER variable.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm
from metrics import evaluate_f1_score, evaluate_clarity, evaluate_precision

load_dotenv()


def get_llm():
    return get_configured_llm(temperature=0)


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Ignore empty lines
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"File not found: {jsonl_path}")
        print("\nMake sure the file datasets/bug_to_user_story.jsonl exists.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSONL: {e}")
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"Creating evaluation dataset: {dataset_name}...")

    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("No examples loaded from .jsonl file")
        return dataset_name

    print(f"Loaded {len(examples)} examples from file {jsonl_path}")

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            print(f"Dataset '{dataset_name}' already exists, using existing one")
            return dataset_name
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)

            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"]
                )

            print(f"   Dataset created with {len(examples)} examples")
            return dataset_name

    except Exception as e:
        print(f"Error creating dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    try:
        print(f"   Pulling prompt from LangSmith Hub: {prompt_name}")
        client = Client()
        prompt = client.pull_prompt(prompt_name)
        print(f"   Prompt loaded successfully")
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"ERROR: Could not load prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("The prompt was not found in LangSmith Hub.\n")
            print("REQUIRED ACTIONS:")
            print("1. Verify if you have already pushed the optimized prompt:")
            print(f"python src/push_prompts.py")
            print()
            print("2. Confirm if the prompt was successfully published at:")
            print(f"https://smith.langchain.com/prompts")
            print()
            print(f"3. Make sure the prompt name is correct: '{prompt_name}'")
            print()
            print("4. If you modified the prompt in YAML, push again:")
            print(f"python src/push_prompts.py")
        else:
            print(f"Technical error: {e}\n")
            print("Check:")
            print("- LANGSMITH_API_KEY is configured correctly in .env")
            print("- You have access to LangSmith workspace")
            print("- Your internet connection is working")

        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any,
    max_retries: int = 5,
    retry_delay: float = 60.0
) -> Dict[str, Any]:
    inputs = example.inputs if hasattr(example, 'inputs') else {}
    outputs = example.outputs if hasattr(example, 'outputs') else {}

    if isinstance(inputs, dict):
        question = inputs.get("question", inputs.get("bug_report", inputs.get("pr_title", "N/A")))
    else:
        question = "N/A"

    chain = prompt_template | llm

    for attempt in range(1, max_retries + 1):
        try:
            response = chain.invoke(inputs)
            return {
                "answer": response.content,
                "reference": outputs.get("reference", "") if isinstance(outputs, dict) else "",
                "question": question
            }
        except Exception as e:
            is_retryable = any(code in str(e) for code in ["503", "429", "UNAVAILABLE", "Resource has been exhausted"])
            if is_retryable and attempt < max_retries:
                wait = retry_delay * attempt
                print(f"   Attempt {attempt}/{max_retries} failed (retryable). Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
            else:
                print(f"Error evaluating example: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                return {"answer": "", "reference": "", "question": question}


def evaluate_prompt(
    prompt_name: str,
    dataset_name: str,
    client: Client
) -> Dict[str, float]:
    print(f"\nEvaluating: {prompt_name}")

    prompt_template = pull_prompt_from_langsmith(prompt_name)
    llm = get_llm()

    # target function receives each example's inputs and returns the model output
    # retry logic mirrors evaluate_prompt_on_example for 503/429 transient errors
    def target(inputs: dict) -> dict:
        chain = prompt_template | llm
        max_retries, retry_delay = 5, 60.0
        for attempt in range(1, max_retries + 1):
            try:
                response = chain.invoke(inputs)
                content = response.content
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                    )
                return {"output": content}
            except Exception as e:
                is_retryable = any(code in str(e) for code in ["503", "429", "UNAVAILABLE", "Resource has been exhausted"])
                if is_retryable and attempt < max_retries:
                    wait = retry_delay * attempt
                    print(f"   target attempt {attempt}/{max_retries} failed. Waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    raise
        return {"output": ""}

    # evaluator wrappers expected by client.evaluate(): receive (outputs, reference_outputs)
    def evaluator_f1(outputs: dict, reference_outputs: dict) -> dict:
        answer = outputs.get("output", "")
        reference = reference_outputs.get("reference", "")
        result = evaluate_f1_score("", answer, reference)
        return {"key": "f1_score", "score": result["score"], "comment": result.get("reasoning", "")}

    def evaluator_clarity(outputs: dict, reference_outputs: dict) -> dict:
        answer = outputs.get("output", "")
        reference = reference_outputs.get("reference", "")
        result = evaluate_clarity("", answer, reference)
        return {"key": "clarity", "score": result["score"], "comment": result.get("reasoning", "")}

    def evaluator_precision(outputs: dict, reference_outputs: dict) -> dict:
        answer = outputs.get("output", "")
        reference = reference_outputs.get("reference", "")
        result = evaluate_precision("", answer, reference)
        return {"key": "precision", "score": result["score"], "comment": result.get("reasoning", "")}

    # Limit to first 10 examples to control cost
    examples = list(client.list_examples(dataset_name=dataset_name))[:10]
    print(f"   Dataset: {len(examples)} examples (limited to 10)")
    print("   Evaluating examples...")

    experiment_results = client.evaluate(
        target,
        data=examples,
        evaluators=[evaluator_f1, evaluator_clarity, evaluator_precision],
        experiment_prefix=prompt_name.replace("/", "--"),
        max_concurrency=1,
        num_repetitions=1,
    )

    # ExperimentResultRow is TypedDict — access via ["key"]
    # EvaluationResult is Pydantic — access via .key and .score
    f1_scores, clarity_scores, precision_scores = [], [], []

    for i, result in enumerate(experiment_results, 1):
        eval_results = (result.get("evaluation_results") or {}).get("results") or []
        row = {r.key: r.score for r in eval_results if r.score is not None}
        f1_scores.append(row.get("f1_score", 0.0))
        clarity_scores.append(row.get("clarity", 0.0))
        precision_scores.append(row.get("precision", 0.0))
        print(f"      [{i}/10] F1:{row.get('f1_score', 0):.2f} Clarity:{row.get('clarity', 0):.2f} Precision:{row.get('precision', 0):.2f}")

    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    avg_helpfulness = (avg_clarity + avg_precision) / 2
    avg_correctness = (avg_f1 + avg_precision) / 2

    return {
        "helpfulness": round(avg_helpfulness, 4),
        "correctness": round(avg_correctness, 4),
        "f1_score": round(avg_f1, 4),
        "clarity": round(avg_clarity, 4),
        "precision": round(avg_precision, 4)
    }


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    print("\nLangSmith Metrics:")
    print(f"  - Helpfulness: {format_score(scores['helpfulness'], threshold=0.9)}")
    print(f"  - Correctness: {format_score(scores['correctness'], threshold=0.9)}")

    print("\nCustom Metrics:")
    print(f"  - F1-Score: {format_score(scores['f1_score'], threshold=0.9)}")
    print(f"  - Clarity: {format_score(scores['clarity'], threshold=0.9)}")
    print(f"  - Precision: {format_score(scores['precision'], threshold=0.9)}")

    average_score = sum(scores.values()) / len(scores)

    print("\n" + "-" * 50)
    print(f"OVERALL AVERAGE: {average_score:.4f}")
    print("-" * 50)

    passed = average_score >= 0.9

    if passed:
        print(f"\nSTATUS: PASSED (average >= 0.9)")
    else:
        print(f"\nSTATUS: FAILED (average < 0.9)")
        print(f"Current average: {average_score:.4f} | Required: 0.9000")

    return passed


def main():
    print_section_header("OPTIMIZED PROMPTS EVALUATION")

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-5.2")
    eval_model = os.getenv("EVAL_MODEL", "gpt-5.2")

    print(f"Provider: {provider}")
    print(f"Main Model: {llm_model}")
    print(f"Evaluation Model: {eval_model}\n")

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", os.getenv("LANGCHAIN_PROJECT", "prompt-optimization-challenge-resolved"))

    jsonl_path = "datasets/bug_to_user_story.jsonl"

    if not Path(jsonl_path).exists():
        print(f"Dataset file not found: {jsonl_path}")
        print("\nMake sure the file exists before continuing.")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    print("\n" + "=" * 70)
    print("PROMPTS TO EVALUATE")
    print("=" * 70)
    print("\nThis script will pull prompts from LangSmith Hub.")
    print("Make sure you have pushed the prompts before evaluating:")
    print("  python src/push_prompts.py\n")

    prompts_to_evaluate = [
        "data-ruy/bug_to_user_story_v2",
    ]

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        evaluated_count += 1

        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client)

            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append({
                "prompt": prompt_name,
                "scores": scores,
                "passed": passed
            })

        except Exception as e:
            print(f"\nFailed to evaluate '{prompt_name}': {e}")
            all_passed = False

            results_summary.append({
                "prompt": prompt_name,
                "scores": {
                    "helpfulness": 0.0,
                    "correctness": 0.0,
                    "f1_score": 0.0,
                    "clarity": 0.0,
                    "precision": 0.0
                },
                "passed": False
            })

    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50 + "\n")

    if evaluated_count == 0:
        print("No prompts were evaluated")
        return 1

    print(f"Prompts evaluated: {evaluated_count}")
    print(f"Passed: {sum(1 for r in results_summary if r['passed'])}")
    print(f"Failed: {sum(1 for r in results_summary if not r['passed'])}\n")

    if all_passed:
        print("All prompts achieved average >= 0.9!")
        print(f"\nCheck the results at:")
        print(f"  https://smith.langchain.com/projects/{project_name}")
        print("\nNext steps:")
        print("1. Document the process in README.md")
        print("2. Capture screenshots of evaluations")
        print("3. Commit and push to GitHub")
        return 0
    else:
        print("Some prompts did not achieve average >= 0.9")
        print("\nNext steps:")
        print("1. Refactor prompts with low scores")
        print("2. Push again: python src/push_prompts.py")
        print("3. Run: python src/evaluate.py again")
        return 1

if __name__ == "__main__":
    sys.exit(main())
