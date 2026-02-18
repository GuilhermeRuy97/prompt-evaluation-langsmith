"""
    Utility functions for the prompt optimization project.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def load_yaml(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Loads YAML file.

    Args:
        file_path: YAML file path

    Returns:
        Dictionary with YAML content or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def save_yaml(data: Dict[str, Any], file_path: str) -> bool:
    """
    Saves data to YAML file.

    Args:
        data: Data to save
        file_path: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False, indent=2)

        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False


def check_env_vars(required_vars: list) -> bool:
    """
    Checks if required environment variables are configured.

    Args:
        required_vars: List of required variables

    Returns:
        True if all configured, False otherwise
    """
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nConfigure them in the .env file before continuing.")
        return False

    return True


def format_score(score: float, threshold: float = 0.9) -> str:
    """
    Formats score with visual approval indicator.

    Args:
        score: Score between 0.0 and 1.0
        threshold: Minimum threshold for approval

    Returns:
        Formatted string with score and symbol
    """
    symbol = "-V-" if score >= threshold else "-X-"
    return f"{score:.2f} {symbol}"


def print_section_header(title: str, char: str = "=", width: int = 50):
    """
    Prints formatted section header.

    Args:
        title: Section title
        char: Character for the line
        width: Line width
    """
    print("\n" + char * width)
    print(title)
    print(char * width + "\n")


def validate_prompt_structure(prompt_data: Dict[str, Any]) -> tuple[bool, list]:
    """
    Validates basic structure of a prompt.

    Args:
        prompt_data: Prompt data

    Returns:
        (is_valid, errors) - Tuple with status and error list
    """
    errors = []

    required_fields = ['description', 'system_prompt', 'version']
    for field in required_fields:
        if field not in prompt_data:
            errors.append(f"Missing required field: {field}")

    system_prompt = prompt_data.get('system_prompt', '').strip()
    if not system_prompt:
        errors.append("system_prompt is empty")

    if 'TODO' in system_prompt:
        errors.append("system_prompt still contains TODOs")

    techniques = prompt_data.get('techniques_applied', [])
    if len(techniques) < 2:
        errors.append(f"Minimum of 2 techniques required, found: {len(techniques)}")

    return (len(errors) == 0, errors)


def extract_json_from_response(response_text) -> Optional[Dict[str, Any]]:
    """
    Extracts JSON from an LLM response that may contain additional text.
    Handles both str and list[dict] content formats (langchain-google-genai 4.x).

    Args:
        response_text: LLM response text (str or list of content parts)

    Returns:
        Extracted dictionary or None if no valid JSON found
    """
    if isinstance(response_text, list):
        response_text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in response_text
        )

    if not isinstance(response_text, str):
        response_text = str(response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1

        if start != -1 and end > start:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass

    return None


def get_llm(model: Optional[str] = None, temperature: float = 0.0):
    """
    Returns a configured LLM instance based on the provider.

    Args:
        model: Model name (optional, uses LLM_MODEL from .env by default)

    Returns:
        Instance of ChatOpenAI or ChatGoogleGenerativeAI

    Raises:
        ValueError: If provider is not supported or API key is not configured
    """
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    model_name = model or os.getenv('LLM_MODEL', 'gpt-4o-mini')

    if provider == 'openai':
        from langchain_openai import ChatOpenAI

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not configured in .env\n"
                "Get a key at: https://platform.openai.com/api-keys"
            )

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0
        )

    elif provider == 'google':
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not configured in .env\n"
                "Get a key at: https://aistudio.google.com/app/apikey"
            )

        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            temperature=0
        )

    else:
        raise ValueError(
            f"Provider '{provider}' not supported.\n"
            f"Use 'openai' or 'google' in the LLM_PROVIDER variable in .env"
        )


def get_eval_llm(temperature: float = 0):
    """
    Returns LLM configured specifically for evaluation (uses EVAL_MODEL).

    Args:
        None

    Returns:
        LLM instance configured for evaluation
    """
    eval_model = os.getenv('EVAL_MODEL', 'gpt-4o')
    return get_llm(model=eval_model, temperature=temperature)
