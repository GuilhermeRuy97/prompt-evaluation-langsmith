"""
    Script to push optimized prompts to LangSmith Prompt Hub.

    This script:
    1. Reads optimized prompts from prompts/bug_to_user_story_v2.yml
    2. Validates the prompts
    3. Makes PUBLIC push to LangSmith Hub
    4. Adds metadata (tags, description, techniques used)

    SIMPLIFIED: Cleaner and more straightforward code.
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header, validate_prompt_structure

load_dotenv()

PROMPT_FILE = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Pushes optimized prompt to LangSmith Hub (PUBLIC).

    Args:
        prompt_name: Prompt name
        prompt_data: Prompt data

    Returns:
        True if successful, False otherwise
    """
    username = os.getenv("USERNAME_LANGSMITH_HUB")
    if not username:
        print("USERNAME_LANGSMITH_HUB not set in .env")
        return False

    # Format: username/repo
    repo_handle = f"{username}/{prompt_name}"

    try:
        # Construct ChatPromptTemplate
        # We use the system prompt and user prompt defined in the YAML
        system_template = prompt_data.get("system_prompt", "")
        user_template = prompt_data.get("user_prompt", "{bug_report}")

        # Create the prompt object
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])

        # Get metadata
        description = prompt_data.get("description", "Optimized prompt")
        tags = prompt_data.get("tags", [])
        
        # Push to Hub
        # new_repo_is_public=True ensures it's accessible
        url = hub.push(
            repo_handle,
            prompt,
            new_repo_is_public=True,
            new_repo_description=description,
            tags=tags
        )

        print(f"Prompt pushed successfully to: {url}")
        return True

    except Exception as e:
        print(f"Error pushing prompt: {e}")
        return False


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Validates basic structure of a prompt (simplified version).

    Args:
        prompt_data: Prompt data

    Returns:
        (is_valid, errors) - Tuple with status and error list
    """
    # Reuse the robust validation from utils
    return validate_prompt_structure(prompt_data)


def main():
    """Main function"""
    print_section_header("Push Prompt to LangSmith Hub")

    # 1. Check environment
    required_vars = ["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]
    if not check_env_vars(required_vars):
        return 1

    # 2. Load Prompt
    print(f"Loading prompt from {PROMPT_FILE}...")
    yaml_data = load_yaml(PROMPT_FILE)
    if not yaml_data:
        return 1

    if PROMPT_KEY not in yaml_data:
        print(f"Key '{PROMPT_KEY}' not found in YAML file.")
        print(f"   Available keys: {list(yaml_data.keys())}")
        return 1

    prompt_data = yaml_data[PROMPT_KEY]

    # 3. Validate
    print("Validating prompt structure...")
    is_valid, errors = validate_prompt(prompt_data)
    
    if not is_valid:
        print("Validation failed:")
        for err in errors:
            print(f"   - {err}")
        return 1
    
    print("Prompt structure is valid.")

    # 4. Push
    print(f"Pushing '{PROMPT_KEY}' to LangSmith Hub...")
    success = push_prompt_to_langsmith(PROMPT_KEY, prompt_data)

    if success:
        print("\nProcess completed successfully!")
        return 0
    else:
        print("\nProcess failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
