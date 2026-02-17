"""
    Script to pull prompts from LangSmith Prompt Hub.

    This script:
    1. Connects to LangSmith using credentials from .env
    2. Pulls prompts from Hub
    3. Saves locally to prompts/bug_to_user_story_v1.yml

    SIMPLIFIED: Uses native LangChain serialization to extract prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()


def pull_prompts_from_langsmith():
    """
    Pull prompts from LangSmith Hub and saves locally to prompts/raw_prompts.yml.

    Args:
        None

    Returns:
        True if successful, False otherwise
    """

    prompt_identifier = "leonanluppi/bug_to_user_story_v1"
    output_path = "prompts/bug_to_user_story_v1.yml"

    try:
        # Pull prompt from LangSmith Hub
        prompt = hub.pull(prompt_identifier)

        # Extract the system prompt
        system_prompt = prompt.messages[0].prompt.template

        # Prompt Data Structure
        prompt_data = {
            "bug_to_user_story_v1":{
                "description": "Prompt para converter relatos de bugs em User Stories",
                "system_prompt": system_prompt,
                "user_prompt": "{bug_report}",
                "version": "v1",
                "created_at": "2025-01-15",
                "tags": ["bug_analysis", "user-story", "product-management"]
            }
        }
        
        # Save prompt to YAML file
        success = save_yaml(prompt_data, output_path)

        if success:
            return True
        else:
            return False

    except Exception as e:
        print(f"Error pulling prompts from LangSmith Hub: {e}")
        return False

def main():
    """
    Main function to orchestrate the prompt pulling process.

    Returns:
        Exit code 0 if successful, 1 for failure
    """
    
    # Check required environment variables
    required_vars = [
        'LANGSMITH_API_KEY',
        'USERNAME_LANGSMITH_HUB'
    ]
    if not check_env_vars(required_vars):
        return 1
    
    # Pull prompts from LangSmith Hub
    success = pull_prompts_from_langsmith()

    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
