"""
    Automated tests for prompt validation.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import validate_prompt_structure

def load_prompts(file_path: str):
    """Loads prompts from YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TestPrompts:
    @pytest.fixture
    def prompt_v2(self):
        """Load v2 prompt for testing."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        return load_prompts(str(prompt_path))
    
    def test_prompt_has_system_prompt(self, prompt_v2):
        """Verifies if the 'system_prompt' field exists and is not empty."""
        # Get the first key in the dict
        prompt_key = list(prompt_v2.keys())[0]
        prompt_data = prompt_v2[prompt_key]
        
        # Check field exists
        assert 'system_prompt' in prompt_data, "Field 'system_prompt' is missing"
        
        # Check not empty
        system_prompt = prompt_data['system_prompt']
        assert system_prompt, "Field 'system_prompt' is empty"
        assert len(system_prompt.strip()) > 0, "Field 'system_prompt' contains only whitespace"

    def test_prompt_has_role_definition(self, prompt_v2):
        """Verifies if the prompt defines a persona (e.g., "You are a Product Manager")."""
        prompt_key = list(prompt_v2.keys())[0]
        prompt_data = prompt_v2[prompt_key]
        system_prompt = prompt_data['system_prompt'].lower()
        
        # Check for role definition patterns in Portuguese and English
        role_patterns = [
            'você é um',  # Portuguese: "você é um Product Manager"
            'você é uma',  # Portuguese: "você é uma Product Manager"
            'you are a',   # English: "you are a Product Manager"
            'you are an',  # English: "you are an expert"
            'atua como',   # Portuguese alternative
            'sua função é', # Portuguese alternative
        ]
        
        has_role = any(pattern in system_prompt for pattern in role_patterns)
        assert has_role, f"Prompt does not define a role/persona. Expected patterns like: {role_patterns}"

    def test_prompt_mentions_format(self, prompt_v2):
        """Verifies if the prompt requires Markdown format or standard User Story."""
        prompt_key = list(prompt_v2.keys())[0]
        prompt_data = prompt_v2[prompt_key]
        system_prompt = prompt_data['system_prompt'].lower()
        
        # Check for format mentions
        format_keywords = [
            'markdown',
            'user story',
            'user stories',
            'formato',  # Portuguese: format
            'structure',
            'estrutura',  # Portuguese: structure
            'critérios de aceitação',  # Portuguese: acceptance criteria
            'acceptance criteria',
        ]
        
        mentions_format = any(keyword in system_prompt for keyword in format_keywords)
        assert mentions_format, f"Prompt does not mention output format. Expected keywords: {format_keywords}"

    def test_prompt_has_few_shot_examples(self, prompt_v2):
        """Verifies if the prompt contains input/output examples (Few-shot technique)."""
        prompt_key = list(prompt_v2.keys())[0]
        prompt_data = prompt_v2[prompt_key]
        system_prompt = prompt_data['system_prompt'].lower()
        
        # Check for example indicators
        example_patterns = [
            'exemplo',  # Portuguese: example
            'example',
            'input:',
            'output:',
            'bug_report":',  # JSON example format
            'reference":',   # JSON example format
        ]
        
        has_examples = any(pattern in system_prompt for pattern in example_patterns)
        
        # Additional check: count occurrences to ensure multiple examples
        example_count = system_prompt.count('exemplo') + system_prompt.count('example')
        
        assert has_examples, "Prompt does not contain few-shot examples (input/output pairs)"
        assert example_count >= 2, f"Prompt should have at least 2 examples, found indicators: {example_count}"

    def test_prompt_no_todos(self, prompt_v2):
        """Ensures that no `[TODO]` was left in the text."""
        prompt_key = list(prompt_v2.keys())[0]
        prompt_data = prompt_v2[prompt_key]
        
        # Check all string fields for [TODO]
        for field_name, field_value in prompt_data.items():
            if isinstance(field_value, str):
                assert '[TODO]' not in field_value, f"Found [TODO] in field '{field_name}'"
                assert '[todo]' not in field_value.lower(), f"Found [todo] in field '{field_name}'"

    def test_minimum_techniques(self, prompt_v2):
        """Verifies (through yaml metadata) if at least 2 techniques were listed."""
        prompt_key = list(prompt_v2.keys())[0]
        prompt_data = prompt_v2[prompt_key]
        
        # Check for techniques_applied field
        assert 'techniques_applied' in prompt_data, "Field 'techniques_applied' is missing in metadata"
        
        techniques = prompt_data['techniques_applied']
        assert isinstance(techniques, list), "Field 'techniques_applied' should be a list"
        assert len(techniques) >= 2, f"Expected at least 2 techniques, found {len(techniques)}: {techniques}"
        
        # Verify techniques are not empty strings
        non_empty_techniques = [t for t in techniques if t and t.strip()]
        assert len(non_empty_techniques) >= 2, "At least 2 non-empty techniques required"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])