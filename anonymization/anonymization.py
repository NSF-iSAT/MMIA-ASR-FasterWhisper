"""
This module provides functionalities for text anonymization using the Presidio Analyzer and Anonymizer engines.
It offers two main functions:
    1. `anonymize_text`: Anonymizes PERSON entities in a given text.
    2. `anonymize_text_with_deny_list`: Anonymizes PERSON and predefined names in a given text using a deny-list.

Both functions use spaCy's en_core_web_lg model as the underlying NLP engine.

The PREDEFINED_NAME_LIST is taken from Column D in SI Data File Catalog 2021/2022 (restricted access)
https://docs.google.com/spreadsheets/d/1Xs5jM9yGRmRbrSxjrRFIUpzbfb-MdcGOi_FyJen6MXk/edit#gid=1111188151

Dependencies:
    - presidio_analyzer: For text entity recognition.
    - presidio_anonymizer: For text anonymization.
"""
import os

from presidio_analyzer import (AnalyzerEngine, PatternRecognizer,
                               RecognizerRegistry)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Define global objects for anonymize_text
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
anonymizer_config = {
    "PERSON": OperatorConfig("replace", {"new_value": "[name redacted]"})
}

# Read the PREDEFINED_NAME_LIST from a file
script_dir = os.path.dirname(__file__)  # get the directory of the current script
file_path = os.path.join(script_dir, 'predefined_name_list.txt')
with open(file_path, 'r') as file:
    lines = file.readlines()
PREDEFINED_NAME_LIST = [line.strip() for line in lines]

# Define global objects for anonymize_text_with_deny_list
deny_list_recognizer = PatternRecognizer(supported_entity="PREDEFINED_NAME", deny_list=PREDEFINED_NAME_LIST)
registry = RecognizerRegistry()
registry.load_predefined_recognizers()
registry.add_recognizer(deny_list_recognizer)

analyzer_with_deny_list = AnalyzerEngine(registry=registry)
anonymizer_with_deny_list = AnonymizerEngine()

anonymizer_config_with_deny_list = {
    "PERSON": OperatorConfig("replace", {"new_value": "[name redacted]"}),
    "PREDEFINED_NAME": OperatorConfig("replace", {"new_value": "[name redacted]"})
}

def anonymize_text(text_to_anonymize: str) -> str:
    """
    Anonymizes PERSON entities in the given text.
    The default NLP model used is spaCy/en_core_web_lg

    Parameters:
        text_to_anonymize (str): The text to be anonymized.

    Returns:
        str: The anonymized text.
    """
    # Perform the analysis to recognize PERSON entities
    analyzer_results = analyzer.analyze(text=text_to_anonymize, entities=["PERSON"], language='en',)

    # Perform the anonymization using the global anonymizer and config
    anonymized_result = anonymizer.anonymize(text=text_to_anonymize,
                                             analyzer_results=analyzer_results,
                                             operators=anonymizer_config)

    # Get the anonymized text
    anonymized_text = anonymized_result.text

    # Return the anonymized text
    return anonymized_text

def anonymize_text_with_deny_list(text_to_anonymize: str) -> str:
    """
    Anonymizes PERSON and PREDEFINED_NAME entities in the given text using a deny-list.
    The default NLP model used is spaCy/en_core_web_lg

    Parameters:
        text_to_anonymize (str): The text to be anonymized.

    Returns:
        str: The anonymized text.
    """
    # Perform the analysis to recognize PERSON and PREDEFINED_NAME entities
    analyzer_results = analyzer_with_deny_list.analyze(text=text_to_anonymize, entities=["PERSON", "PREDEFINED_NAME"], language='en',)

    # Perform the anonymization using the global anonymizer and config
    anonymized_result = anonymizer_with_deny_list.anonymize(text=text_to_anonymize,
                                             analyzer_results=analyzer_results,
                                             operators=anonymizer_config_with_deny_list)

    # Get the anonymized text
    anonymized_text = anonymized_result.text

    # Return the anonymized text
    return anonymized_text
