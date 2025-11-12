import re

def sanitize(response: str) -> str:
    """
    Sanitizes the LLM's response.
    """
    # Redact email addresses
    response = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED EMAIL]", response)

    # Check for hallucination patterns
    hallucination_patterns = [
        r"as an ai language model",
        r"i cannot answer that question",
    ]
    for pattern in hallucination_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return "I'm sorry, I don't have enough information to answer that question."

    return response

if __name__ == '__main__':
    # Example usage
    valid_response = "Professor Smith's email is john.smith@university.edu."
    hallucination_response = "As an AI language model, I cannot provide that information."

    print(f"Original: '{valid_response}'\nSanitized: '{sanitize(valid_response)}'")
    print(f"\nOriginal: '{hallucination_response}'\nSanitized: '{sanitize(hallucination_response)}'")
