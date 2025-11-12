import re

def is_valid(query: str) -> bool:
    """
    Validates the user's query.
    """
    # Check for query length
    if len(query) > 500:
        return False

    # Check for SQL injection patterns
    sql_injection_patterns = [
        r"(\s*(--|#|;))",
        r"(\s*(union|select|insert|update|delete|drop|alter)\s+)",
    ]
    for pattern in sql_injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False

    return True

if __name__ == '__main__':
    # Example usage
    valid_query = "What are the computer science courses?"
    invalid_query_long = "a" * 501
    invalid_query_sql = "SELECT * FROM students;"

    print(f"'{valid_query}' is valid: {is_valid(valid_query)}")
    print(f"'long query' is valid: {is_valid(invalid_query_long)}")
    print(f"'{invalid_query_sql}' is valid: {is_valid(invalid_query_sql)}")
