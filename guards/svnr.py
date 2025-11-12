import re

WEIGHTS = [3, 7, 9, 0, 5, 8, 4, 2, 1, 6]
SVNR_REGEX = re.compile(r"^[0-9]{10}$")

def is_valid_svnr(svnr: str) -> bool:
    """
    Validates an Austrian social security number (SVNR) based on its checksum.
    """
    if not SVNR_REGEX.match(svnr) or svnr.startswith('0'):
        return False

    checksum = calculate_checksum(svnr)
    check_digit = int(svnr[3])

    return checksum == check_digit

def calculate_checksum(svnr: str) -> int:
    """
    Calculates the checksum for a 10-digit SVNR string.
    The checksum is the modulo 11 of the sum of each digit multiplied by its corresponding weight.
    """
    digits = [int(d) for d in svnr]
    
    weighted_sum = sum(digit * weight for digit, weight in zip(digits, WEIGHTS))
    
    return weighted_sum % 11
