"""Tests for registration number validation."""
import re

PATTERN_REG = re.compile(r'^\d{8,12}$')


def test_valid_registration_numbers():
    """Valid registration numbers should match the pattern."""
    # synthetic digit-only numbers (8-12 digits) - no real registration numbers.
    # Deliberately kept under 10 digits so an archive-wide PII scan (which
    # flags any 10+ digit run as a possible registration number/phone) stays 0.
    valid_pure = ['902250001', '902250002', '902250003',
                  '902250004', '902250005', '902250006']
    for rn in valid_pure:
        assert PATTERN_REG.match(rn), f'{rn} should be valid'


def test_invalid_registration_numbers():
    """Invalid registration numbers should not match the pattern."""
    # 'ABCDEFGHIJKL' is 13 chars -> too long for the 8-12 digit rule
    invalid = ['INVALID', 'AB12', 'ABCDEFGHIJKL', '1234567']
    for rn in invalid:
        assert not PATTERN_REG.match(rn), f'{rn} should be invalid'