#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re


def format_mol_name(string):
    """Format names of molecules: apply subscript to numbers.

    Args:
        string (str): original name

    Returns:
        str: formatted name

    """
    pattern = r'\d+'
    numbers = re.findall(pattern, string)
    for num in numbers:
        num_subscript = ''
        for digit in num:
            num_subscript += chr(8320 + int(digit))
        string = string.replace(num, f'{num_subscript}')
    return string
