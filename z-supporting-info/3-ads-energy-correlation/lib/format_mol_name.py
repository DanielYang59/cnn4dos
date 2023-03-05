#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re


def format_mol_name(string):
    pattern = r'\d+'
    numbers = re.findall(pattern, string)
    for num in numbers:
        num_subscript = ''
        for digit in num:
            num_subscript += chr(8320 + int(digit))
        string = string.replace(num, f'{num_subscript}')
    return string
