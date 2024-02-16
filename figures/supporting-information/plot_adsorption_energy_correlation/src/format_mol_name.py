"""Format molecule names to allow subscripts."""

import re


def format_mol_name(string):
    """Format names of molecules: apply subscript to numbers using LaTeX math mode.

    Args:
        string (str): original name

    Returns:
        str: formatted name

    """
    pattern = r"\d+"
    numbers = re.findall(pattern, string)

    for num in numbers:
        num_subscript = ""
        for digit in num:
            num_subscript += f"$_\mathregular{{{digit}}}$"
        string = string.replace(num, f"{num_subscript}")

    return string


# Example usage
example_name = "CO2"
formatted_name = format_mol_name(example_name)
print(formatted_name)
