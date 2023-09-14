import sys
from typing import Dict, Literal


def ask_confirmation(question: str, default: Literal["yes", "no", None] = "no") -> bool:
    """Ask a yes/no question and return their answer. Notebook-friendly.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.

    Return: True = 'yes', False = 'no'
    """
    valid: Dict[str, bool] = {
        "yes": True,
        "y": True,
        "ye": True,
        "no": False,
        "n": False,
    }
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
