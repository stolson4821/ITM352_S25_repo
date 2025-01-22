import textwrap

# Function to display information with clear formatting
def display_content(title, content):
    print(f"\n{'='*15}")
    print(f"{title.upper()}\n{'='*60}")
    for line in content:
        print(textwrap.fill(line, width=40))  # Adjust the width for smartphone readability
        print()

# Learning Objectives
learning_objectives = [
    "1. Explain Python’s background and important features",
    "2. Describe free, open-source software (FOSS)",
    "3. Summarize Python’s user community and available resources",
    "4. Install Python’s platform-independent interpreter",
    "5. Execute Python code in an Interactive Development Environment (IDE)",
    "6. Describe the two data sets used throughout the book"
]

content = [
    """
    Python is an interpreted computer programming language 
    in which you can enter code instructions one at a time 
    or as part of a larger program. Syntax is a set of rules 
    that dictate how to specify instructions in a programming language.
    Packages are libraries of code modules that other programming code 
    can access and use.
    """,
    """
    FOSS is an inclusive term covering both free software and open-source 
    software. Free software gives users the freedom to run, copy, distribute, 
    study, change, and improve the software. Open-source software allows 
    unrestricted selling or giving away of the software.
    """,
    """
    Python has a robust user community and extensive resources. 
    The official Python website (https://www.python.org) offers downloads, 
    documentation, and other resources. PyPI (https://pypi.org) hosts over 
    212,000 Python projects for various applications.
    
    As of late 2019, Python versions:
    - Python 3.9: In development
    - Python 3.8: Stable
    - Python 3.7: Stable
    - Python 3.6: Security fixes
    - Python 2.7: Stable
    """,
    "Python is platform-independent software that runs on most operating systems.",
    "Python code can be executed in an Interactive Development Environment (IDE).",
    """
    Taxi Trips Data Set: Comprised of 26 fields, over 100 million records.
    General Social Survey (GSS) Data Set: Contains over 5000 variables collected 
    over 40+ years. Codebook: http://gss.norc.org/get-documentation
    """
]

# Glossary
glossary = {
    "Case sensitive": "Interprets uppercase letters as different from lowercase.",
    "FOSS": "Inclusive term covering free and open-source software.",
    "Free software": "Freedom to run, copy, distribute, study, and improve software.",
    "IDE": "Environment for writing, editing, testing, and debugging code.",
    "Module": "Text file containing Python code.",
    "Open-source software": "License does not restrict selling/giving away the software.",
    "Package": "Library of code modules.",
    "pip": "Python's package manager.",
    "Platform": "Combination of device and operating system.",
    "Platform independent": "Software that runs on multiple platforms.",
    "Python": "Interpreted programming language.",
    "Syntax": "Set of rules for writing code instructions."
}

# Display sections
display_content("Learning Objectives", learning_objectives)
for i, section in enumerate(content, start=1):
    display_content(f"Objective {i}", [section])

display_content("Glossary", [f"{key}: {value}" for key, value in glossary.items()])
