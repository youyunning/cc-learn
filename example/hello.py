"""Greeting module.

This module provides a simple greeting function and a main entry point.
"""


def greet(name: str) -> None:
    """Print a greeting message to the console.

    Args:
        name: The name of the person to greet.

    Returns:
        None
    """
    message = f"Hello, {name}"
    print(message)


def main() -> None:
    """Run the main entry point of the script.

    Calls greet with a default name to demonstrate usage.
    """
    greet("Claude")


if __name__ == "__main__":
    main()
