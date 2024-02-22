"""List folders at given path."""

from pathlib import Path
from typing import Optional


def list_folders(
    path: Path,
    must_have: Optional[list[str]] = None,
    ignore_hidden: bool = True,
):
    """List folders at the given path.

    Args:
        path (Path): The path to the directory to search for folders.
        must_have (Optional[list[str]], optional): A list of file that
            folders must contain. Defaults to None.
        ignore_hidden (bool, optional): Whether to ignore hidden folders
            (those starting with a dot "."). Defaults to True.

    Returns:
        list[Path]: A list of Path objects representing the matched folders.

    Raises:
        FileNotFoundError: If the provided directory path does not exist.
    """
    # Check path existence
    if not path.is_dir():
        raise FileNotFoundError(f"Directory {path} doesn't exist.")

    # Find matched folders
    matched_folders = []
    for folder in path.iterdir():
        if folder.is_dir() and (not ignore_hidden or not folder.name.startswith(".")):
            if must_have is None or all((folder / file).exists() for file in must_have):
                matched_folders.append(folder)

    return matched_folders
