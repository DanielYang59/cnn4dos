"""Handle config yaml file."""

# TODO: add internal checking methods

from pathlib import Path

import yaml


class Config:
    def __init__(self, file: Path) -> None:
        file = Path(file)
        if not file.is_file():
            raise FileNotFoundError("Config file not found.")

        self.file = file

    def _check(self) -> None:
        # TODO
        pass

    def read(self) -> dict:
        with open(self.file, "r") as file:
            config = yaml.safe_load(file)
        return config
