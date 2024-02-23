"""Pytest unit test for Config class."""

import pytest

from cnn4dos.config import Config
from cnn4dos.utils import ROOT_DIR


class Test_config:
    test_config_file = ROOT_DIR / "cnn4dos" / "config" / "config_template.yaml"
    non_existent_config_file = ROOT_DIR / "non_existent_folder" / "config.yaml"

    def test_read_config(self) -> None:
        config_reader = Config(self.test_config_file)
        config = config_reader.read()

        assert isinstance(config, dict)

    def test_config_not_exist(self) -> None:
        with pytest.raises(FileNotFoundError):
            config_reader = Config(self.non_existent_config_file)
            config_reader.read()
