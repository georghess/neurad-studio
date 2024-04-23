# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""This file contains the configuration for external methods which are not included in this repository."""
import inspect
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tyro
from rich.prompt import Confirm

from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ExternalMethod:
    """External method class. Represents a link to a nerfstudio-compatible method not included in this repository."""

    instructions: str
    """Instructions for installing the method. This will be printed to
    the console when the user tries to use the method."""
    configurations: List[Tuple[str, str]]
    """List of configurations for the method. Each configuration is a tuple of (registered slug, description)
    as it will be printed in --help."""
    pip_package: Optional[str] = None
    """Specifies a pip package if the method can be installed by running `pip install <pip_package>`."""


external_methods = []

external_methods.append(
    ExternalMethod(
        """[bold yellow]UniSim[/bold yellow]

To enable UniSim, you must install it first by running:
  [grey]pip install git+https://github.com/carlinds/unisim[/grey]""",
        configurations=[
            ("unisim", "UniSim reproduction, as specified in the paper"),
            ("unisim++", "UniSim with some improvements/tweaks"),
        ],
        pip_package="git+https://github.com/carlinds/unisim",
    )
)


@dataclass
class ExternalMethodDummyTrainerConfig:
    """Dummy trainer config for external methods (a) which do not have an
    implementation in this repository, and (b) are not yet installed. When this
    config is instantiated, we give the user the option to install the method.
    """

    # tyro.conf.Suppress will prevent these fields from appearing as CLI arguments.
    method_name: tyro.conf.Suppress[str]
    method: tyro.conf.Suppress[ExternalMethod]

    def __post_init__(self):
        """Offer to install an external method."""

        # Don't trigger install message from get_external_methods() below; only
        # if this dummy object is instantiated from the CLI.
        if inspect.stack()[2].function == "get_external_methods":
            return

        CONSOLE.print(self.method.instructions)
        if self.method.pip_package and Confirm.ask(
            "\nWould you like to run the install it now?", default=False, console=CONSOLE
        ):
            # Install the method
            install_command = f"{sys.executable} -m pip install {self.method.pip_package}"
            CONSOLE.print(f"Running: [cyan]{install_command}[/cyan]")
            result = subprocess.run(install_command, shell=True, check=False)
            if result.returncode != 0:
                CONSOLE.print("[bold red]Error installing method.[/bold red]")
                sys.exit(1)

        sys.exit(0)


def get_external_methods() -> Tuple[Dict[str, ExternalMethodDummyTrainerConfig], Dict[str, str]]:
    """Returns the external methods trainer configs and the descriptions."""
    method_configs: Dict[str, ExternalMethodDummyTrainerConfig] = {}
    descriptions: Dict[str, str] = {}
    for external_method in external_methods:
        for config_slug, config_description in external_method.configurations:
            method_configs[config_slug] = ExternalMethodDummyTrainerConfig(
                method_name=config_slug, method=external_method
            )
            descriptions[config_slug] = f"""[External, run 'ns-train {config_slug}' to install] {config_description}"""
    return method_configs, descriptions
