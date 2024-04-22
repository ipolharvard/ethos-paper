"""Project command line interface: load commands from commands directory."""
import pkgutil
from typing import Any, List

import click

from ethos import commands


class ProjectCLI(click.MultiCommand):
    """Main CLI object for the project.

    Dynamically loads in commands from the
    ethos.commands package. Each module in that
    package is expected to define a command (a function decorated with
    @click.command) with the same name as the module (with underscores
    replaced by hyphens to conform with conventions).

    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Passes through arguments to the base class constructor."""
        super().__init__(self, *args, **kwargs)

        self.avail_modules = {}
        for mod_info in pkgutil.iter_modules(commands.__path__):
            name = mod_info.name
            full_name = f"ethos.commands.{name}"

            spec = mod_info.module_finder.find_spec(name)
            if spec is None:
                continue
            mod = spec.loader.load_module(name)

            if hasattr(mod, name):
                cmd = getattr(mod, name)
                if isinstance(cmd, click.Command):
                    norm_name = name.replace("_", "-")
                    self.avail_modules[norm_name] = cmd
                else:
                    raise TypeError(
                        f"The function {name} in module {full_name} is not of type click.Command."
                    )
            else:
                raise AttributeError(
                    "Each module in the commands package is expected to define a function with "
                    f"the same name as the module. Module {full_name} has no attribute {name}."
                )

    def list_commands(self, ctx: click.Context) -> List[str]:
        """List all commands available through the CLI.

        Parameters
        ----------
        ctx: click.Context
            Context object passed from parent.

        Returns
        ------
        List[str]:
            List of command names that can be called.

        """
        return sorted(list(self.avail_modules.keys()))

    def get_command(self, ctx: click.Context, name: str) -> click.Command:
        """Get a command from the CLI.

        Parameters
        ----------
        ctx: click.Context
            Context object passed from parent.
        name: str
            Name of command. This should match one of the modules within
            the commands package (with underscores replaced with hyphens).

        Returns
        ------
        click.Command:
            Command object implementing the requested command.

        """
        try:
            return self.avail_modules[name]
        except KeyError:
            print(f"No such command found '{name}'. Options are:")

            for n in self.avail_modules.keys():
                print("    ", n)

            raise SystemExit()


def run_cli() -> None:
    """Run project CLI. This is the entrypoint for any project task."""
    cli = ProjectCLI(help="Main entrypoint for the ethos project.")
    cli()
