"""Command line interface ``acwf``."""
import click
from aiida.cmdline.groups import VerdiCommandGroup
from aiida.cmdline.params import options


@click.group('acwf', cls=VerdiCommandGroup, context_settings={'help_option_names': ['-h', '--help']})
@options.PROFILE(expose_value=False)
def cmd_root():
    """CLI for the ``aiida-common-workflows`` plugin."""
