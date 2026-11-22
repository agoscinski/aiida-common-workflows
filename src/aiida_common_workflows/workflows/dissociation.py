"""
Workflow calculating the dissociation curve of diatomic molecules.
It can use any code plugin implementing the common relax workflow.
"""
import inspect

from aiida import orm
from aiida.common import exceptions
from aiida.engine import WorkChain
from aiida.plugins import WorkflowFactory

from aiida_common_workflows.workflows.relax.generator import ElectronicType, RelaxType, SpinType
from aiida_common_workflows.workflows.relax.workchain import CommonRelaxWorkChain

from ._geometry import set_distance as set_distance  # noqa: PLC0414
from ._scan import point_results, submit_scan


def validate_inputs(value, _):
    """Validate the entire input namespace."""
    if 'distances' not in value:
        if any(key not in value for key in ['distances_count', 'distance_min', 'distance_max']):
            return 'neither `distances` nor the `distances_count`, `distance_min`, and `distance_max` set were defined.'
        if value['distance_min'] >= value['distance_max']:
            return '`distance_min` must be smaller than `distance_max`'

    # Validate that the provided ``generator_inputs`` are valid for the associated input generator.
    process_class = WorkflowFactory(value['sub_process_class'])
    generator = process_class.get_input_generator()

    try:
        generator.get_builder(structure=value['molecule'], **value['generator_inputs'])
    except Exception as exc:
        return f'`{generator.__class__.__name__}.get_builder()` fails for the provided `generator_inputs`: {exc}'


def validate_sub_process_class(value, _):
    """Validate the sub process class."""
    try:
        process_class = WorkflowFactory(value)
    except exceptions.EntryPointError:
        return f'`{value}` is not a valid or registered workflow entry point.'

    if not inspect.isclass(process_class) or not issubclass(process_class, CommonRelaxWorkChain):
        return f'`{value}` is not a subclass of the `CommonRelaxWorkChain` common workflow.'


def validate_molecule(value, _):
    """Validate the `molecule` input."""
    if len(value.sites) != 2:
        return '`molecule`. only diatomic molecules are supported.'


def validate_distances(value, _):
    """Validate the `dinstances` input."""
    if value and len(value) < 2:
        return 'need at least 2 distances.'
    if value:
        for dist in value:
            if dist < 0.0:
                return 'distances must be positive.'


def validate_distances_count(value, _):
    """Validate the `dinstances_count` input."""
    if value is not None and value < 2:
        return 'need at least 2 distances.'


def validate_distance_max(value, _):
    """Validate the `distance_max` input."""
    if value is not None and value <= 0:
        return '`distance_max` must be bigger than zero.'


def validate_distance_min(value, _):
    """Validate the `distance_min` input."""
    if value is not None and value < 0:
        return '`distance_min` must be bigger than zero.'


def validate_relax(value, _):
    """Validate the `generator_inputs.relax_type` input."""
    if value is not None and isinstance(value, str):
        value = RelaxType(value)

    if value is not RelaxType.NONE:
        return '`generator_inputs.relax_type`. Only `RelaxType.NONE` supported.'



class DissociationCurveWorkChain(WorkChain):
    """Workflow to compute the dissociation curve of for a given diatomic molecule."""

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.input('molecule', valid_type=orm.StructureData, validator=validate_molecule, help='The input molecule.')
        spec.input('distances', valid_type=orm.List, required=False, validator=validate_distances,
            help='The list of distances in Ångstrom at which the total energy of the molecule should be computed.')
        spec.input('distances_count', valid_type=orm.Int, default=lambda: orm.Int(20),
            validator=validate_distances_count,
            help='The number of points to compute in the dissociation curve.')
        spec.input('distance_min', valid_type=orm.Float, default=lambda: orm.Float(0.5),
            validator=validate_distance_min,
            help='The minimum tested distance in Ångstrom.')
        spec.input('distance_max', valid_type=orm.Float, default=lambda: orm.Float(3),
            validator=validate_distance_max,
            help='The maximum tested distance in Ångstrom.')
        spec.input_namespace('generator_inputs',
            help='The inputs that will be passed to the input generator of the specified `sub_process`.')
        spec.input('generator_inputs.engines', valid_type=dict, non_db=True)
        spec.input('generator_inputs.protocol', valid_type=str, non_db=True,
            help='The protocol to use when determining the workchain inputs.')
        spec.input('generator_inputs.relax_type', valid_type=(RelaxType, str), non_db=True, validator=validate_relax,
            help='The type of relaxation to perform.')
        spec.input('generator_inputs.spin_type', valid_type=(SpinType, str), required=False, non_db=True,
            help='The type of spin for the calculation.')
        spec.input('generator_inputs.electronic_type', valid_type=(ElectronicType, str), required=False, non_db=True,
            help='The type of electronics (insulator/metal) for the calculation.')
        spec.input('generator_inputs.magnetization_per_site', valid_type=(list, tuple), required=False, non_db=True,
            help='List containing the initial magnetization fer each site.')
        spec.input_namespace('sub_process', dynamic=True, populate_defaults=False)
        spec.input('sub_process_class', non_db=True, validator=validate_sub_process_class)
        spec.inputs.validator = validate_inputs
        spec.outline(
            cls.run_scan,
            cls.inspect_results,
        )
        spec.output_namespace('distances', valid_type=orm.Float,
            help='The distance between the two atoms.')
        spec.output_namespace('total_energies', valid_type=orm.Float,
            help='The computed total energy of the molecule at each distance.')
        spec.output_namespace('total_magnetizations', valid_type=orm.Float,
            help='The computed total magnetization of the molecule at each distance.')
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED',
            message='At least one of the `{cls}` sub processes did not finish successfully.')

    def get_distances(self):
        """Return the list of scale factors."""
        if 'distances' in self.inputs:
            return [orm.Float(i) for i in self.inputs.distances]

        count = self.inputs.distances_count.value
        maximum = self.inputs.distance_max.value
        minimum = self.inputs.distance_min.value
        return [orm.Float(minimum + i * (maximum - minimum) / (count - 1)) for i in range(count)]

    def run_scan(self):
        """Submit the reference-first distance scan."""
        return submit_scan(
            self, structure=self.inputs.molecule, coordinates=[distance.value for distance in self.get_distances()],
            mode='distance', reference=True,
        )

    def inspect_results(self):
        """Publish distances, energies and available magnetizations in input order."""
        if not self.ctx.scan.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(cls=self.inputs.sub_process_class)

        for index, (outputs, distance, _) in enumerate(point_results(self.ctx.scan)):
            energy = outputs['total_energy']
            self.report(f'Image {index}: distance={distance.value}, total energy={energy.value}')
            self.out(f'distances.{index}', distance)
            self.out(f'total_energies.{index}', energy)
            if 'total_magnetization' in outputs:
                self.out(f'total_magnetizations.{index}', outputs['total_magnetization'])
