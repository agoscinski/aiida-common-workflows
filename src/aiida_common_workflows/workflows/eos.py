"""Equation of state workflow that can use any code plugin implementing the common relax workflow."""
import inspect
from collections.abc import Mapping, Sequence
from importlib import import_module

from aiida import orm
from aiida.common import exceptions
from aiida.engine import GraphHandle, PreparedRun, ToContext, WorkChain
from aiida.plugins import WorkflowFactory

from aiida_common_workflows.workflows.relax.generator import ElectronicType, RelaxType, SpinType
from aiida_common_workflows.workflows.relax.workchain import CommonRelaxWorkChain

from ._geometry import scale_structure as scale_structure  # noqa: PLC0414
from ._scan import point_results, submit_scan


def validate_inputs(value, _):
    """Validate the entire input namespace."""
    if 'scale_factors' not in value:
        if 'scale_count' not in value or 'scale_increment' not in value:
            return 'neither `scale_factors` nor the pair of `scale_count` and `scale_increment` were defined.'

    # Validate that the provided ``generator_inputs`` are valid for the associated input generator.
    process_class = WorkflowFactory(value['sub_process_class'])
    generator = process_class.get_input_generator()

    try:
        generator.get_builder(structure=value['structure'], **value['generator_inputs'])
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


def validate_scale_factors(value, _):
    """Validate the `validate_scale_factors` input."""
    if value and len(value) < 3:
        return 'need at least 3 scaling factors.'


def validate_scale_count(value, _):
    """Validate the `scale_count` input."""
    if value is not None and value < 3:
        return 'need at least 3 scaling factors.'


def validate_scale_increment(value, _):
    """Validate the `scale_increment` input."""
    if value is not None and not 0 < value < 1:
        return 'scale increment needs to be between 0 and 1.'


def validate_relax_type(value, _):
    """Validate the `generator_inputs.relax_type` input."""
    if value is not None and isinstance(value, str):
        value = RelaxType(value)

    if value not in [RelaxType.NONE, RelaxType.POSITIONS, RelaxType.SHAPE, RelaxType.POSITIONS_SHAPE]:
        return '`generator_inputs.relax_type`. Equation of state and relaxation with variable volume not compatible.'


_QE_BACKEND = 'common_workflows.relax.quantum_espresso'


def prepare_eos(  # noqa: PLR0913
    *,
    backend: str,
    structure: orm.StructureData,
    scale_factors: Sequence[float],
    protocol: str,
    generator_inputs: Mapping,
    overrides: Mapping | None = None,
) -> tuple[GraphHandle, PreparedRun]:
    """Select an EOS graph and resolve its inputs before launch.

    ``backend`` is a common relaxation entry point name. Only Quantum ESPRESSO
    currently provides a prepared EOS graph; other backends use the legacy API.

    :param backend: Common relaxation entry point.
    :param structure: Unscaled reference structure.
    :param scale_factors: Ordered volume factors; the first selects the reference.
    :param protocol: Backend protocol name.
    :param generator_inputs: Inputs to the backend protocol resolver, such as engines and relaxation type.
    :param overrides: Optional concrete relaxation port overrides.
    :return: Concrete graph and its prepared launch settings.
    :raises ValueError: If the backend or EOS inputs are unsupported.
    """
    if backend != _QE_BACKEND:
        msg = f'No prepared EOS graph is available for {backend!r}.'
        raise ValueError(msg)
    if error := validate_relax_type(generator_inputs.get('relax_type'), None):
        raise ValueError(error)
    module = import_module('aiida_common_workflows.workflows.relax.quantum_espresso.eos')
    return module.qe_eos, module.prepare(
        structure=structure, scale_factors=scale_factors, protocol=protocol,
        generator_inputs=generator_inputs, overrides=overrides,
    )


class EquationOfStateWorkChain(WorkChain):
    """Workflow to compute the equation of state for a given crystal structure."""

    @classmethod
    def define(cls, spec):
        # yapf: disable
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData, help='The structure at equilibrium volume.')
        spec.input('scale_factors', valid_type=orm.List, required=False,
            validator=validate_scale_factors, serializer=orm.to_aiida_type,
            help='The list of scale factors at which the volume and total energy of the structure should be computed.')
        spec.input('scale_count', valid_type=orm.Int, default=lambda: orm.Int(7),
            validator=validate_scale_count, serializer=orm.to_aiida_type,
            help='The number of points to compute for the equation of state.')
        spec.input('scale_increment', valid_type=orm.Float, default=lambda: orm.Float(0.02),
            validator=validate_scale_increment, serializer=orm.to_aiida_type,
            help='The relative difference between consecutive scaling factors.')
        spec.input_namespace('generator_inputs',
            help='The inputs that will be passed to the input generator of the specified `sub_process`.')
        spec.input('generator_inputs.engines', valid_type=dict, non_db=True)
        spec.input('generator_inputs.protocol', valid_type=str, non_db=True,
            help='The protocol to use when determining the workchain inputs.')
        spec.input('generator_inputs.relax_type',
             valid_type=(RelaxType, str), non_db=True, validator=validate_relax_type,
            help='The type of relaxation to perform.')
        spec.input('generator_inputs.spin_type', valid_type=(SpinType, str), required=False, non_db=True,
            help='The type of spin for the calculation.')
        spec.input('generator_inputs.electronic_type', valid_type=(ElectronicType, str), required=False, non_db=True,
            help='The type of electronics (insulator/metal) for the calculation.')
        spec.input('generator_inputs.magnetization_per_site', valid_type=(list, tuple), required=False, non_db=True,
            help='List containing the initial magnetization per atomic site.')
        spec.input('generator_inputs.fixed_total_cell_magnetization', valid_type=float, required=False, non_db=True,
            help='The fixed total magnetization of the cell in Bohr magnetons (μB).')
        spec.input('generator_inputs.threshold_forces', valid_type=float, required=False, non_db=True,
            help='Target threshold for the forces in eV/Å.')
        spec.input('generator_inputs.threshold_stress', valid_type=float, required=False, non_db=True,
            help='Target threshold for the stress in eV/Å^3.')
        spec.input_namespace('sub_process', dynamic=True, populate_defaults=False)
        spec.input('sub_process_class', non_db=True, validator=validate_sub_process_class)
        spec.inputs.validator = validate_inputs
        spec.outline(
            cls.run_scan,
            cls.inspect_eos,
        )
        spec.output_namespace('structures', valid_type=orm.StructureData,
            help='The relaxed structures at each scaling factor.')
        spec.output_namespace('total_energies', valid_type=orm.Float,
            help='The computed total energy of the relaxed structures at each scaling factor.')
        spec.output_namespace('total_magnetizations', valid_type=orm.Float,
            help='The computed total magnetization of the relaxed structures at each scaling factor.')
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED',
            message='At least one of the `{cls}` sub processes did not finish successfully.')

    def get_scale_factors(self):
        """Return the list of scale factors."""
        if 'scale_factors' in self.inputs:
            return tuple(self.inputs.scale_factors)

        count = self.inputs.scale_count.value
        increment = self.inputs.scale_increment.value
        return tuple(float(1 + i * increment - (count - 1) * increment / 2) for i in range(count))

    def run_scan(self):
        """Submit the prepared QE EOS graph or fall back to legacy backends."""
        if self.inputs.sub_process_class == _QE_BACKEND and 'structure' not in self.inputs.get('sub_process', {}):
            generator_inputs = dict(self.inputs.generator_inputs)
            protocol = generator_inputs.pop('protocol')
            workflow, prepared = prepare_eos(
                backend=_QE_BACKEND,
                structure=self.inputs.structure,
                scale_factors=self.get_scale_factors(),
                protocol=protocol,
                generator_inputs=generator_inputs,
                overrides=self.inputs.get('sub_process'),
            )
            return ToContext(scan=self.submit(workflow, prepared_run=prepared))
        return submit_scan(
            self, structure=self.inputs.structure, coordinates=self.get_scale_factors(), mode='volume', reference=True
        )

    def inspect_eos(self):
        """Publish structures, energies and available magnetizations in scale order."""
        if not self.ctx.scan.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED.format(cls=self.inputs.sub_process_class)

        for index, (outputs, _, structure) in enumerate(point_results(self.ctx.scan)):
            energy = outputs['total_energy']
            self.report(f'Image {index}: volume={structure.get_cell_volume()}, total energy={energy.value}')
            self.out(f'structures.{index}', structure)
            self.out(f'total_energies.{index}', energy)
            if 'total_magnetization' in outputs:
                self.out(f'total_magnetizations.{index}', outputs['total_magnetization'])
