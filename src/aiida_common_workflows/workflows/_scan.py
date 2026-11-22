###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Graph scheduling shared by the equation-of-state, dissociation and magnetization scans."""

from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from typing import TypedDict

from aiida import orm
from aiida.engine import ToContext, WorkChain, append_, calcfunction, graph, par_map, task
from aiida.plugins import WorkflowFactory

from ._geometry import scale_structure, set_distance
from .relax.workchain import CommonRelaxWorkChain


@calcfunction
def _make_request(settings: orm.Dict, **nodes: orm.Data) -> orm.Dict:
    """Carry the open builder configuration through a static port, linking its original data nodes."""
    return settings.clone()


def _pack_request(generator_inputs: Mapping, overrides: Mapping) -> orm.Dict:
    nodes = {}

    def encode(value: object) -> list:
        if isinstance(value, orm.Data):
            key = f'node_{len(nodes)}'
            nodes[key] = value
            return ['node', key]
        if isinstance(value, Mapping):
            return ['mapping', [[key, encode(item)] for key, item in value.items()]]
        if isinstance(value, (list, tuple)):
            return ['tuple' if isinstance(value, tuple) else 'list', [encode(item) for item in value]]
        if isinstance(value, Enum):
            value = value.value
        return ['value', value]

    settings = {'generator_inputs': encode(generator_inputs), 'overrides': encode(overrides)}
    return _make_request(orm.Dict(dict=settings), **nodes)


def _unpack_request(request: orm.Dict) -> tuple[dict, dict]:
    def decode(encoded: list) -> object:
        kind, value = encoded
        if kind == 'node':
            return request.creator.inputs[value]
        if kind == 'mapping':
            return {key: decode(item) for key, item in value}
        if kind == 'tuple':
            return tuple(map(decode, value))
        if kind == 'list':
            return list(map(decode, value))
        return value

    return decode(request['generator_inputs']), decode(request['overrides'])


# A point has heterogeneous outputs. Each field uses a singleton DataList so the
# outer DataDict is homogeneous and par_map still sees one data node per point.
@calcfunction
def _pack_point(**fields: orm.Data) -> orm.DataDict:
    """Pack the point's data, not a process identifier."""
    return orm.DataDict({name: orm.DataList([value]) for name, value in fields.items()})


@calcfunction
def _unpack_point(point: orm.DataDict) -> dict[str, orm.Data]:
    """Create standalone outputs that the public WorkChain can return."""
    return {name: point[name][0].clone() for name in point}


def _point_data(child: orm.WorkChainNode, structure: orm.StructureData, coordinate: orm.Float) -> orm.DataDict:
    outputs = child.outputs
    fields = {
        'coordinate': coordinate,
        'structure': outputs.relaxed_structure if 'relaxed_structure' in outputs else structure,
        'total_energy': outputs.total_energy,
    }
    for name in ('total_magnetization', 'fermi_energy_up', 'fermi_energy_down'):
        if name in outputs:
            fields[name] = outputs[name]
    return _pack_point(**fields)


def _build_point(inputs, coordinate, *, reference_workchain=None):
    """Build a relaxation with the workflow's geometry, protocol and overrides."""
    process_class = WorkflowFactory(inputs.backend.value)
    if not isinstance(process_class, type) or not issubclass(process_class, CommonRelaxWorkChain):
        msg = 'The scan backend must be a CommonRelaxWorkChain subclass.'
        raise ValueError(msg)

    structure = inputs.structure
    mode = inputs.mode.value
    generator_inputs, overrides = _unpack_request(inputs.request)
    if mode == 'volume':
        structure = scale_structure(structure, coordinate)
    elif mode == 'distance':
        structure = set_distance(structure, coordinate)
    elif mode == 'magnetization':
        generator_inputs['fixed_total_cell_magnetization'] = coordinate.value
    else:
        msg = f'Unknown scan mode {mode!r}.'
        raise ValueError(msg)

    if reference_workchain is not None:
        generator_inputs['reference_workchain'] = reference_workchain

    generator = process_class.get_input_generator()
    builder = generator.get_builder(structure=structure, **generator_inputs)
    builder._merge(**overrides)
    # Apply the shared mesh last: a per-point density or mesh override must not
    # silently take precedence over the mesh actually used by the reference.
    if 'reference_kpoints' in inputs:
        generator.apply_reference_kpoints(builder, inputs.reference_kpoints)
    return builder, structure


def _define_scan_inputs(spec):
    spec.input('structure', valid_type=orm.StructureData)
    spec.input('mode', valid_type=orm.Str)
    spec.input('backend', valid_type=orm.Str)
    spec.input('request', valid_type=orm.Dict)


class ScanPointWorkChain(WorkChain):
    """Adapt runtime-selected relaxation builders to a statically typed graph placement."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        _define_scan_inputs(spec)
        spec.input('coordinate', valid_type=orm.Float)
        spec.input('reference_kpoints', valid_type=orm.KpointsData, required=False)
        spec.output('point', valid_type=orm.DataDict)
        spec.outline(cls.run_backend, cls.record_point)
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED', message='The relaxation subprocess failed.')

    def run_backend(self):
        """Build and submit the selected plugin's relaxation for this point."""
        builder, self.ctx.structure = _build_point(self.inputs, self.inputs.coordinate)
        return ToContext(child=self.submit(builder))

    def record_point(self):
        """Publish scientific outputs after a successful relaxation."""
        child = self.ctx.child
        self.report(f'{child.process_label}<{child.pk}> finished with exit status {child.exit_status}.')
        if not child.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        self.out('point', _point_data(child, self.ctx.structure, self.inputs.coordinate))


class ReferencePointWorkChain(ScanPointWorkChain):
    """Publish the actual k-point mesh on a required graph output port."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.output('kpoints_used', valid_type=orm.KpointsData)

    def record_point(self):
        """Forward the exact mesh node returned by the backend, without copying it."""
        if (failure := super().record_point()) is not None:
            return failure
        self.out('kpoints_used', self.ctx.child.outputs.kpoints_used)


class LegacyReferenceScanWorkChain(WorkChain):
    """Keep the original reference protocol for plugins that still require a process node."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        _define_scan_inputs(spec)
        spec.input('coordinates', valid_type=orm.List)
        spec.output('result', valid_type=orm.DataList)
        spec.outline(cls.run_reference, cls.run_remaining, cls.collect)
        spec.exit_code(400, 'ERROR_SUB_PROCESS_FAILED', message='A relaxation subprocess failed.')

    def run_reference(self):
        """Start the first relaxation and keep its process in the WorkChain context."""
        self.ctx.structures = []
        self.ctx.coordinates = []
        return ToContext(first=self._submit_point(self.inputs.coordinates[0]))

    def run_remaining(self):
        """Give the original process node to the legacy input generators."""
        if not self.ctx.first.is_finished_ok:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        for value in self.inputs.coordinates[1:]:
            self.to_context(remaining=append_(self._submit_point(value, reference_workchain=self.ctx.first)))

    def collect(self):
        """Publish data in submission order."""
        children = [self.ctx.first, *self.ctx.get('remaining', [])]
        if any(not child.is_finished_ok for child in children):
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED
        points = [
            _point_data(child, structure, coordinate)
            for child, structure, coordinate in zip(children, self.ctx.structures, self.ctx.coordinates, strict=True)
        ]
        self.out('result', _pack_scan(*points))

    def _submit_point(self, value: float, *, reference_workchain: orm.WorkChainNode | None = None):
        coordinate = orm.Float(value)
        builder, structure = _build_point(self.inputs, coordinate, reference_workchain=reference_workchain)
        self.ctx.structures.append(structure)
        self.ctx.coordinates.append(coordinate)
        return self.submit(builder)


@calcfunction
def _pack_scan(*points: orm.DataDict) -> orm.DataList:
    """Collect data from the compatibility WorkChain in submission order."""
    return orm.DataList(points)


scan_point = task(ScanPointWorkChain)
reference_point = task(ReferencePointWorkChain)


class ReferenceScanOutputs(TypedDict):
    first: orm.DataDict
    remaining: orm.DataList


@graph
def reference_scan(  # noqa: PLR0913, PLR0917
    first: orm.Float, remaining: orm.DataList, structure: orm.StructureData, mode: str, backend: str, request: orm.Dict
) -> ReferenceScanOutputs:
    """Run a reference point, then fan out the remaining points using its actual mesh."""
    initial = reference_point(coordinate=first, structure=structure, mode=mode, backend=backend, request=request)
    run_point = partial(scan_point, structure=structure, mode=mode, backend=backend, request=request)  # noqa: F821
    run_remaining = partial(run_point, reference_kpoints=initial.kpoints_used)  # noqa: F821
    points = par_map(run_remaining.alias('remaining'), remaining)
    return {'first': initial.point, 'remaining': points}


@graph
def unreferenced_point(
    coordinate: orm.Float, structure: orm.StructureData, mode: str, backend: str, request: orm.Dict
) -> orm.DataDict:
    """Expose a unary map target after binding shared inputs, without an optional reference port."""
    point = scan_point(coordinate=coordinate, structure=structure, mode=mode, backend=backend, request=request)
    return point.point


@graph
def independent_scan(
    coordinates: orm.DataList, structure: orm.StructureData, mode: str, backend: str, request: orm.Dict
) -> orm.DataList:
    """Run independent points in parallel."""
    run_point = partial(unreferenced_point, structure=structure, mode=mode, backend=backend, request=request)  # noqa: F821
    return par_map(run_point, coordinates)


def submit_scan(
    workflow: WorkChain, *, structure: orm.StructureData, coordinates: Sequence[float], mode: str, reference: bool
) -> ToContext:
    """Launch graph orchestration, or the compatibility path for legacy references."""
    backend = workflow.inputs.sub_process_class
    common = {
        'structure': structure,
        'mode': mode,
        'backend': backend,
        'request': _pack_request(workflow.inputs.generator_inputs, workflow.inputs.get('sub_process', {})),
    }
    if reference and 'kpoints_used' not in WorkflowFactory(backend).spec().outputs:
        scan = workflow.submit(LegacyReferenceScanWorkChain, coordinates=orm.List(list(coordinates)), **common)
    else:
        points = [orm.Float(value) for value in coordinates]
        if reference:
            scan = workflow.submit(reference_scan, first=points[0], remaining=orm.DataList(points[1:]), **common)
        else:
            scan = workflow.submit(independent_scan, coordinates=orm.DataList(points), **common)
    return ToContext(scan=scan)


def point_results(scan: orm.WorkflowNode) -> Iterator[tuple[dict[str, orm.Data], orm.Float, orm.StructureData]]:
    """Read ordered outputs from the owned map results."""
    if 'first' in scan.outputs:
        points = [scan.outputs.first, *scan.outputs.remaining]
    else:
        points = scan.outputs.result
    for point in points:
        outputs = _unpack_point(point)
        yield outputs, outputs.pop('coordinate'), outputs.pop('structure')
