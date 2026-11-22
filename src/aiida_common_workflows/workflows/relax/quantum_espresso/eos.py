###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Quantum ESPRESSO equation-of-state graph and launch preparation."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import TypedDict

from aiida import orm
from aiida.engine import ExecutionEnvironment, PreparedRun, ToContext, WorkChain, expose, graph, par_map, task

from ..._geometry import scale_structure
from ..._scan import _point_data
from .generator import QuantumEspressoCommonRelaxInputGenerator
from .workchain import QuantumEspressoCommonRelaxWorkChain

_RELAX = QuantumEspressoCommonRelaxWorkChain
_STAGES = tuple(
    name for name in ('base', 'base_final_scf', 'base_relax', 'base_init_relax') if name in _RELAX.spec().inputs
)

_EXECUTION_PORTS = ('metadata', 'pw.code', 'pw.metadata', 'pw.monitors', 'pw.remote_folder')
_EXECUTION = ('metadata', *(f'{stage}.{port}' for stage in _STAGES for port in _EXECUTION_PORTS))


class QuantumEspressoEosPoint(WorkChain):
    """Relax a scaled structure with prepared QE inputs and an optional shared mesh."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.expose_inputs(_RELAX, include=_EXECUTION)
        spec.expose_inputs(_RELAX, namespace='settings', exclude=('structure', *_EXECUTION))
        for namespace in (spec.inputs, spec.inputs['settings']):
            namespace.dynamic = False
            # QE's namespace validator expects the reunited execution and science inputs.
            namespace.validator = None
            for name in _STAGES:
                namespace[name].dynamic = False
                namespace[name]['pw'].dynamic = False
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('factor', valid_type=orm.Float)
        spec.input('reference_mesh', valid_type=orm.KpointsData, required=False)
        spec.output('point', valid_type=orm.DataDict)
        spec.output('kpoints_used', valid_type=orm.KpointsData)
        spec.outline(cls.relax, cls.collect)
        spec.exit_code(400, 'ERROR_RELAXATION_FAILED', message='The QE relaxation failed.')

    def relax(self):
        """Submit the concrete QE relaxation with the scaled structure."""
        self.ctx.scaled = scale_structure(self.inputs.structure, self.inputs.factor)
        builder = _RELAX.get_builder()
        builder._merge(self.exposed_inputs(_RELAX, namespace='settings'))
        execution = self.exposed_inputs(_RELAX)
        # The graph owns this point's CALL label, not the relaxation's.
        if 'metadata' in execution:
            execution['metadata'] = {
                key: value for key, value in execution['metadata'].items() if key != 'call_link_label'
            }
        builder._merge(execution)
        builder.structure = self.ctx.scaled
        if 'reference_mesh' in self.inputs:
            QuantumEspressoCommonRelaxInputGenerator.apply_reference_kpoints(builder, self.inputs.reference_mesh)
        return ToContext(relaxation=self.submit(builder))

    def collect(self):
        """Return the point data and the original mesh used by QE."""
        relaxation = self.ctx.relaxation
        if not relaxation.is_finished_ok:
            return self.exit_codes.ERROR_RELAXATION_FAILED
        self.out('point', _point_data(relaxation, self.ctx.scaled, self.inputs.factor))
        self.out('kpoints_used', relaxation.outputs.kpoints_used)


qe_point = task(QuantumEspressoEosPoint)
QeSettings = expose(qe_point, exclude=('structure', 'factor', 'reference_mesh', *_STAGES))


class EosResults(TypedDict):
    """Reference result and remaining results in scan order."""

    first: orm.DataDict
    remaining: orm.DataList


@graph
def followup(
    factor: orm.Float, structure: orm.StructureData, settings: QeSettings, mesh: orm.KpointsData
) -> orm.DataDict:
    """Run one follow-up relaxation with the shared mesh."""
    result = qe_point(factor=factor, structure=structure, reference_mesh=mesh, settings=settings.settings).alias(
        'followup_relaxation'
    )
    return result.point


@graph
def qe_eos(first: orm.Float, remaining: orm.DataList, structure: orm.StructureData, settings: QeSettings) -> EosResults:
    """Run the QE reference point and then map the rest using its actual mesh."""
    reference = qe_point(factor=first, structure=structure, settings=settings.settings).alias('reference')
    point = partial(followup, structure=structure, settings=settings, mesh=reference.kpoints_used)  # noqa: F821
    results = par_map(point.alias('remaining'), remaining)
    return {'first': reference.point, 'remaining': results}


def prepare(
    *,
    structure: orm.StructureData,
    scale_factors: Sequence[float],
    protocol: str,
    generator_inputs: Mapping,
    overrides: Mapping | None = None,
) -> PreparedRun:
    """Resolve the QE protocol and split scientific inputs from execution settings."""
    if len(scale_factors) < 3 or any(not isfinite(value) or value <= 0 for value in scale_factors):
        msg = 'QE EOS requires at least three finite, positive scale factors.'
        raise ValueError(msg)
    if overrides and 'structure' in overrides:
        msg = 'Set the EOS structure directly, not through subprocess overrides.'
        raise ValueError(msg)
    generator = _RELAX.get_input_generator()
    builder = generator.get_builder(structure=structure, protocol=protocol, **generator_inputs)
    builder._merge(**(overrides or {}))
    settings = builder._inputs(prune=True)
    ports = _RELAX.spec().inputs
    if error := ports.validate(ports.pre_process(settings)):
        msg = f'Invalid QE relaxation inputs: {error}'
        raise ValueError(msg)
    prepared_structure = settings.pop('structure')
    execution = {}
    if 'metadata' in settings:
        execution['metadata'] = settings.pop('metadata')
    for name in _STAGES:
        if name not in settings:
            continue
        stage = settings[name]
        pw = stage['pw']
        pw_execution = {key: pw.pop(key) for key in ('code', 'metadata', 'monitors', 'remote_folder') if key in pw}
        execution[name] = {'pw': pw_execution}
        if 'metadata' in stage:
            execution[name]['metadata'] = stage.pop('metadata')

    factors = [orm.Float(value) for value in scale_factors]
    prepared = PreparedRun(
        inputs={
            'structure': prepared_structure,
            'first': factors[0],
            'remaining': orm.DataList(factors[1:]),
            'settings': {'settings': settings},
        },
        execution_environment=ExecutionEnvironment(
            runtime_config={'reference': execution, 'followup_relaxation': execution}
        ),
        protocol=f'common_workflows.quantum_espresso.{protocol}',
        version='1',
        overrides={'sub_process_ports': sorted(overrides or {})},
    )
    qe_eos.build(prepared_run=prepared)
    return prepared
