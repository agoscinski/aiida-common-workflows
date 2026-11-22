###########################################################################
# Copyright (c), The AiiDA team. All rights reserved.                     #
# This file is part of the AiiDA code.                                    #
#                                                                         #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-core #
# For further information on the license, see the LICENSE.txt file        #
# For further information please visit http://www.aiida.net               #
###########################################################################
"""Exercise the scan WorkChain API, graph scheduling and runtime backend dispatch together."""

from math import dist

import pytest
from aiida import orm
from aiida.engine import WorkChain, calcfunction, run_get_node

from aiida_common_workflows.workflows._scan import (
    LegacyReferenceScanWorkChain,
    ReferencePointWorkChain,
    ScanPointWorkChain,
    independent_scan,
    reference_scan,
)
from aiida_common_workflows.workflows.dissociation import DissociationCurveWorkChain
from aiida_common_workflows.workflows.em import EnergyMagnetizationWorkChain
from aiida_common_workflows.workflows.eos import EquationOfStateWorkChain
from aiida_common_workflows.workflows.relax.workchain import CommonRelaxWorkChain


@calcfunction
def make_energy(value):
    return value.clone()


@calcfunction
def make_mesh(structure):
    mesh = orm.KpointsData()
    mesh.set_cell_from_structure(structure)
    mesh.set_kpoints_mesh([2, 3, 4])
    return mesh


class Backend(WorkChain):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('energy', valid_type=orm.Float)
        spec.input('marker', valid_type=orm.Str)
        spec.input('kpoints', valid_type=orm.KpointsData, required=False)
        spec.input('magnetization', valid_type=orm.Float, required=False)
        spec.input('fail', valid_type=orm.Bool, required=False)
        spec.output('energy', valid_type=orm.Float)
        spec.output('kpoints_used', valid_type=orm.KpointsData)
        spec.outline(cls.compute)
        spec.exit_code(401, 'ERROR_TEST_FAILURE', message='Requested failure.')

    def compute(self):
        if self.inputs.get('fail', False):
            return self.exit_codes.ERROR_TEST_FAILURE
        self.out('energy', make_energy(self.inputs.energy))
        self.out('kpoints_used', self.inputs.kpoints if 'kpoints' in self.inputs else make_mesh(self.inputs.structure))


class ScanTestRelax(CommonRelaxWorkChain):
    _process_class = Backend

    @classmethod
    def get_input_generator(cls):
        return cls

    @staticmethod
    def supports_feature(feature):
        return True

    @classmethod
    def get_builder(
        cls, *, structure=None, engines=None, reference_workchain=None, fixed_total_cell_magnetization=None, **kwargs
    ):
        builder = super().get_builder()
        if structure is None:
            return builder
        builder.structure = structure
        builder.marker = engines['relax']['marker']
        builder.energy = orm.Float(
            fixed_total_cell_magnetization
            if fixed_total_cell_magnetization is not None
            else structure.get_cell_volume()
        )
        if reference_workchain is not None:
            assert reference_workchain.is_finished_ok
            builder.kpoints = reference_workchain.called[0].outputs.kpoints_used
        if fixed_total_cell_magnetization is not None:
            builder.magnetization = orm.Float(fixed_total_cell_magnetization)
        return builder

    def convert_outputs(self):
        energy = self.ctx.workchain.outputs.energy
        self.out('total_energy', energy)
        if 'magnetization' in self.inputs:
            self.out('total_magnetization', self.inputs.magnetization)
            self.out('fermi_energy_up', energy)
            self.out('fermi_energy_down', energy)


class ScanTestMeshRelax(ScanTestRelax):
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.output('kpoints_used', valid_type=orm.KpointsData)

    @staticmethod
    def apply_reference_kpoints(builder, kpoints):
        builder.kpoints = kpoints

    def convert_outputs(self):
        super().convert_outputs()
        self.out('kpoints_used', self.ctx.workchain.outputs.kpoints_used)


@pytest.fixture
def scan_inputs(entry_points):
    entry_points.add(ScanTestMeshRelax, 'aiida.workflows:scan_test.relax')
    return {
        'sub_process_class': 'scan_test.relax',
        'generator_inputs': {
            'engines': {'relax': {'marker': orm.Str('generator').store()}},
            'protocol': 'test',
            'relax_type': 'none',
        },
    }


def test_scan_graphs_build():
    assert reference_scan.build() is None
    assert independent_scan.build() is None


def test_volume_scan_reuses_reference_and_preserves_outputs(scan_inputs, generate_structure):
    override = orm.Str('override')
    results, node = run_get_node(
        EquationOfStateWorkChain,
        **scan_inputs,
        structure=generate_structure(['Si']),
        scale_factors=[1.0, 0.9, 1.1],
        sub_process={'marker': override},
    )
    assert node.is_finished_ok
    assert [results['total_energies'][str(i)].value for i in range(3)] == pytest.approx([1, 0.9, 1.1])
    assert not results.get('total_magnetizations')
    graph_node = next(child for child in node.called if isinstance(child, orm.GraphNode))
    (reference,) = [child for child in graph_node.called if child.process_label == ReferencePointWorkChain.__name__]
    points = [child for child in graph_node.called if child.process_label == ScanPointWorkChain.__name__]
    assert len(points) == 2
    mesh = reference.outputs.kpoints_used
    assert mesh.get_kpoints_mesh()[0] == [2, 3, 4]
    assert all(point.inputs.reference_kpoints.pk == mesh.pk for point in points)
    relaxations = [child for child in graph_node.called_descendants if child.process_label == Backend.__name__]
    assert len(relaxations) == 3
    assert all(child.inputs.marker.pk == override.pk for child in relaxations)
    followups = [child for child in relaxations if 'kpoints' in child.inputs]
    assert len(followups) == 2
    assert all(child.inputs.kpoints.pk == mesh.pk for child in followups)
    first_relaxation = next(child for child in reference.called if child.process_label == ScanTestMeshRelax.__name__)
    assert results['total_energies']['0'].pk != first_relaxation.outputs.total_energy.pk
    assert results['total_energies']['0'].creator.process_label == '_unpack_point'
    assert [results['structures'][str(i)].get_cell_volume() for i in range(3)] == pytest.approx([1, 0.9, 1.1])


def test_dissociation_scan_preserves_distances(scan_inputs, generate_structure):
    results, node = run_get_node(
        DissociationCurveWorkChain,
        **scan_inputs,
        molecule=generate_structure(['H', 'H']),
        distances=orm.List([1.5, 0.7]),
    )
    assert node.is_finished_ok
    assert [results['distances'][str(i)].value for i in range(2)] == [1.5, 0.7]
    assert not results.get('total_magnetizations')
    molecules = [child.inputs.structure for child in node.called_descendants if child.process_label == Backend.__name__]
    actual = sorted(dist(*[site.position for site in molecule.sites]) for molecule in molecules)
    assert actual == pytest.approx([0.7, 1.5])


def test_magnetization_scan_preserves_order_and_outputs(scan_inputs, generate_structure):
    values = [2, 0, -1]
    results, node = run_get_node(
        EnergyMagnetizationWorkChain,
        **scan_inputs,
        structure=generate_structure(['Si']),
        fixed_total_magnetizations=values,
    )
    assert node.is_finished_ok
    for key in ['total_energies', 'total_magnetizations', 'fermi_energies_up', 'fermi_energies_down']:
        assert [results[key][str(i)].value for i in range(3)] == values


def test_failed_reference_does_not_start_remaining_points(scan_inputs, generate_structure):
    results, node = run_get_node(
        EquationOfStateWorkChain,
        **scan_inputs,
        structure=generate_structure(['Si']),
        scale_factors=[1.0, 0.9, 1.1],
        sub_process={'fail': orm.Bool(True)},
    )
    assert node.exit_status == 400
    assert results == {}
    graph_node = next(child for child in node.called if isinstance(child, orm.GraphNode))
    assert [child.process_label for child in graph_node.called] == [ReferencePointWorkChain.__name__]


def test_legacy_backend_still_receives_reference_node(scan_inputs, entry_points, generate_structure):
    entry_points.add(ScanTestRelax, 'aiida.workflows:scan_test.legacy')
    scan_inputs['sub_process_class'] = 'scan_test.legacy'
    results, node = run_get_node(
        EquationOfStateWorkChain,
        **scan_inputs,
        structure=generate_structure(['Si']),
        scale_factors=[1, 0.9, 1.1],
    )
    assert node.is_finished_ok
    assert [results['total_energies'][str(i)].value for i in range(3)] == pytest.approx([1, 0.9, 1.1])
    (legacy,) = [child for child in node.called if child.process_label == LegacyReferenceScanWorkChain.__name__]
    relaxations = [child for child in legacy.called if child.process_label == ScanTestRelax.__name__]
    assert len(relaxations) == 3
    (first,) = [child for child in relaxations if 'kpoints' not in child.inputs]
    mesh = first.called[0].outputs.kpoints_used
    assert all(child.inputs.kpoints.pk == mesh.pk for child in relaxations if child != first)
