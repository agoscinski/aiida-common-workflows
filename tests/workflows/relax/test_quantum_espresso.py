"""Tests for the :mod:`aiida_common_workflows.workflows.relax.quantum_espresso` module."""

import pytest
from aiida import engine, orm, plugins
from aiida.common import LinkType
from aiida.engine import WorkChain, calcfunction, run_get_node

from aiida_common_workflows.workflows.relax.generator import ElectronicType, RelaxType, SpinType


@pytest.fixture
def generator():
    return plugins.WorkflowFactory('common_workflows.relax.quantum_espresso').get_input_generator()


@pytest.fixture
def default_builder_inputs(generate_code, generate_structure):
    """Return a dictionary with minimum required inputs for the ``get_builder`` method of the inputs generator."""
    return {
        'structure': generate_structure(symbols=('Si',)),
        'engines': {
            'relax': {
                'code': generate_code('quantumespresso.pw').store().uuid,
                'options': {'resources': {'num_machines': 1, 'tot_num_mpiprocs': 1}},
            }
        },
    }


def test_reference_kpoints_are_used_by_all_active_qe_stages():
    """Use the actual final mesh node instead of computing it from a distance."""
    from aiida_common_workflows.workflows.relax.quantum_espresso.generator import (
        QuantumEspressoCommonRelaxInputGenerator,
        get_kpoints_used,
    )

    relax = orm.WorkChainNode().store()
    base = orm.WorkChainNode()
    base.base.links.add_incoming(relax, link_type=LinkType.CALL_WORK, link_label='last_base')
    base.store()
    mesh = orm.KpointsData()
    mesh.set_kpoints_mesh([2, 4, 6])
    mesh.store()
    calc = orm.CalcJobNode()
    calc.base.links.add_incoming(base, link_type=LinkType.CALL_CALC, link_label='last_pw')
    calc.base.links.add_incoming(mesh, link_type=LinkType.INPUT_CALC, link_label='kpoints')
    calc.store()

    used = get_kpoints_used(relax)
    assert used.pk == mesh.pk
    builder = plugins.WorkflowFactory('common_workflows.relax.quantum_espresso').get_builder()
    stages = ['base', 'base_final_scf'] if hasattr(builder, 'base') else ['base_relax', 'base_init_relax']
    for name in stages:
        builder[name].kpoints_distance = orm.Float(0.2)
        builder[name].kpoints_force_parity = orm.Bool(True)

    QuantumEspressoCommonRelaxInputGenerator.apply_reference_kpoints(builder, used)

    for name in stages:
        assert builder[name].kpoints.pk == mesh.pk
        assert 'kpoints_distance' not in builder[name]
        assert 'kpoints_force_parity' not in builder[name]


@pytest.mark.usefixtures('sssp')
def test_prepare_eos_separates_execution_settings(default_builder_inputs):
    from aiida_common_workflows.workflows.eos import prepare_eos

    default_builder_inputs['engines']['relax']['options']['max_wallclock_seconds'] = 1234
    workflow, prepared = prepare_eos(
        backend='common_workflows.relax.quantum_espresso',
        structure=default_builder_inputs['structure'],
        protocol='fast',
        scale_factors=[0.98, 1, 1.02],
        generator_inputs={'engines': default_builder_inputs['engines'], 'relax_type': RelaxType.NONE},
    )
    assert workflow.build() is None
    assert set(prepared.inputs) == {'structure', 'first', 'remaining', 'settings'}
    config = prepared.execution_environment.runtime_config
    assert config['reference'] == config['followup_relaxation']
    stages = prepared.inputs['settings']['settings']
    assert any(isinstance(stage, dict) and 'pw' in stage for stage in stages.values())
    for name, stage in stages.items():
        if isinstance(stage, dict) and 'pw' in stage:
            assert 'code' not in stage['pw']
            assert 'metadata' not in stage['pw']
            assert config['reference'][name]['pw']['code'].uuid == default_builder_inputs['engines']['relax']['code']
            options = config['reference'][name]['pw']['metadata']['options']
            assert options['resources']['num_machines'] == 1
            assert options['max_wallclock_seconds'] == 1234


@calcfunction
def _test_qe_results(structure):
    mesh = orm.KpointsData()
    mesh.set_kpoints_mesh([2, 3, 4])
    return {'kpoints_used': mesh, 'total_energy': orm.Float(structure.get_cell_volume())}


class _QeTestRelaxation(WorkChain):
    """Replace the external QE calculation while exercising the real graph and point wrapper."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=orm.StructureData)
        spec.input('mesh', valid_type=orm.KpointsData, required=False)
        spec.output('kpoints_used', valid_type=orm.KpointsData)
        spec.output('total_energy', valid_type=orm.Float)
        spec.outline(cls.results)

    def results(self):
        outputs = _test_qe_results(self.inputs.structure)
        if 'mesh' in self.inputs:
            outputs['kpoints_used'] = self.inputs.mesh
        self.out_many(outputs)


@pytest.mark.usefixtures('sssp')
@pytest.mark.parametrize('legacy_entry_point', [False, True])
def test_run_prepared_eos(default_builder_inputs, monkeypatch, legacy_entry_point):
    from aiida_common_workflows.workflows.eos import EquationOfStateWorkChain, prepare_eos
    from aiida_common_workflows.workflows.relax.quantum_espresso.eos import QuantumEspressoEosPoint

    original_submit = QuantumEspressoEosPoint.submit
    engines = default_builder_inputs['engines']

    def submit_test_relaxation(self, builder):
        values = {'structure': builder.structure}
        stages = [name for name in ('base', 'base_final_scf', 'base_relax', 'base_init_relax') if builder.get(name)]
        assert stages
        for name in stages:
            assert builder[name].pw.code.uuid == engines['relax']['code']
            assert builder[name].pw.metadata.options.resources['num_machines'] == 1
            if 'reference_mesh' in self.inputs:
                assert builder[name].kpoints.pk == self.inputs.reference_mesh.pk
                assert 'kpoints_distance' not in builder[name]
                values['mesh'] = builder[name].kpoints
        return original_submit(self, _QeTestRelaxation, **values)

    monkeypatch.setattr(QuantumEspressoEosPoint, 'submit', submit_test_relaxation)
    backend = 'common_workflows.relax.quantum_espresso'
    structure = default_builder_inputs['structure']
    factors = [0.98, 1.02, 1.0]
    generator_inputs = {'engines': engines, 'relax_type': 'none'}
    if legacy_entry_point:
        outputs, node = run_get_node(
            EquationOfStateWorkChain,
            structure=structure,
            sub_process_class=backend,
            scale_factors=factors,
            generator_inputs={'protocol': 'fast', **generator_inputs},
        )
        graph_node = next(child for child in node.called if isinstance(child, orm.GraphNode))
        energies = [item.value for item in outputs['total_energies'].values()]
    else:
        workflow, prepared = prepare_eos(
            backend=backend,
            structure=structure,
            scale_factors=factors,
            protocol='fast',
            generator_inputs=generator_inputs,
        )
        outputs, graph_node = run_get_node(workflow, prepared_run=prepared)
        node = graph_node
        energies = [point['total_energy'][0].value for point in [outputs['first'], *outputs['remaining']]]

    assert node.is_finished_ok
    assert graph_node.is_finished_ok
    assert graph_node.base.attributes.get(graph_node.PROTOCOL_KEY)['protocol'].endswith('.fast')
    assert energies == pytest.approx(factors)
    points = [child for child in graph_node.called if child.process_label == QuantumEspressoEosPoint.__name__]
    (reference,) = [child for child in points if 'reference_mesh' not in child.inputs]
    assert len(points) == 3
    assert all(
        child.inputs.reference_mesh.pk == reference.outputs.kpoints_used.pk for child in points if child != reference
    )


@pytest.mark.usefixtures('sssp')
def test_get_builder(generator, default_builder_inputs):
    """Test the ``get_builder`` with default arguments."""
    builder = generator.get_builder(**default_builder_inputs)
    assert isinstance(builder, engine.ProcessBuilder)


@pytest.mark.usefixtures('sssp')
@pytest.mark.skip('Running this test will fail with an `UnroutableError` in `kiwipy`.')
def test_submit(generator, default_builder_inputs):
    """Test submitting the builder returned by ``get_builder`` called with default arguments.

    This will actually create the ``WorkChain`` instance, so if it doesn't raise, that means the input spec was valid.
    """
    builder = generator.get_builder(**default_builder_inputs)
    engine.submit(builder)


@pytest.mark.usefixtures('sssp')
def test_options(generator, default_builder_inputs):
    """Test that the ``options`` of the ``engines`` argument are added to all namespaces."""
    inputs = default_builder_inputs
    builder = generator.get_builder(**inputs)
    assert isinstance(builder, engine.ProcessBuilder)
    assert builder.base.pw.metadata.options.resources == inputs['engines']['relax']['options']['resources']
    assert builder.base_final_scf.pw.metadata.options.resources == inputs['engines']['relax']['options']['resources']


@pytest.mark.usefixtures('sssp')
def test_supported_electronic_types(generator, default_builder_inputs):
    """Test calling ``get_builder`` for the supported ``electronic_types``."""
    inputs = default_builder_inputs

    for electronic_type in generator.spec().inputs['electronic_type'].choices:
        inputs['electronic_type'] = electronic_type
        builder = generator.get_builder(**inputs)
        assert isinstance(builder, engine.ProcessBuilder)


@pytest.mark.usefixtures('sssp')
def test_supported_relax_types(generator, default_builder_inputs):
    """Test calling ``get_builder`` for the supported ``relax_types``."""
    inputs = default_builder_inputs

    for relax_type in generator.spec().inputs['relax_type'].choices:
        inputs['relax_type'] = relax_type
        builder = generator.get_builder(**inputs)
        assert isinstance(builder, engine.ProcessBuilder)


@pytest.mark.usefixtures('sssp')
def test_supported_spin_types(generator, default_builder_inputs):
    """Test calling ``get_builder`` for the supported ``spin_types``."""
    inputs = default_builder_inputs

    for spin_type in generator.spec().inputs['spin_type'].choices:
        inputs['spin_type'] = spin_type
        builder = generator.get_builder(**inputs)
        assert isinstance(builder, engine.ProcessBuilder)


@pytest.mark.usefixtures('sssp')
def test_relax_type(generator, generate_code, generate_structure):
    """Test the ``relax_type`` keyword argument."""
    code = generate_code('quantumespresso.pw')
    structure = generate_structure(symbols=('Si',))
    engines = {'relax': {'code': code, 'options': {}}}

    builder = generator.get_builder(structure=structure, engines=engines, relax_type=RelaxType.NONE)
    assert builder['base']['pw']['parameters']['CONTROL']['calculation'] == 'scf'
    assert 'CELL' not in builder['base']['pw']['parameters'].base.attributes.all

    builder = generator.get_builder(structure=structure, engines=engines, relax_type=RelaxType.POSITIONS)
    assert builder['base']['pw']['parameters']['CONTROL']['calculation'] == 'relax'
    assert 'CELL' not in builder['base']['pw']['parameters'].base.attributes.all

    builder = generator.get_builder(structure=structure, engines=engines, relax_type=RelaxType.CELL)
    assert builder['base']['pw']['parameters']['CONTROL']['calculation'] == 'vc-relax'
    assert builder['base']['pw']['parameters']['CELL']['cell_dofree'] == 'all'

    builder = generator.get_builder(structure=structure, engines=engines, relax_type=RelaxType.SHAPE)
    assert builder['base']['pw']['parameters']['CONTROL']['calculation'] == 'vc-relax'
    assert builder['base']['pw']['parameters']['CELL']['cell_dofree'] == 'shape'

    builder = generator.get_builder(structure=structure, engines=engines, relax_type=RelaxType.POSITIONS_CELL)
    assert builder['base']['pw']['parameters']['CONTROL']['calculation'] == 'vc-relax'
    assert builder['base']['pw']['parameters']['CELL']['cell_dofree'] == 'all'

    builder = generator.get_builder(structure=structure, engines=engines, relax_type=RelaxType.POSITIONS_SHAPE)
    assert builder['base']['pw']['parameters']['CONTROL']['calculation'] == 'vc-relax'
    assert builder['base']['pw']['parameters']['CELL']['cell_dofree'] == 'shape'

    with pytest.raises(ValueError):
        builder = generator.get_builder(structure=structure, engines=engines, relax_type=RelaxType.VOLUME)

    with pytest.raises(ValueError):
        builder = generator.get_builder(structure=structure, engines=engines, relax_type=RelaxType.POSITIONS_VOLUME)


@pytest.mark.usefixtures('sssp')
def test_spin_type(generator, generate_code, generate_structure):
    """Test the ``spin_type`` keyword argument."""
    code = generate_code('quantumespresso.pw')
    structure = generate_structure(symbols=('Si',))
    engines = {'relax': {'code': code, 'options': {}}}

    builder = generator.get_builder(structure=structure, engines=engines, spin_type=SpinType.NONE)
    assert 'nspin' not in builder['base']['pw']['parameters']['SYSTEM']

    builder = generator.get_builder(structure=structure, engines=engines, spin_type=SpinType.COLLINEAR)
    assert builder['base']['pw']['parameters']['SYSTEM']['nspin'] == 2
    assert builder['base']['pw']['parameters']['SYSTEM']['starting_magnetization'] is not None


@pytest.mark.usefixtures('sssp')
def test_electronic_type(generator, generate_code, generate_structure):
    """Test the ``electronic_type`` keyword argument."""
    code = generate_code('quantumespresso.pw')
    structure = generate_structure(symbols=('Si',))
    engines = {'relax': {'code': code, 'options': {}}}

    builder = generator.get_builder(structure=structure, engines=engines, electronic_type=ElectronicType.METAL)
    assert builder['base']['pw']['parameters']['SYSTEM']['degauss'] is not None
    assert builder['base']['pw']['parameters']['SYSTEM']['smearing'] is not None

    builder = generator.get_builder(structure=structure, engines=engines, electronic_type=ElectronicType.INSULATOR)
    assert builder['base']['pw']['parameters']['SYSTEM']['occupations'] == 'fixed'
    assert 'degauss' not in builder['base']['pw']['parameters']['SYSTEM']
    assert 'smearing' not in builder['base']['pw']['parameters']['SYSTEM']


@pytest.mark.usefixtures('sssp')
def test_threshold_forces(generator, generate_code, generate_structure):
    """Test the ``threshold_forces`` keyword argument."""
    from qe_tools import CONSTANTS

    code = generate_code('quantumespresso.pw')
    structure = generate_structure(symbols=('Si',))
    engines = {'relax': {'code': code, 'options': {}}}

    threshold_forces = 0.1
    expected = threshold_forces * CONSTANTS.bohr_to_ang / CONSTANTS.ry_to_ev
    builder = generator.get_builder(structure=structure, engines=engines, threshold_forces=threshold_forces)
    assert builder['base']['pw']['parameters']['CONTROL']['forc_conv_thr'] == expected


@pytest.mark.usefixtures('sssp')
def test_threshold_stress(generator, generate_code, generate_structure):
    """Test the ``threshold_stress`` keyword argument."""
    from qe_tools import CONSTANTS

    code = generate_code('quantumespresso.pw')
    structure = generate_structure(symbols=('Si',))
    engines = {'relax': {'code': code, 'options': {}}}

    threshold_stress = 0.1
    expected = threshold_stress * CONSTANTS.bohr_to_ang**3 / CONSTANTS.ry_to_ev
    builder = generator.get_builder(structure=structure, engines=engines, threshold_stress=threshold_stress)
    assert builder['base']['pw']['parameters']['CELL']['press_conv_thr'] == expected


@pytest.mark.usefixtures('sssp')
def test_magnetization_per_site(generator, generate_code, generate_structure):
    """Test the ``magnetization_per_site`` keyword argument."""
    code = generate_code('quantumespresso.pw')
    structure = generate_structure(symbols=('Si', 'Ge'))
    engines = {'relax': {'code': code, 'options': {}}}

    magnetization_per_site = [0.0, 0.1, 0.2]
    with pytest.raises(ValueError):
        builder = generator.get_builder(
            structure=structure, engines=engines, magnetization_per_site=magnetization_per_site
        )

    magnetization_per_site = [0.0, 0.1]
    builder = generator.get_builder(
        structure=structure,
        engines=engines,
        magnetization_per_site=magnetization_per_site,
        spin_type=SpinType.COLLINEAR,
    )
    assert builder['base']['pw']['parameters']['SYSTEM']['starting_magnetization'] == {'Si': 0.0, 'Ge': 0.025}
