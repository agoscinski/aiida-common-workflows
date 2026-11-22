# Graph rewrite report

Quantum ESPRESSO equation of state uses the [QE-specific graph](src/aiida_common_workflows/workflows/relax/quantum_espresso/eos.py).
Dissociation still uses the shared reference-first graph in [`_scan.py`](src/aiida_common_workflows/workflows/_scan.py).
It waits for the reference relaxation to succeed, then uses `par_map` to run subsequent points with the exact `KpointsData` used by the reference's final calculation.
The energy–magnetization workflow uses a graph to map independent points.
Other reference backends retain WorkChain orchestration and receive the actual reference process in the legacy generator API.
Neither scan path passes process UUIDs for references or results.
The public WorkChains still validate inputs and publish the same ordered, indexed output namespaces, but the result nodes are provenance-tracked copies.
No graph-language or Quantum ESPRESSO plugin changes are required.

## Launching a prepared QE equation of state

```python
from aiida.engine import submit
from aiida_common_workflows.workflows.eos import prepare_eos

workflow, prepared = prepare_eos(
    backend='common_workflows.relax.quantum_espresso',
    structure=structure,
    scale_factors=[0.98, 1.0, 1.02],
    protocol='fast',
    generator_inputs={
        'relax_type': 'positions',
        'engines': {'relax': {'code': code, 'options': {'resources': {'num_machines': 1}}}},
    },
)
node = submit(workflow, prepared_run=prepared)
```

The launcher resolves the QE protocol before submission, previews the graph, and passes code and scheduler options in an `ExecutionEnvironment`.
The graph sees typed QE scientific ports and the reference mesh, not a backend name or serialized builder request.
The QE point WorkChain still delegates to the existing QE relaxation.
The existing EOS WorkChain delegates to the same prepared graph for standard QE calls and continues to publish the legacy indexed outputs.
Other backends and other kinds of scans were not ported to the new launch API.

## Graph-language blockers

None identified for the migrated scans.
The remaining reference backends still need to expose the data their follow-up calculations reuse; that is plugin migration work, not a graph-language blocker.

Scripts that look for direct relaxation children must traverse descendants instead.
Existing in-flight WorkChains should be completed before upgrading because their outlines have changed.

## Verification

Focused tests exercise all three scans with a local test backend, including same-node mesh reuse, reference failure, ordering, missing optional magnetization and the legacy fallback.
A Quantum ESPRESSO test verifies mesh extraction from recorded calculation inputs and mesh injection into a real QE process builder.
External simulation executables and daemon restart were not tested.
