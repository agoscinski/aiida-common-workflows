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

## Graph-language problems

| Limitation | Impact and workaround |
| --- | --- |
| **Placements have a static process contract.** | The shared compatibility path still constructs a backend-specific builder for each point. Prepared QE EOS instead selects a concrete graph and validates its relaxation template before launch; no point resolves the backend or protocol. |
| **Ports and spreads require statically known fields.** | The existing `sub_process` namespace accepts arbitrary backend overrides, including nested metadata and AiiDA data. The shared compatibility graph accepts a packed request; a calcfunction links the original input nodes. Prepared QE EOS applies and validates overrides before graph launch. |
| **A graph call wires data, not the child WorkChain node.** | Quantum ESPRESSO now exposes `kpoints_used` on the common wrapper and wires that data directly. The shared mesh takes precedence over per-point mesh or density settings on follow-up runs. Other reference generators still inspect their reference WorkChain, so they stay on the compatibility path until they expose what they actually reuse. |
| **`par_map` collects one leaf output per item in an owned container.** | A point has heterogeneous and optional fields. The adapter packs its fields into a homogeneous `DataDict` of singleton `DataList` values. The public WorkChain publishes standalone copies made by a calcfunction, because owned members cannot be returned directly. |
| **`par_map(partial(...))` counts optional ports when checking for one unbound input.** | The independent scan maps a small nested graph that omits the optional reference mesh port. |

Scripts that look for direct relaxation children must traverse descendants instead.
Existing in-flight WorkChains should be completed before upgrading because their outlines have changed.

## Verification

Focused tests exercise all three scans with a local test backend, including same-node mesh reuse, reference failure, ordering, missing optional magnetization and the legacy fallback.
A Quantum ESPRESSO test verifies mesh extraction from recorded calculation inputs and mesh injection into a real QE process builder.
External simulation executables and daemon restart were not tested.
