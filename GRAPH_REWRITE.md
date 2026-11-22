# Graph rewrite report

The equation-of-state and dissociation workflows use the reference-first graph in [`_scan.py`](src/aiida_common_workflows/workflows/_scan.py) for Quantum ESPRESSO.
It waits for the reference relaxation to succeed, then uses `par_map` to run subsequent points with the exact `KpointsData` used by the reference's final calculation.
The energy–magnetization workflow uses a graph to map independent points.
Other reference backends retain WorkChain orchestration and receive the actual reference process in the legacy generator API.
Neither scan path passes process UUIDs for references or results.
The public WorkChains still validate inputs and publish the same ordered, indexed output namespaces, but the result nodes are provenance-tracked copies.
No graph-language or Quantum ESPRESSO plugin changes are required.

## Graph-language problems

| Limitation | Impact and workaround |
| --- | --- |
| **Placements have a static process contract.** | Scan entry points select a relaxation plugin and construct a backend-specific builder for each point. Graph placements use a static WorkChain adapter for the dynamic submission. `build()` checks the adapter's ports and graph wiring, not the backend builder. |
| **Ports and spreads require statically known fields.** | The existing `sub_process` namespace accepts arbitrary backend overrides, including nested metadata and AiiDA data. The graph accepts a packed request instead; a calcfunction links the original input nodes. |
| **A graph call wires data, not the child WorkChain node.** | Quantum ESPRESSO now exposes `kpoints_used` on the common wrapper and wires that data directly. The shared mesh takes precedence over per-point mesh or density settings on follow-up runs. Other reference generators still inspect their reference WorkChain, so they stay on the compatibility path until they expose what they actually reuse. |
| **`par_map` collects one leaf output per item in an owned container.** | A point has heterogeneous and optional fields. The adapter packs its fields into a homogeneous `DataDict` of singleton `DataList` values. The public WorkChain publishes standalone copies made by a calcfunction, because owned members cannot be returned directly. |
| **`par_map(partial(...))` counts optional ports when checking for one unbound input.** | The independent scan maps a small nested graph that omits the optional reference mesh port. |

Scripts that look for direct relaxation children must traverse descendants instead.
Existing in-flight WorkChains should be completed before upgrading because their outlines have changed.

## Verification

Focused tests exercise all three scans with a local test backend, including same-node mesh reuse, reference failure, ordering, missing optional magnetization and the legacy fallback.
A Quantum ESPRESSO test verifies mesh extraction from recorded calculation inputs and mesh injection into a real QE process builder.
External simulation executables and daemon restart were not tested.
