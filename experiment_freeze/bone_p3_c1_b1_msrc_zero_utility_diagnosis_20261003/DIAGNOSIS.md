# MSRC Seed11 Zero-Utility Diagnosis

This freeze records a CPU-only, read-only diagnosis of the already-created
MSRC development carrier. No model training, CUDA call, full-ground-truth
load, or clustering metric was performed.

The stored `U_cycle` exactly matches the frozen formula recomputed from
`p_gen` and `p_ver` at `rtol=1e-6, atol=1e-7` (observed maximum absolute
difference: 0). Sample IDs are unique and the six generator/verifier action
pairs are complementary partitions of the four views.

Four of 210 samples have exactly zero total utility mass: raw row IDs 15, 47,
114, and 175. Of the 14 frozen sparse labeled anchors, only raw row 175,
class 5, has zero mass. All six actions at row 175 have closure zero. Its
paired class-5 anchor, raw row 153, has positive mass, so the A1 class-level
feasibility condition is satisfied for this carrier.

`ADAPTER_BUG_EVIDENCE=false` means only that the checks recorded here found no
evidence of an adapter error. It is not a claim that no adapter bug could
exist. The original B1 fail-closed stop remains a valid historical result.
