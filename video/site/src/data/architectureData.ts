import type { ArchitectureStep } from '../motion/motionTypes'

export const architectureSteps = [
  'construct',
  'validate',
  'measure',
  'promote',
  'resolve',
] as const satisfies readonly ArchitectureStep[]

export type ArchitectureNodeId =
  | 'official-shape'
  | 'environment-identity'
  | 'program-search'
  | 'program-config'
  | 'schedule-config'
  | 'config-spec'
  | 'static-legality'
  | 'structured-rejection'
  | 'execution-plan'
  | 'execution-runtime'
  | 'gpu-evidence'
  | 'screen'
  | 'enhanced'
  | 'formal'
  | 'promotion'
  | 'keep-incumbent'
  | 'registry'
  | 'runtime-resolution'
  | 'official-output'

export type ArchitectureNodeKind =
  | 'identity'
  | 'search'
  | 'config'
  | 'gate'
  | 'plan'
  | 'runtime'
  | 'measurement'
  | 'fidelity'
  | 'decision'
  | 'registry'
  | 'output'
  | 'exit'

export type ArchitectureNode = {
  id: ArchitectureNodeId
  label: string
  eyebrow?: string
  step: ArchitectureStep
  kind: ArchitectureNodeKind
  x: number
  y: number
  repository: string
  input: string
  output: string
  detail: string
  evidence?: string
  inspectorSide: 'left' | 'right'
}

export type ArchitectureEdgeKind =
  | 'data'
  | 'candidate'
  | 'accepted'
  | 'rejected'
  | 'measurement'
  | 'promotion'
  | 'deployment'
  | 'feedback'
  | 'output'

export type ArchitectureEdge = {
  id: string
  from: ArchitectureNodeId
  to: ArchitectureNodeId
  stage: ArchitectureStep
  kind: ArchitectureEdgeKind
  path: string
  label?: string
}

export type ArchitecturePhase = {
  id: ArchitectureStep
  label: string
  sentence: string
  side: 'left' | 'right'
  index: string
}

export const architecturePhases: ArchitecturePhase[] = [
  {
    id: 'construct',
    index: '01',
    label: 'CONSTRUCT',
    sentence: 'Shape and device define the legal program space.',
    side: 'left',
  },
  {
    id: 'validate',
    index: '02',
    label: 'VALIDATE',
    sentence: 'One ConfigSpec becomes one immutable plan—or a rejection.',
    side: 'right',
  },
  {
    id: 'measure',
    index: '03',
    label: 'MEASURE',
    sentence: 'The planned path is executed and measured in isolation.',
    side: 'left',
  },
  {
    id: 'promote',
    index: '04',
    label: 'PROMOTE',
    sentence: 'Only paired Formal evidence can replace the incumbent.',
    side: 'right',
  },
  {
    id: 'resolve',
    index: '05',
    label: 'RESOLVE',
    sentence: 'Runtime resolves the approved ConfigSpec, then PlanBuilder rebuilds its plan.',
    side: 'left',
  },
]

export const architectureNodes: ArchitectureNode[] = [
  {
    id: 'official-shape',
    label: 'OFFICIAL SHAPE',
    eyebrow: 'INPUT IDENTITY',
    step: 'construct',
    kind: 'identity',
    x: 4,
    y: 38,
    repository: 'official/ · benchmarking/protocols.py',
    input: 'official JSON',
    output: 'TransformerShape',
    detail: 'The supplied workload contract fixes B, S, D, H and F.',
    evidence: 'One shape fingerprint addresses one specialized program.',
    inspectorSide: 'left',
  },
  {
    id: 'environment-identity',
    label: 'ENVIRONMENT FINGERPRINT',
    eyebrow: 'MEASUREMENT IDENTITY',
    step: 'construct',
    kind: 'identity',
    x: 4,
    y: 57,
    repository: 'deployment/environment.py',
    input: 'GPU + driver + CUDA + PyTorch + Triton + math mode',
    output: 'EnvironmentFingerprint.measurement_identity',
    detail: 'Search evidence and deployment must resolve under the same environment identity.',
    evidence: 'RTX 4080 · SM 8.9 · CUDA 13.2',
    inspectorSide: 'left',
  },
  {
    id: 'program-search',
    label: 'PROGRAM SEARCH',
    eyebrow: 'CONDITIONAL SPACE',
    step: 'construct',
    kind: 'search',
    x: 17,
    y: 47.5,
    repository: 'autotune/search_space.py · autotune/search_engine.py',
    input: 'SearchRequest',
    output: 'candidate ConfigSpec',
    detail: 'A shape-conditioned search proposes complete programs inside compatible branches.',
    evidence: 'Latency, failures and coverage update the sampler.',
    inspectorSide: 'left',
  },
  {
    id: 'program-config',
    label: 'PROGRAM',
    eyebrow: 'PROGRAM CONFIG',
    step: 'construct',
    kind: 'config',
    x: 29,
    y: 31,
    repository: 'solution/config.py',
    input: 'attention · projection · layout · precision · FFN · norm',
    output: 'ProgramConfig',
    detail: 'The high-level operator and fusion choices form one half of candidate identity.',
    inspectorSide: 'left',
  },
  {
    id: 'schedule-config',
    label: 'SCHEDULE',
    eyebrow: 'SCHEDULE CONFIG',
    step: 'construct',
    kind: 'config',
    x: 29,
    y: 64,
    repository: 'solution/config.py',
    input: 'runtime · launch params · compile mode · batch tile',
    output: 'ScheduleConfig',
    detail: 'The executable schedule is part of the searched program—not an afterthought.',
    inspectorSide: 'left',
  },
  {
    id: 'config-spec',
    label: 'CONFIG SPEC',
    eyebrow: 'PROGRAM + SCHEDULE',
    step: 'construct',
    kind: 'config',
    x: 39,
    y: 47.5,
    repository: 'solution/config.py',
    input: 'ProgramConfig + ScheduleConfig',
    output: 'stable config_id',
    detail: 'The canonical, serializable identity of one complete candidate program.',
    evidence: 'The same ConfigSpec is validated, measured, promoted and resolved.',
    inspectorSide: 'left',
  },
  {
    id: 'static-legality',
    label: 'STATIC LEGALITY',
    eyebrow: 'PLAN BUILDER',
    step: 'validate',
    kind: 'gate',
    x: 51,
    y: 47.5,
    repository: 'solution/plan_builder.py',
    input: 'ConfigSpec + ExecutionContext',
    output: 'ExecutionPlan or structured rejection',
    detail: 'Checks shape, capability, layout, precision, fusion and runtime before GPU work.',
    evidence: 'Rejected combinations receive no synthetic latency and consume no GPU time.',
    inspectorSide: 'right',
  },
  {
    id: 'structured-rejection',
    label: 'STRUCTURED REJECTION',
    eyebrow: 'NO GPU TIME',
    step: 'validate',
    kind: 'exit',
    x: 51,
    y: 73,
    repository: 'solution/plan_builder.py',
    input: 'illegal candidate + violation',
    output: 'CompileRejection',
    detail: 'An infeasible branch exits before benchmark execution.',
    inspectorSide: 'right',
  },
  {
    id: 'execution-plan',
    label: 'EXECUTION PLAN',
    eyebrow: 'IMMUTABLE PLAN',
    step: 'validate',
    kind: 'plan',
    x: 62,
    y: 47.5,
    repository: 'solution/plan.py',
    input: 'one accepted ConfigSpec',
    output: 'ExecutionPlan + ExpectedExecutionTrace',
    detail: 'A single immutable plan selects the exact runtime building blocks and expected path.',
    evidence: 'No second policy lookup occurs inside model execution.',
    inspectorSide: 'right',
  },
  {
    id: 'execution-runtime',
    label: 'TRANSFORMER RUNTIME',
    eyebrow: 'PLAN-DIRECTED EXECUTION',
    step: 'measure',
    kind: 'runtime',
    x: 72,
    y: 38,
    repository: 'solution/transformer.py · solution/operators/ · solution/kernels/ · solution/runtimes/',
    input: 'ExecutionPlan + input tensor',
    output: 'official-compatible output + observed trace',
    detail: 'Operators, Triton kernels and runtime wrappers execute exactly as the plan specifies.',
    evidence: 'EXPECTED TRACE = OBSERVED TRACE · PATH MATCH',
    inspectorSide: 'right',
  },
  {
    id: 'gpu-evidence',
    label: 'GPU EVIDENCE',
    eyebrow: 'ISOLATED MEASUREMENT',
    step: 'measure',
    kind: 'measurement',
    x: 72,
    y: 63,
    repository: 'benchmarking/',
    input: 'model + input + MeasurementProtocol',
    output: 'BenchmarkResult + observed trace',
    detail: 'Each Shape search or suite runs in a fresh process under a serialized single-GPU lease.',
    evidence: 'ACCURACY · PATH · MEDIAN + P90 · PEAK MEMORY',
    inspectorSide: 'left',
  },
  {
    id: 'screen',
    label: 'SCREEN',
    eyebrow: 'FIDELITY 01',
    step: 'measure',
    kind: 'fidelity',
    x: 82,
    y: 71,
    repository: 'autotune/evaluation.py · autotune/search_engine.py',
    input: 'candidate observation',
    output: 'feasible screen evidence',
    detail: 'Cheap evidence rejects weak or infeasible candidates early.',
    inspectorSide: 'left',
  },
  {
    id: 'enhanced',
    label: 'ENHANCED',
    eyebrow: 'FIDELITY 02',
    step: 'measure',
    kind: 'fidelity',
    x: 89,
    y: 71,
    repository: 'autotune/evaluation.py · autotune/search_engine.py',
    input: 'screen survivor',
    output: 'stronger constrained observation',
    detail: 'More measurement effort verifies a promising branch before Formal comparison.',
    inspectorSide: 'right',
  },
  {
    id: 'formal',
    label: 'FORMAL',
    eyebrow: 'FIDELITY 03',
    step: 'measure',
    kind: 'fidelity',
    x: 95,
    y: 64,
    repository: 'benchmarking/measurement_core.py',
    input: 'locked challenger + incumbent',
    output: 'alternating AB/BA paired ratios',
    detail: 'Challenger and incumbent alternate through the same measurement gate.',
    evidence: 'Only paired evidence may replace an incumbent.',
    inspectorSide: 'right',
  },
  {
    id: 'promotion',
    label: 'PROMOTE OR REJECT',
    eyebrow: 'PAIRED DECISION',
    step: 'promote',
    kind: 'decision',
    x: 95,
    y: 47.5,
    repository: 'autotune/promotion.py',
    input: 'growing tuple of paired ratios',
    output: 'continue · promote · reject',
    detail: 'A feasible Formal result may initialize deployment; paired evidence is required to replace an incumbent.',
    evidence: 'ONLY PAIRED FORMAL EVIDENCE CAN REPLACE AN INCUMBENT',
    inspectorSide: 'right',
  },
  {
    id: 'keep-incumbent',
    label: 'KEEP INCUMBENT',
    eyebrow: 'REJECT',
    step: 'promote',
    kind: 'exit',
    x: 88,
    y: 52,
    repository: 'autotune/promotion.py',
    input: 'terminal reject decision',
    output: 'registry unchanged',
    detail: 'Rejected challengers fade from the deployment path.',
    inspectorSide: 'right',
  },
  {
    id: 'registry',
    label: 'MEASURED STACK × SHAPE VARIANT',
    eyebrow: 'DEPLOYMENT REGISTRY',
    step: 'promote',
    kind: 'registry',
    x: 87,
    y: 23,
    repository: 'deployment/registry.py · deployment/deployed_configs.json',
    input: 'EnvironmentFingerprint.measurement_identity + ShapeFingerprint + approved ConfigSpec',
    output: 'matching current winner',
    detail: 'Formal approval replaces one addressable cell—not a generic global configuration.',
    evidence: 'Only a promoted full ConfigSpec enters deployed_configs.json.',
    inspectorSide: 'right',
  },
  {
    id: 'runtime-resolution',
    label: 'RESOLVE CONFIG → REBUILD PLAN',
    eyebrow: 'EXACT MATCH',
    step: 'resolve',
    kind: 'registry',
    x: 72,
    y: 19,
    repository: 'deployment/registry.py · solution/transformer.py',
    input: 'current environment + shape',
    output: 'approved ConfigSpec or portable fallback',
    detail: 'Ordinary model construction resolves the approved ConfigSpec, then PlanBuilder rebuilds an ExecutionPlan.',
    evidence: 'RESOLVE CONFIG → REBUILD PLAN → EXECUTE',
    inspectorSide: 'left',
  },
  {
    id: 'official-output',
    label: 'OFFICIAL-COMPATIBLE OUTPUT',
    eyebrow: 'OUTPUT PORT',
    step: 'resolve',
    kind: 'output',
    x: 90,
    y: 36,
    repository: 'solution/transformer.py',
    input: 'plan-directed runtime result',
    output: 'official-compatible tensor',
    detail: 'The deployed program returns through the same interface as the supplied workload.',
    inspectorSide: 'right',
  },
]

export const architectureEdges: ArchitectureEdge[] = [
  { id: 'shape-search', from: 'official-shape', to: 'program-search', stage: 'construct', kind: 'data', path: 'M 130 385 C 220 385 235 450 325 466' },
  { id: 'environment-search', from: 'environment-identity', to: 'program-search', stage: 'construct', kind: 'data', path: 'M 130 560 C 220 560 235 500 325 479' },
  { id: 'search-program', from: 'program-search', to: 'program-config', stage: 'construct', kind: 'candidate', path: 'M 405 465 C 475 430 480 335 545 310' },
  { id: 'search-schedule', from: 'program-search', to: 'schedule-config', stage: 'construct', kind: 'candidate', path: 'M 405 485 C 475 515 480 620 545 635' },
  { id: 'program-config-spec', from: 'program-config', to: 'config-spec', stage: 'construct', kind: 'candidate', path: 'M 625 310 C 680 325 690 410 740 450' },
  { id: 'schedule-config-spec', from: 'schedule-config', to: 'config-spec', stage: 'construct', kind: 'candidate', path: 'M 625 635 C 680 615 690 535 740 495' },
  { id: 'config-legality', from: 'config-spec', to: 'static-legality', stage: 'validate', kind: 'candidate', path: 'M 820 475 L 950 475' },
  { id: 'legality-plan', from: 'static-legality', to: 'execution-plan', stage: 'validate', kind: 'accepted', path: 'M 1030 475 L 1165 475', label: 'ACCEPTED' },
  { id: 'legality-rejection', from: 'static-legality', to: 'structured-rejection', stage: 'validate', kind: 'rejected', path: 'M 990 510 L 990 695', label: 'NO GPU TIME' },
  { id: 'plan-runtime', from: 'execution-plan', to: 'execution-runtime', stage: 'measure', kind: 'measurement', path: 'M 1240 455 C 1295 430 1320 390 1370 375', label: 'EXPECTED TRACE' },
  { id: 'runtime-evidence', from: 'execution-runtime', to: 'gpu-evidence', stage: 'measure', kind: 'measurement', path: 'M 1415 410 L 1415 595', label: 'OBSERVED TRACE' },
  { id: 'evidence-screen', from: 'gpu-evidence', to: 'screen', stage: 'measure', kind: 'measurement', path: 'M 1480 635 C 1520 650 1530 690 1570 700' },
  { id: 'screen-enhanced', from: 'screen', to: 'enhanced', stage: 'measure', kind: 'measurement', path: 'M 1620 700 L 1690 700' },
  { id: 'enhanced-formal', from: 'enhanced', to: 'formal', stage: 'measure', kind: 'measurement', path: 'M 1740 695 C 1780 675 1790 635 1810 615' },
  { id: 'formal-promotion', from: 'formal', to: 'promotion', stage: 'promote', kind: 'promotion', path: 'M 1825 595 L 1825 505' },
  { id: 'promotion-incumbent', from: 'promotion', to: 'keep-incumbent', stage: 'promote', kind: 'rejected', path: 'M 1805 490 C 1765 495 1740 510 1700 515', label: 'REJECT' },
  { id: 'promotion-registry', from: 'promotion', to: 'registry', stage: 'promote', kind: 'promotion', path: 'M 1815 455 C 1785 390 1730 295 1685 250', label: 'PROMOTE' },
  { id: 'registry-resolution', from: 'registry', to: 'runtime-resolution', stage: 'resolve', kind: 'deployment', path: 'M 1645 220 C 1575 180 1475 170 1410 195' },
  { id: 'resolution-plan', from: 'runtime-resolution', to: 'execution-plan', stage: 'resolve', kind: 'deployment', path: 'M 1380 230 C 1315 280 1265 345 1225 430', label: 'SAME EXECUTION PATH' },
  { id: 'runtime-output', from: 'execution-runtime', to: 'official-output', stage: 'resolve', kind: 'output', path: 'M 1465 365 C 1545 350 1645 350 1725 355' },
  { id: 'measurement-feedback', from: 'gpu-evidence', to: 'program-search', stage: 'measure', kind: 'feedback', path: 'M 1385 675 C 1220 860 560 875 350 520', label: 'LATENCY · FAILURES · COVERAGE' },
]

export const runtimeLanes = [
  { id: 'operators', label: 'OPERATORS', repository: 'solution/operators/' },
  { id: 'kernels', label: 'TRITON KERNELS', repository: 'solution/kernels/' },
  { id: 'wrappers', label: 'RUNTIME WRAPPERS', repository: 'solution/runtimes/' },
] as const

export const evidenceChannels = [
  'ACCURACY',
  'EXECUTION PATH',
  'MEDIAN + P90 LATENCY',
  'PEAK MEMORY',
] as const

export const promotionThresholds = [
  '6/6 ≥ 1.10×',
  '8/9 ≥ 1.05×',
  '11/13 ≥ 1.02×',
] as const

export const evidenceStores = [
  { label: 'OPTUNA SQLITE', detail: 'Screen + reusable Enhanced evidence' },
  { label: 'JSONL RUN LOG', detail: 'coverage · failures · stage duration · paired ratios' },
  { label: 'DEPLOYED CONFIGS JSON', detail: 'current formally approved winner' },
] as const

export const workflowCommands = [
  { label: 'PROBE', scope: 'ENVIRONMENT' },
  { label: 'BENCHMARK', scope: 'ISOLATED MEASUREMENT' },
  { label: 'PROFILE', scope: 'OPERATION EVIDENCE' },
  { label: 'SEARCH', scope: 'ONE BOUNDED SWEEP' },
  { label: 'OPTIMIZE', scope: 'REPEAT TO STOP RULE' },
] as const
