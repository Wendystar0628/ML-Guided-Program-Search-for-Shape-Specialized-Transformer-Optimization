import type { ArchitectureStep } from '../motion/motionTypes'

export const architectureSteps = [
  'construct',
  'validate',
  'measure',
  'promote',
  'resolve',
] as const satisfies readonly ArchitectureStep[]

export type ArchitectureLaneId = 'synthesis' | 'evidence' | 'deployment'

export type ArchitectureNodeId =
  | 'official-context'
  | 'conditional-search'
  | 'config-spec'
  | 'plan-builder'
  | 'execution-plan'
  | 'screen'
  | 'enhanced'
  | 'formal-paired'
  | 'sequential-promotion'
  | 'exact-device-registry'
  | 'resident-runtime'
  | 'streamed-runtime'
  | 'official-output'

export type ArchitectureNodeKind =
  | 'identity'
  | 'search'
  | 'config'
  | 'gate'
  | 'plan'
  | 'fidelity'
  | 'decision'
  | 'registry'
  | 'runtime'
  | 'output'

export type ArchitectureNode = {
  id: ArchitectureNodeId
  lane: ArchitectureLaneId
  label: string
  eyebrow: string
  note: string
  step: ArchitectureStep
  kind: ArchitectureNodeKind
  repository: string
  input: string
  output: string
  detail: string
  evidence?: string
}

export type ArchitectureEdgeKind =
  | 'data'
  | 'candidate'
  | 'accepted'
  | 'measurement'
  | 'promotion'
  | 'deployment'

export type ArchitectureEdge = {
  id: string
  lane: ArchitectureLaneId
  from: ArchitectureNodeId
  to: ArchitectureNodeId
  stage: ArchitectureStep
  kind: ArchitectureEdgeKind
}

export type ArchitectureLane = {
  id: ArchitectureLaneId
  index: string
  title: string
  summary: string
  nodeIds: ArchitectureNodeId[]
}

export type ArchitectureConnector = {
  afterLane: ArchitectureLaneId
  primary: string
  feedback?: string
}

export const architectureNodes: ArchitectureNode[] = [
  {
    id: 'official-context',
    lane: 'synthesis',
    label: 'Workload and device identity',
    eyebrow: 'Inputs',
    note: 'Official shape · GPU · software stack',
    step: 'construct',
    kind: 'identity',
    repository: 'official/ · deployment/environment.py',
    input: 'official shape + measured device stack',
    output: 'bounded search identity',
    detail: 'The official shape and measured device stack define the search and deployment identity.',
  },
  {
    id: 'conditional-search',
    lane: 'synthesis',
    label: 'Conditional program search',
    eyebrow: 'Search module',
    note: 'Resident TPE · separate bounded streamed branch',
    step: 'construct',
    kind: 'search',
    repository: 'autotune/search_space.py · autotune/search_engine.py',
    input: 'shape-conditioned program branches',
    output: 'complete candidate program',
    detail: 'Conditional TPE searches resident programs; a separate bounded branch handles streamed execution.',
    evidence: 'Screen evidence updates the resident sampler.',
  },
  {
    id: 'config-spec',
    lane: 'synthesis',
    label: 'ConfigSpec',
    eyebrow: 'Candidate representation',
    note: 'ProgramConfig + ScheduleConfig',
    step: 'construct',
    kind: 'config',
    repository: 'solution/config.py',
    input: 'program and schedule choices',
    output: 'stable config_id',
    detail: 'ConfigSpec serializes the complete program and schedule candidate used by every downstream stage.',
  },
  {
    id: 'plan-builder',
    lane: 'synthesis',
    label: 'PlanBuilder and static legality',
    eyebrow: 'Validation module',
    note: 'ExecutionPlan or structured rejection',
    step: 'validate',
    kind: 'gate',
    repository: 'solution/plan_builder.py',
    input: 'ConfigSpec + execution context',
    output: 'ExecutionPlan or structured rejection',
    detail: 'PlanBuilder checks shape, capability and configuration constraints before GPU measurement.',
  },
  {
    id: 'execution-plan',
    lane: 'synthesis',
    label: 'ExecutionPlan',
    eyebrow: 'Immutable plan representation',
    note: 'Runtime selections + expected trace',
    step: 'validate',
    kind: 'plan',
    repository: 'solution/plan.py',
    input: 'legal ConfigSpec',
    output: 'plan + expected execution trace',
    detail: 'ExecutionPlan fixes the runtime components and records the expected execution trace.',
  },
  {
    id: 'screen',
    lane: 'evidence',
    label: 'Screen',
    eyebrow: 'Measurement stage 1',
    note: 'Feasibility · execution path · latency',
    step: 'measure',
    kind: 'fidelity',
    repository: 'autotune/evaluation.py',
    input: 'legal ExecutionPlan',
    output: 'screen observation',
    detail: 'Screen records low-cost feasibility and performance observations for candidate filtering and sampler updates.',
  },
  {
    id: 'enhanced',
    lane: 'evidence',
    label: 'Enhanced',
    eyebrow: 'Measurement stage 2',
    note: 'Higher-repeat candidate measurement',
    step: 'measure',
    kind: 'fidelity',
    repository: 'autotune/evaluation.py',
    input: 'Screen survivor',
    output: 'enhanced observation',
    detail: 'Enhanced increases measurement effort for candidates that pass Screen.',
  },
  {
    id: 'formal-paired',
    lane: 'evidence',
    label: 'Formal paired measurement',
    eyebrow: 'Measurement stage 3',
    note: 'Alternating challenger / incumbent · AB / BA',
    step: 'measure',
    kind: 'fidelity',
    repository: 'benchmarking/measurement_core.py',
    input: 'locked challenger + incumbent',
    output: 'paired AB / BA ratios',
    detail: 'Formal measurement alternates challenger and incumbent to produce paired latency ratios.',
  },
  {
    id: 'sequential-promotion',
    lane: 'deployment',
    label: 'Sequential promotion test',
    eyebrow: 'Statistical decision',
    note: 'Pre-specified paired thresholds',
    step: 'promote',
    kind: 'decision',
    repository: 'autotune/promotion.py',
    input: 'paired Formal ratios',
    output: 'promote or keep incumbent',
    detail: 'A challenger is promoted only when its paired ratios satisfy the pre-specified 6-, 9-, or 13-block rule.',
  },
  {
    id: 'exact-device-registry',
    lane: 'deployment',
    label: 'Exact-device deployment registry',
    eyebrow: 'Registry key',
    note: 'Environment identity × shape variant',
    step: 'promote',
    kind: 'registry',
    repository: 'deployment/registry.py · deployed_configs.json',
    input: 'approved ConfigSpec + exact device and shape',
    output: 'current deployment winner',
    detail: 'The registry stores the approved ConfigSpec for one environment identity and shape variant.',
  },
  {
    id: 'resident-runtime',
    lane: 'deployment',
    label: 'Resident runtime',
    eyebrow: 'Resident execution',
    note: 'Shapes 01–13 · resolved ExecutionPlan',
    step: 'resolve',
    kind: 'runtime',
    repository: 'solution/transformer.py',
    input: 'resident workload + approved ConfigSpec',
    output: 'plan-directed execution',
    detail: 'Resident workloads resolve their approved ConfigSpec and rebuild the corresponding ExecutionPlan.',
  },
  {
    id: 'streamed-runtime',
    lane: 'deployment',
    label: 'Streamed runtime',
    eyebrow: 'Streamed execution',
    note: 'Shape 14 · finite streamed execution',
    step: 'resolve',
    kind: 'runtime',
    repository: 'solution/transformer.py',
    input: 'streamed workload + bounded configuration',
    output: 'finite streamed execution',
    detail: 'Shape 14 executes through the finite streamed runtime branch.',
  },
  {
    id: 'official-output',
    lane: 'deployment',
    label: 'Official-compatible output',
    eyebrow: 'Output',
    note: 'Official tensor contract',
    step: 'resolve',
    kind: 'output',
    repository: 'solution/transformer.py',
    input: 'resident or streamed runtime result',
    output: 'official-compatible tensor',
    detail: 'Both runtime branches return a tensor compatible with the official workload interface.',
  },
]

export const architectureEdges: ArchitectureEdge[] = [
  { id: 'context-search', lane: 'synthesis', from: 'official-context', to: 'conditional-search', stage: 'construct', kind: 'data' },
  { id: 'search-config', lane: 'synthesis', from: 'conditional-search', to: 'config-spec', stage: 'construct', kind: 'candidate' },
  { id: 'config-builder', lane: 'synthesis', from: 'config-spec', to: 'plan-builder', stage: 'validate', kind: 'candidate' },
  { id: 'builder-plan', lane: 'synthesis', from: 'plan-builder', to: 'execution-plan', stage: 'validate', kind: 'accepted' },
  { id: 'screen-enhanced', lane: 'evidence', from: 'screen', to: 'enhanced', stage: 'measure', kind: 'measurement' },
  { id: 'enhanced-formal', lane: 'evidence', from: 'enhanced', to: 'formal-paired', stage: 'measure', kind: 'measurement' },
  { id: 'promotion-registry', lane: 'deployment', from: 'sequential-promotion', to: 'exact-device-registry', stage: 'promote', kind: 'promotion' },
  { id: 'registry-resident', lane: 'deployment', from: 'exact-device-registry', to: 'resident-runtime', stage: 'resolve', kind: 'deployment' },
  { id: 'registry-streamed', lane: 'deployment', from: 'exact-device-registry', to: 'streamed-runtime', stage: 'resolve', kind: 'deployment' },
  { id: 'resident-output', lane: 'deployment', from: 'resident-runtime', to: 'official-output', stage: 'resolve', kind: 'deployment' },
  { id: 'streamed-output', lane: 'deployment', from: 'streamed-runtime', to: 'official-output', stage: 'resolve', kind: 'deployment' },
]

export const architectureLanes: ArchitectureLane[] = [
  {
    id: 'synthesis',
    index: '01',
    title: 'Program Synthesis and Static Compilation',
    summary: 'Inputs: official shape and device fingerprint. Output: a statically legal ExecutionPlan.',
    nodeIds: ['official-context', 'conditional-search', 'config-spec', 'plan-builder', 'execution-plan'],
  },
  {
    id: 'evidence',
    index: '02',
    title: 'Isolated Multi-fidelity Measurement',
    summary: 'Input: a legal ExecutionPlan. Output: Screen, Enhanced and paired Formal evidence.',
    nodeIds: ['screen', 'enhanced', 'formal-paired'],
  },
  {
    id: 'deployment',
    index: '03',
    title: 'Promotion, Deployment and Runtime Execution',
    summary: 'Input: paired Formal ratios. Output: a registry entry and official-compatible tensor.',
    nodeIds: ['sequential-promotion', 'exact-device-registry', 'resident-runtime', 'streamed-runtime', 'official-output'],
  },
]

export const architectureConnectors: ArchitectureConnector[] = [
  {
    afterLane: 'synthesis',
    primary: 'Legal ExecutionPlan enters isolated measurement',
    feedback: 'Screen evidence updates the conditional sampler',
  },
  {
    afterLane: 'evidence',
    primary: 'Paired Formal evidence enters sequential promotion',
  },
]
