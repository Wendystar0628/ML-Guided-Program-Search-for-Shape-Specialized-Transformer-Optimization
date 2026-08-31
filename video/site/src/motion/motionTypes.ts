export type NarrativeSection =
  | 'outcome'
  | 'workloads'
  | 'architecture'
  | 'search'
  | 'evidence'

export type OutcomeStep = 'reveal' | 'winner'
export type WorkloadStep = 'map' | 'specialize'
export type ArchitectureStep = 'construct' | 'validate' | 'measure' | 'promote' | 'resolve'
export type SearchStep = 'sample' | 'reject' | 'screen' | 'enhanced' | 'formal' | 'registry'
export type EvidenceStep = 'compare' | 'aggregate' | 'close'

export type NarrativeStep =
  | OutcomeStep
  | WorkloadStep
  | ArchitectureStep
  | SearchStep
  | EvidenceStep

export type MotionState = 'before' | 'active' | 'settled'

export type InspectState = {
  hoveredId: string | null
  focusedId: string | null
}

export type ArchitectureInspectState = InspectState & {
  pinnedId: string | null
}

export type EvidenceTraceVariant =
  | 'outcome-workloads'
  | 'workloads-architecture'
  | 'architecture-search'
  | 'search-evidence'
  | 'evidence-closing'
