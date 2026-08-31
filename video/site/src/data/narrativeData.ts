export const narrativeAnchors = [
  { id: 'outcome', navLabel: 'OUTCOME' },
  { id: 'workloads', navLabel: 'WORKLOADS' },
  { id: 'architecture', navLabel: 'ARCHITECTURE' },
  { id: 'search', navLabel: 'SEARCH' },
  { id: 'evidence', navLabel: 'EVIDENCE' },
] as const

export const relayLabels = {
  outcomeWorkloads: 'WHY DOES EACH SHAPE NEED A DIFFERENT PATH?',
  workloadsArchitecture: 'SHAPE FINGERPRINT + ENVIRONMENT FINGERPRINT',
  architectureSearch: 'EXECUTION PLAN → CANDIDATE PROGRAMS',
  searchEvidence: 'APPROVED WINNER → DEPLOYED MEASUREMENT',
  evidenceClosing: 'THIRTEEN CORRECT RATIOS → ONE EQUAL-SHAPE AGGREGATE',
} as const
