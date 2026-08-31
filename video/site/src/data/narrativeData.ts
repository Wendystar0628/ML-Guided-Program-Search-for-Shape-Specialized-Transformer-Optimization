export const narrativeAnchors = [
  { id: 'outcome', navLabel: 'Study Overview' },
  { id: 'workloads', navLabel: 'Problem Setting' },
  { id: 'architecture', navLabel: 'System Architecture' },
  { id: 'search', navLabel: 'Search Method' },
  { id: 'evidence', navLabel: 'Evaluation Results' },
] as const

export const relayLabels = {
  outcomeWorkloads: 'The official workload set spans distinct execution regimes.',
  workloadsArchitecture: 'Workload and device identity define the search and deployment context.',
  architectureSearch: 'Legal ExecutionPlans enter staged GPU measurement.',
  searchEvidence: 'Paired Formal evidence controls exact-device registry updates.',
  evidenceClosing: 'Thirteen correct ratios → one equal-shape aggregate',
} as const
