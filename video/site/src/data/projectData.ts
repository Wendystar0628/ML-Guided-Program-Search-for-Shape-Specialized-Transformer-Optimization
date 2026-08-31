export type Workload = {
  id: string
  batch: number
  sequence: number
  width: number
  heads: number
  ffn: number
  totalTokens: number
  attentionElements: number
}

export type PerformancePoint = {
  id: string
  baselineMs: number
  deployedMs: number
  speedup: number
}

export const headline = {
  kicker: 'TIKTOK TECHJAM 2026 · HARDWARE EFFICIENCY',
  title: ['LEARNING-GUIDED', 'PROGRAM SEARCH'],
  subject: 'SHAPE-SPECIALIZED TRANSFORMER EXECUTION ON AN RTX 4080',
  resultClaim: 'MEASURED PROGRAMS — NOT HAND-PICKED POLICY LABELS',
  geomean: 14.4926,
  residentPassed: 13,
  residentTotal: 13,
} as const

export const workloadRanges = {
  batch: '1 — 10,000',
  sequence: '32 — 100,000',
  width: '32 — 1,024',
  heads: '1 — 16',
} as const

export const workloads: Workload[] = [
  { id: '01', batch: 64, sequence: 128, width: 128, heads: 4, ffn: 128, totalTokens: 8192, attentionElements: 16777216 },
  { id: '02', batch: 1, sequence: 128, width: 128, heads: 4, ffn: 128, totalTokens: 128, attentionElements: 262144 },
  { id: '03', batch: 4, sequence: 128, width: 128, heads: 4, ffn: 128, totalTokens: 512, attentionElements: 1048576 },
  { id: '04', batch: 16, sequence: 128, width: 128, heads: 4, ffn: 128, totalTokens: 2048, attentionElements: 4194304 },
  { id: '05', batch: 128, sequence: 128, width: 128, heads: 4, ffn: 128, totalTokens: 16384, attentionElements: 33554432 },
  { id: '06', batch: 10000, sequence: 128, width: 128, heads: 4, ffn: 128, totalTokens: 1280000, attentionElements: 2621440000 },
  { id: '07', batch: 64, sequence: 128, width: 32, heads: 4, ffn: 32, totalTokens: 8192, attentionElements: 16777216 },
  { id: '08', batch: 64, sequence: 128, width: 1024, heads: 4, ffn: 1024, totalTokens: 8192, attentionElements: 16777216 },
  { id: '09', batch: 64, sequence: 128, width: 128, heads: 1, ffn: 128, totalTokens: 8192, attentionElements: 4194304 },
  { id: '10', batch: 64, sequence: 128, width: 128, heads: 2, ffn: 128, totalTokens: 8192, attentionElements: 8388608 },
  { id: '11', batch: 64, sequence: 128, width: 128, heads: 16, ffn: 128, totalTokens: 8192, attentionElements: 67108864 },
  { id: '12', batch: 64, sequence: 32, width: 128, heads: 4, ffn: 128, totalTokens: 2048, attentionElements: 1048576 },
  { id: '13', batch: 64, sequence: 1024, width: 128, heads: 4, ffn: 128, totalTokens: 65536, attentionElements: 1073741824 },
  { id: '14', batch: 32, sequence: 100000, width: 1024, heads: 16, ffn: 1024, totalTokens: 3200000, attentionElements: 10240000000000 },
]

export const programExamples = [
  { id: '02', schedule: 'CUDA GRAPH', attention: 'EFFICIENT SDPA', ffn: 'LINEAR + GELU' },
  { id: '08', schedule: 'COMPILED', attention: 'CAUSAL SDPA', ffn: 'TORCH' },
  { id: '13', schedule: 'EAGER', attention: 'TRITON S1024', ffn: 'COMPILED' },
] as const

export const searchEvidence = [
  { label: 'SCREEN', value: 3933 },
  { label: 'ENHANCED', value: 381 },
  { label: 'FORMAL', value: 50 },
  { label: 'UPDATES', value: 6 },
] as const

export const promotionThresholds = [
  { blocks: '6/6', ratio: '≥ 1.10×' },
  { blocks: '8/9', ratio: '≥ 1.05×' },
  { blocks: '11/13', ratio: '≥ 1.02×' },
] as const

export const performance: PerformancePoint[] = [
  { id: '01', baselineMs: 1.727088, deployedMs: 0.218112, speedup: 7.918354 },
  { id: '02', baselineMs: 1.87136, deployedMs: 0.07168, speedup: 26.107142 },
  { id: '03', baselineMs: 1.866464, deployedMs: 0.08192, speedup: 22.783985 },
  { id: '04', baselineMs: 1.910128, deployedMs: 0.09728, speedup: 19.635361 },
  { id: '05', baselineMs: 3.904512, deployedMs: 0.420864, speedup: 9.277372 },
  { id: '06', baselineMs: 491.954132, deployedMs: 36.817921, speedup: 13.361812 },
  { id: '07', baselineMs: 1.793472, deployedMs: 0.073728, speedup: 24.325521 },
  { id: '08', baselineMs: 21.433951, deployedMs: 6.280704, speedup: 3.412667 },
  { id: '09', baselineMs: 1.512576, deployedMs: 0.201728, speedup: 7.498096 },
  { id: '10', baselineMs: 1.68464, deployedMs: 0.197632, speedup: 8.524126 },
  { id: '11', baselineMs: 7.94216, deployedMs: 0.242688, speedup: 32.725804 },
  { id: '12', baselineMs: 1.75584, deployedMs: 0.101376, speedup: 17.320076 },
  { id: '13', baselineMs: 120.429569, deployedMs: 3.337216, speedup: 36.086837 },
]

export const equations = {
  tpe: 'l(x) / g(x)',
  speedup: 'sᵢ = median(Tbaseline,ᵢ) / median(Tdeployed,ᵢ)',
  geomean: 'G = exp[(1/13) Σ log(sᵢ)] = 14.4926',
} as const

export const shape14Note = 'S14 · STREAMED EXECUTION · EXCLUDED FROM GEOMEAN'

export const measurementProtocol = {
  comparator: 'OFFICIAL ELEMENTWISE TOLERANCE',
  precision: 'FP32 INPUT / OUTPUT',
  isolation: ['EXCLUSIVE GPU LEASE', 'FRESH PROCESS PER SHAPE'],
  timing: 'CUDA EVENT TIMING',
  residentPreset: '5 CORRECTNESS · 20 WARMUPS · 100 REPEATS · 3 ROUNDS',
} as const

export const resultBoundary = [
  'MEASURED ON ONE RTX 4080',
  '≈ 12 HOURS OF CUMULATIVE SEARCH',
  'BEST FOUND — NOT A GLOBAL OPTIMUM',
  'LOCAL ENGINEERING RESULT — NOT AN OFFICIAL SCORE',
] as const

export const evidenceSources = {
  performance: '../../docs/technical_report/figures/source_data/performance.csv',
  workloads: '../../docs/technical_report/figures/source_data/workloads.csv',
  search: '../../docs/technical_report/figures/source_data/search_flow.csv',
  programs: '../../docs/technical_report/figures/source_data/deployed_programs.csv',
} as const
