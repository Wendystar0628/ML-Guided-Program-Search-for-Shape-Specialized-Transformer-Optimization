export type ProgressRange = Readonly<{
  start: number
  end: number
}>

export const PROGRAM_FIELD_RANGES = {
  programSpace: { start: 0, end: 0.24 },
  converge: { start: 0.24, end: 0.5 },
  measure: { start: 0.5, end: 0.72 },
  winner: { start: 0.72, end: 0.88 },
  dock: { start: 0.88, end: 1 },
} as const satisfies Record<string, ProgressRange>

export function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1)
}

export function mix(from: number, to: number, amount: number): number {
  return from + (to - from) * clamp01(amount)
}

export function smoothstep(edge0: number, edge1: number, value: number): number {
  if (edge0 === edge1) {
    return value < edge0 ? 0 : 1
  }

  const amount = clamp01((value - edge0) / (edge1 - edge0))
  return amount * amount * (3 - 2 * amount)
}

export function rangeProgress(value: number, range: ProgressRange): number {
  return smoothstep(range.start, range.end, value)
}
