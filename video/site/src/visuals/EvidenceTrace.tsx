import type { CSSProperties } from 'react'
import type { EvidenceTraceVariant } from '../motion/motionTypes'

export type EvidenceTraceProps = {
  variant: EvidenceTraceVariant
  label?: string
}

const defaultLabels: Record<EvidenceTraceVariant, string> = {
  'outcome-workloads': 'Measured program evaluation across workload shapes',
  'workloads-architecture': 'Workload and environment fingerprints define the ConfigSpec',
  'architecture-search': 'ExecutionPlans define the searchable program domain',
  'search-evidence': 'Formal evidence determines deployment registry updates',
  'evidence-closing': 'Equal-shape aggregation of thirteen deployment ratios',
}

const traceMeta: Record<EvidenceTraceVariant, { index: string; eyebrow: string }> = {
  'outcome-workloads': { index: '01→02', eyebrow: 'Outcome to problem setting' },
  'workloads-architecture': { index: '02→03', eyebrow: 'Problem setting to system architecture' },
  'architecture-search': { index: '03→04', eyebrow: 'System architecture to search method' },
  'search-evidence': { index: '04→05', eyebrow: 'Search method to evaluation results' },
  'evidence-closing': { index: 'Σ', eyebrow: 'Aggregate equal-shape gains' },
}

function TraceOrnament({ variant }: { variant: EvidenceTraceVariant }) {
  if (variant === 'outcome-workloads') {
    return (
      <div className="trace-ornament trace-ornament--fan" aria-hidden="true">
        <i className="trace-fan__seed" />
        <span /><span /><span />
      </div>
    )
  }

  if (variant === 'workloads-architecture') {
    return (
      <div className="trace-ornament trace-ornament--fingerprint" aria-hidden="true">
        <span>B</span><span>S</span><span>D</span><span>H</span><span>F</span>
        <i>ConfigSpec</i>
      </div>
    )
  }

  if (variant === 'architecture-search') {
    return (
      <div className="trace-ornament trace-ornament--loop" aria-hidden="true">
        <span>Plan</span><i>→</i><span>Measure</span><i>→</i><span>Learn</span><b>↺</b>
      </div>
    )
  }

  if (variant === 'search-evidence') {
    return (
      <div className="trace-ornament trace-ornament--pair" aria-hidden="true">
        <span>Challenger</span><span>Incumbent</span><i>≥ 2%</i><b>Registry</b>
      </div>
    )
  }

  return (
    <div className="trace-ornament trace-ornament--ratios" aria-hidden="true">
      {Array.from({ length: 13 }, (_, index) => (
        <i
          key={index}
          style={{
            '--ratio-height': `${28 + index * 4}px`,
            '--ratio-opacity': `${0.42 + index * 0.045}`,
          } as CSSProperties}
        />
      ))}
      <strong>Geomean</strong>
    </div>
  )
}

export function EvidenceTrace({ variant, label }: EvidenceTraceProps) {
  const resolvedLabel = label ?? defaultLabels[variant]
  const meta = traceMeta[variant]

  return (
    <aside
      className={`evidence-trace evidence-trace--${variant}`}
      aria-label={resolvedLabel}
      data-trace-variant={variant}
    >
      <div className="evidence-trace__copy">
        <span>{meta.index}</span>
        <div>
          <small>{meta.eyebrow}</small>
          <p>{resolvedLabel}</p>
        </div>
      </div>
      <TraceOrnament variant={variant} />
    </aside>
  )
}
