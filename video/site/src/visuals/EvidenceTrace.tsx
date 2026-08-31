import type { EvidenceTraceVariant } from '../motion/motionTypes'

export type EvidenceTraceProps = {
  variant: EvidenceTraceVariant
  label?: string
}

const defaultLabels: Record<EvidenceTraceVariant, string> = {
  'outcome-workloads': 'MEASURED WINNER → WORKLOAD',
  'workloads-architecture': 'ROUTES MERGE → CONFIGSPEC',
  'architecture-search': 'EXECUTIONPLAN → CANDIDATE FIELD',
  'search-evidence': 'FORMAL WINNER → DEPLOYED MARKER',
  'evidence-closing': 'MEASURED ROWS → GEOMEAN',
}

function OutcomeToWorkloads() {
  const path = 'M1080 28 C1044 94 914 108 758 134 C562 168 356 214 112 250'

  return (
    <>
      <path className="evidence-trace__context" d={path} />
      <path className="evidence-trace__signal" d={path} pathLength="1" />
      <circle className="evidence-trace__node evidence-trace__node--winner" cx="1080" cy="28" r="16" />
      <circle className="evidence-trace__node" cx="112" cy="250" r="12" />
      <text className="evidence-trace__endpoint-label" x="1018" y="70" textAnchor="end">
        WINNER
      </text>
      <text className="evidence-trace__endpoint-label" x="142" y="232">
        S01
      </text>
    </>
  )
}

function WorkloadsToArchitecture() {
  const routes = [
    'M88 32 C268 32 356 90 520 138',
    'M88 138 H520',
    'M88 244 C268 244 356 186 520 138',
  ]
  const ribbon = 'M520 138 C682 138 814 138 1110 138'

  return (
    <>
      {routes.map((route) => (
        <path key={route} className="evidence-trace__context" d={route} />
      ))}
      {routes.map((route) => (
        <path key={`signal-${route}`} className="evidence-trace__signal" d={route} pathLength="1" />
      ))}
      <path className="evidence-trace__ribbon" d={ribbon} />
      <circle className="evidence-trace__node" cx="520" cy="138" r="13" />
      <circle className="evidence-trace__node" cx="1110" cy="138" r="12" />
      <text className="evidence-trace__endpoint-label" x="550" y="112">
        CONFIGSPEC
      </text>
    </>
  )
}

function ArchitectureToSearch() {
  const candidates = [46, 104, 170, 232]

  return (
    <>
      <path className="evidence-trace__context" d="M92 138 H456" />
      <path className="evidence-trace__signal" d="M92 138 H456" pathLength="1" />
      {candidates.map((y) => {
        const path = `M456 138 C646 138 744 ${y} 1106 ${y}`
        return (
          <g key={y}>
            <path className="evidence-trace__context" d={path} />
            <path className="evidence-trace__candidate" d={path} pathLength="1" />
            <circle className="evidence-trace__node" cx="1106" cy={y} r="9" />
          </g>
        )
      })}
      <circle className="evidence-trace__node" cx="456" cy="138" r="13" />
      <text className="evidence-trace__endpoint-label" x="112" y="112">
        EXECUTIONPLAN
      </text>
      <text className="evidence-trace__endpoint-label" x="1034" y="270">
        CANDIDATES
      </text>
    </>
  )
}

function SearchToEvidence() {
  const path = 'M92 76 C342 76 590 76 832 76 Q990 76 990 188 V254'

  return (
    <>
      <path className="evidence-trace__context" d={path} />
      <path className="evidence-trace__winner-path" d={path} pathLength="1" />
      <circle className="evidence-trace__node evidence-trace__node--winner" cx="832" cy="76" r="15" />
      <path className="evidence-trace__deployed-marker" d="M962 254 H1018" />
      <text className="evidence-trace__endpoint-label" x="804" y="48" textAnchor="end">
        FORMAL WINNER
      </text>
      <text className="evidence-trace__endpoint-label" x="1036" y="262">
        DEPLOYED
      </text>
    </>
  )
}

function EvidenceToClosing() {
  const rows = Array.from({ length: 13 }, (_, index) => 18 + index * 20)

  return (
    <>
      {rows.map((y, index) => {
        const startX = 92 + (index % 4) * 20
        const mergeX = 596 + (index % 3) * 34
        const path = `M${startX} ${y} H${mergeX} C820 ${y} 846 138 1038 138`
        return (
          <path
            key={y}
            className="evidence-trace__aggregate-row"
            d={path}
            pathLength="1"
          />
        )
      })}
      <circle className="evidence-trace__node evidence-trace__node--winner" cx="1038" cy="138" r="24" />
      <circle className="evidence-trace__aggregate-ring" cx="1038" cy="138" r="40" />
      <text className="evidence-trace__aggregate-value" x="1094" y="130">
        14.49×
      </text>
      <text className="evidence-trace__endpoint-label" x="1094" y="158">
        GEOMEAN
      </text>
    </>
  )
}

export function EvidenceTrace({ variant, label }: EvidenceTraceProps) {
  return (
    <svg
      className={`evidence-trace evidence-trace--${variant}`}
      viewBox="0 0 1200 280"
      role="img"
      aria-label={label ?? defaultLabels[variant]}
      data-trace-variant={variant}
    >
      <title>{label ?? defaultLabels[variant]}</title>
      {variant === 'outcome-workloads' && <OutcomeToWorkloads />}
      {variant === 'workloads-architecture' && <WorkloadsToArchitecture />}
      {variant === 'architecture-search' && <ArchitectureToSearch />}
      {variant === 'search-evidence' && <SearchToEvidence />}
      {variant === 'evidence-closing' && <EvidenceToClosing />}
      <text className="evidence-trace__label" x="600" y="274" textAnchor="middle">
        {label ?? defaultLabels[variant]}
      </text>
    </svg>
  )
}
