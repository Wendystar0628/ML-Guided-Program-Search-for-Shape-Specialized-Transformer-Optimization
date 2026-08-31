export function EvidenceTrace() {
  return (
    <svg
      className="evidence-trace"
      viewBox="0 0 80 1000"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <path className="trace-context" d="M40 0V1000" />
      <path className="trace-signal" d="M40 0V1000" pathLength="1" />
      {[80, 330, 580, 830].map((y, index) => (
        <g key={y} transform={'translate(40 ' + y + ')'}>
          <circle r="12" className="trace-node" />
          <text x="-25" y="5" textAnchor="end">
            0{index + 1}
          </text>
        </g>
      ))}
    </svg>
  )
}
