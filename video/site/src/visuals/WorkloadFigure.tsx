import {
  programExamples,
  workloads,
  type Workload,
} from '../data/projectData'

const plot = {
  left: 128,
  right: 1040,
  top: 96,
  bottom: 886,
} as const

const domain = {
  tokens: [2, 6.2],
  attention: [5, 9.6],
} as const

const residentWorkloads = workloads.filter((workload) => workload.id !== '14')

const highlightedShapes = new Set(['02', '08', '13'])

const programLaneY: Record<string, number> = {
  '02': 796,
  '08': 566,
  '13': 330,
}

const programColumns = [1190, 1390, 1590] as const

type PlotPoint = Workload & {
  x: number
  y: number
  radius: number
}

function scaleLog(
  value: number,
  input: readonly [number, number],
  output: readonly [number, number],
) {
  const position = (Math.log10(value) - input[0]) / (input[1] - input[0])
  return output[0] + position * (output[1] - output[0])
}

function buildPlotPoints(): PlotPoint[] {
  const clusters = new Map<string, Workload[]>()

  residentWorkloads.forEach((workload) => {
    const key = `${workload.totalTokens}:${workload.attentionElements}`
    const cluster = clusters.get(key) ?? []
    cluster.push(workload)
    clusters.set(key, cluster)
  })

  const clusterOffsets = [
    { x: 0, y: -20 },
    { x: -20, y: 14 },
    { x: 20, y: 14 },
    { x: 0, y: 30 },
  ]

  return residentWorkloads.map((workload) => {
    const key = `${workload.totalTokens}:${workload.attentionElements}`
    const cluster = clusters.get(key) ?? [workload]
    const clusterIndex = cluster.findIndex((item) => item.id === workload.id)
    const offset = cluster.length > 1
      ? clusterOffsets[clusterIndex] ?? { x: 0, y: clusterIndex * 12 }
      : { x: 0, y: 0 }

    return {
      ...workload,
      x: scaleLog(
        workload.totalTokens,
        domain.tokens,
        [plot.left, plot.right],
      ) + offset.x,
      y: scaleLog(
        workload.attentionElements,
        domain.attention,
        [plot.bottom, plot.top],
      ) + offset.y,
      radius: 6 + Math.max(0, Math.log2(workload.width / 32)) * 1.35,
    }
  })
}

const plotPoints = buildPlotPoints()

function getPoint(id: string) {
  return plotPoints.find((point) => point.id === id)
}

function getRoutePath(id: string, point: PlotPoint, laneY: number) {
  if (id === '02') {
    return `M ${point.x} ${point.y} C 870 ${point.y}, 1010 ${laneY}, 1160 ${laneY}`
  }

  if (id === '08') {
    return `M ${point.x} ${point.y} H 930 V ${laneY} H 1160`
  }

  return `M ${point.x} ${point.y} C 920 ${point.y}, 1030 ${laneY}, 1160 ${laneY}`
}

function ProgramGlyph({
  column,
  y,
  kind,
}: {
  column: number
  y: number
  kind: number
}) {
  if (kind === 1) {
    return (
      <path
        className="workload-program__glyph"
        d={`M ${column} ${y - 10} L ${column + 10} ${y} L ${column} ${y + 10} L ${column - 10} ${y} Z`}
      />
    )
  }

  if (kind === 2) {
    return (
      <path
        className="workload-program__glyph"
        d={`M ${column - 11} ${y - 7} H ${column + 4} L ${column + 11} ${y} L ${column + 4} ${y + 7} H ${column - 11} Z`}
      />
    )
  }

  return <circle className="workload-program__glyph" cx={column} cy={y} r="9" />
}

export function WorkloadFigure() {
  const tokenTicks = [100, 1_000, 10_000, 100_000, 1_000_000]
  const attentionTicks = [100_000, 10_000_000, 1_000_000_000, 100_000_000_000, 10_000_000_000_000]

  return (
    <svg
      className="workload-figure"
      viewBox="0 0 1720 1040"
      role="img"
      aria-labelledby="workload-figure-title workload-figure-description"
    >
      <title id="workload-figure-title">Resident workload shapes and their execution programs</title>
      <desc id="workload-figure-description">
        Log-scale resident workload geometry connects Shapes 02, 08, and 13 to three different GPU execution programs.
      </desc>

      <defs>
        <marker
          id="workload-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="6"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 Z" className="workload-arrowhead" />
        </marker>
        <pattern id="workload-hatch" width="12" height="12" patternUnits="userSpaceOnUse">
          <path d="M -3 3 L 3 -3 M 0 12 L 12 0 M 9 15 L 15 9" className="workload-hatch" />
        </pattern>
      </defs>

      <g className="workload-figure__plot">
        <rect
          className="workload-figure__field"
          x={plot.left}
          y={plot.top}
          width={plot.right - plot.left}
          height={plot.bottom - plot.top}
        />

        <path
          className="workload-regime workload-regime--attention"
          d="M 128 96 H 1040 V 282 C 820 248 646 292 492 360 C 350 424 232 412 128 372 Z"
        />
        <path
          className="workload-regime workload-regime--dense"
          d="M 128 886 V 666 C 322 608 480 634 630 572 C 794 504 894 456 1040 474 V 886 Z"
        />

        {tokenTicks.map((tick) => {
          const x = scaleLog(tick, domain.tokens, [plot.left, plot.right])
          return (
            <g key={tick} className="workload-axis workload-axis--x">
              <line x1={x} x2={x} y1={plot.top} y2={plot.bottom} />
              <text x={x} y={plot.bottom + 38} textAnchor="middle">
                10^{Math.round(Math.log10(tick))}
              </text>
            </g>
          )
        })}

        {attentionTicks.map((tick) => {
          const y = scaleLog(tick, domain.attention, [plot.bottom, plot.top])
          return (
            <g key={tick} className="workload-axis workload-axis--y">
              <line x1={plot.left} x2={plot.right} y1={y} y2={y} />
              <text x={plot.left - 24} y={y + 5} textAnchor="end">
                10^{Math.round(Math.log10(tick))}
              </text>
            </g>
          )
        })}

        <line
          className="workload-axis__baseline"
          x1={plot.left}
          x2={plot.right}
          y1={plot.bottom}
          y2={plot.bottom}
        />
        <line
          className="workload-axis__baseline"
          x1={plot.left}
          x2={plot.left}
          y1={plot.top}
          y2={plot.bottom}
        />
        <text
          className="workload-axis__label"
          x={(plot.left + plot.right) / 2}
          y={plot.bottom + 84}
          textAnchor="middle"
        >
          TOTAL TOKENS · B × S · LOG₁₀
        </text>
        <text
          className="workload-axis__label"
          x="42"
          y={(plot.top + plot.bottom) / 2}
          textAnchor="middle"
          transform={`rotate(-90 42 ${(plot.top + plot.bottom) / 2})`}
        >
          ATTENTION ELEMENTS · B × H × S² · LOG₁₀
        </text>

        {plotPoints.map((point) => {
          const highlighted = highlightedShapes.has(point.id)
          const labelToLeft = point.id === '06'
          return (
            <g
              key={point.id}
              className={`workload-point${highlighted ? ' workload-point--highlighted' : ''}`}
              transform={`translate(${point.x} ${point.y})`}
            >
              {highlighted && <circle className="workload-point__halo" r={point.radius + 15} />}
              <circle className="workload-point__mark" r={point.radius} />
              <line
                className="workload-point__tick"
                x1={labelToLeft ? -point.radius : point.radius}
                x2={labelToLeft ? -point.radius - 11 : point.radius + 11}
                y1="0"
                y2="0"
              />
              <text
                className="workload-point__label"
                x={labelToLeft ? -point.radius - 16 : point.radius + 16}
                y="5"
                textAnchor={labelToLeft ? 'end' : 'start'}
              >
                S{point.id}
              </text>
            </g>
          )
        })}
      </g>

      <g className="workload-programs">
        <text className="workload-programs__heading" x="1170" y="126">
          EXECUTION PROGRAMS
        </text>
        {['SCHEDULE', 'ATTENTION', 'FFN'].map((label, index) => (
          <text
            key={label}
            className="workload-programs__column"
            x={programColumns[index]}
            y="180"
            textAnchor="middle"
          >
            {label}
          </text>
        ))}

        {programExamples.map((program) => {
          const point = getPoint(program.id)
          const laneY = programLaneY[program.id]
          if (!point || laneY === undefined) return null

          const labels = [program.schedule, program.attention, program.ffn]
          return (
            <g key={program.id} className={`workload-program workload-program--${program.id}`}>
              <path
                className="workload-program__route"
                d={getRoutePath(program.id, point, laneY)}
                markerEnd="url(#workload-arrow)"
              />
              <line
                className="workload-program__rail"
                x1="1170"
                x2="1642"
                y1={laneY}
                y2={laneY}
              />
              <text className="workload-program__shape" x="1170" y={laneY - 42}>
                S{program.id}
              </text>
              {labels.map((label, index) => (
                <g key={label}>
                  <ProgramGlyph column={programColumns[index]} y={laneY} kind={index} />
                  <text
                    className="workload-program__label"
                    x={programColumns[index]}
                    y={laneY + 42}
                    textAnchor="middle"
                  >
                    {label}
                  </text>
                </g>
              ))}
            </g>
          )
        })}
      </g>

      <g className="workload-figure__calibration" aria-hidden="true">
        <path d="M 1128 96 H 1152 M 1140 84 V 108" />
        <path d="M 1652 886 H 1676 M 1664 874 V 898" />
        <text x="1128" y="1010">LOG GEOMETRY / PROGRAM DIVERGENCE</text>
      </g>
    </svg>
  )
}
