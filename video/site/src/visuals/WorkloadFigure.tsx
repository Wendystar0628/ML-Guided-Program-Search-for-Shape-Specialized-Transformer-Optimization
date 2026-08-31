import { useState, type CSSProperties } from 'react'

import {
  programExamples,
  workloads,
  type Workload,
} from '../data/projectData'

type WorkloadStep = 'map' | 'specialize'

type WorkloadFigureProps = {
  activeStep?: WorkloadStep | string
}

type PlotPoint = Workload & {
  x: number
  y: number
  radius: number
}

const plot = {
  left: 130,
  right: 1010,
  top: 118,
  bottom: 892,
} as const

const domain = {
  tokens: [2, 6.2],
  attention: [5, 9.6],
} as const

const residentWorkloads = workloads.filter((workload) => workload.id !== '14')
const highlightedShapes = new Set(['02', '08', '13'])

const programLaneY: Record<string, number> = {
  '02': 804,
  '08': 566,
  '13': 328,
}

const representativeLabels: Record<
  string,
  { geometry: string; route: string; tail: string }
> = {
  '02': {
    geometry: 'SMALL BATCH · B1',
    route: 'CUDA GRAPH · EFFICIENT SDPA',
    tail: 'LINEAR + GELU FFN',
  },
  '08': {
    geometry: 'WIDE MODEL · D1024',
    route: 'COMPILED · CAUSAL SDPA',
    tail: 'TORCH FFN',
  },
  '13': {
    geometry: 'LONG SEQUENCE · S1024',
    route: 'TRITON S1024 · EAGER',
    tail: 'COMPILED FFN',
  },
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
    { x: 0, y: -26 },
    { x: -28, y: 18 },
    { x: 28, y: 18 },
    { x: 0, y: 38 },
  ]

  return residentWorkloads.map((workload) => {
    const key = `${workload.totalTokens}:${workload.attentionElements}`
    const cluster = clusters.get(key) ?? [workload]
    const clusterIndex = cluster.findIndex((item) => item.id === workload.id)
    const offset = cluster.length > 1
      ? clusterOffsets[clusterIndex] ?? { x: 0, y: clusterIndex * 14 }
      : { x: 0, y: 0 }

    return {
      ...workload,
      x: scaleLog(workload.totalTokens, domain.tokens, [plot.left, plot.right]) + offset.x,
      y: scaleLog(
        workload.attentionElements,
        domain.attention,
        [plot.bottom, plot.top],
      ) + offset.y,
      radius: 9 + Math.max(0, Math.log2(workload.width / 32)) * 1.8,
    }
  })
}

const plotPoints = buildPlotPoints()

function getPoint(id: string) {
  return plotPoints.find((point) => point.id === id)
}

function getRoutePath(id: string, point: PlotPoint, laneY: number) {
  if (id === '02') {
    return `M ${point.x} ${point.y} C 820 ${point.y}, 990 ${laneY}, 1146 ${laneY}`
  }

  if (id === '08') {
    return `M ${point.x} ${point.y} C 790 ${point.y}, 970 ${laneY}, 1146 ${laneY}`
  }

  return `M ${point.x} ${point.y} C 850 ${point.y}, 1010 ${laneY}, 1146 ${laneY}`
}

function pointAriaLabel(point: PlotPoint) {
  const program = programExamples.find((candidate) => candidate.id === point.id)
  const geometry = `Shape ${point.id}. Batch ${point.batch}, sequence ${point.sequence}, width ${point.width}, heads ${point.heads}, feed forward width ${point.ffn}.`

  if (!program) return geometry

  return `${geometry} ${program.schedule}, ${program.attention}, ${program.ffn}.`
}

function annotationStyle(point: PlotPoint): CSSProperties {
  return {
    '--annotation-x': `${(point.x / 1720) * 100}%`,
    '--annotation-y': `${(point.y / 1040) * 100}%`,
  } as CSSProperties
}

export function WorkloadFigure({ activeStep = 'specialize' }: WorkloadFigureProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [focusedId, setFocusedId] = useState<string | null>(null)
  const inspectedId = focusedId ?? hoveredId
  const inspectedPoint = inspectedId ? getPoint(inspectedId) : undefined
  const inspectedProgram = inspectedId
    ? programExamples.find((program) => program.id === inspectedId)
    : undefined
  const isSpecialized = activeStep === 'specialize'

  const tokenTicks = [100, 1_000, 10_000, 100_000, 1_000_000]
  const attentionTicks = [
    100_000,
    10_000_000,
    1_000_000_000,
    100_000_000_000,
  ]

  return (
    <div
      className="workload-figure-shell"
      data-state={isSpecialized ? 'specialize' : 'map'}
      data-inspected-shape={inspectedId ?? undefined}
    >
      <svg
        className="workload-figure"
        viewBox="0 0 1720 1040"
        role="img"
        aria-labelledby="workload-figure-title workload-figure-description"
      >
        <title id="workload-figure-title">
          Resident workload geometry and shape-specialized execution programs
        </title>
        <desc id="workload-figure-description">
          Thirteen resident workload shapes share one Transformer meaning but occupy
          different GPU regimes. Shapes 02, 08, and 13 branch into three measured
          execution programs.
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
            <path
              d="M -3 3 L 3 -3 M 0 12 L 12 0 M 9 15 L 15 9"
              className="workload-hatch"
            />
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
            d="M 130 118 H 1010 V 304 C 822 270 656 300 506 372 C 350 446 230 418 130 390 Z"
          />
          <path
            className="workload-regime workload-regime--dense"
            d="M 130 892 V 684 C 316 620 472 650 626 584 C 786 516 884 470 1010 486 V 892 Z"
          />
          <text className="workload-regime__label" x="158" y="166">
            ATTENTION WORKING SET
          </text>
          <text className="workload-regime__label" x="158" y="850">
            LAUNCH + SMALL-MATRIX REGIME
          </text>

          {tokenTicks.map((tick) => {
            const x = scaleLog(tick, domain.tokens, [plot.left, plot.right])
            return (
              <g key={tick} className="workload-axis workload-axis--x">
                <line x1={x} x2={x} y1={plot.top} y2={plot.bottom} />
                <text x={x} y={plot.bottom + 42} textAnchor="middle">
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
                <text x={plot.left - 25} y={y + 7} textAnchor="end">
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
            y={plot.bottom + 94}
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

          {plotPoints.map((point, index) => {
            const highlighted = highlightedShapes.has(point.id)
            const inspected = inspectedId === point.id
            const muted = inspectedId !== null && !inspected
            const labelToLeft = point.id === '06'

            return (
              <g
                key={point.id}
                className={[
                  'workload-point',
                  highlighted ? 'workload-point--highlighted' : '',
                  inspected ? 'is-inspected' : '',
                  muted ? 'is-muted' : '',
                ].filter(Boolean).join(' ')}
                style={{ '--point-index': index } as CSSProperties}
                transform={`translate(${point.x} ${point.y})`}
                tabIndex={0}
                focusable="true"
                aria-label={pointAriaLabel(point)}
                onMouseEnter={() => setHoveredId(point.id)}
                onMouseLeave={() => setHoveredId(null)}
                onFocus={() => setFocusedId(point.id)}
                onBlur={() => setFocusedId(null)}
              >
                {highlighted && (
                  <circle className="workload-point__halo" r={point.radius + 18} />
                )}
                <circle className="workload-point__mark" r={point.radius} />
                <line
                  className="workload-point__tick"
                  x1={labelToLeft ? -point.radius : point.radius}
                  x2={labelToLeft ? -point.radius - 14 : point.radius + 14}
                  y1="0"
                  y2="0"
                />
                <text
                  className="workload-point__label"
                  x={labelToLeft ? -point.radius - 20 : point.radius + 20}
                  y="7"
                  textAnchor={labelToLeft ? 'end' : 'start'}
                >
                  S{point.id}
                </text>
              </g>
            )
          })}
        </g>

        <g className="workload-programs">
          <text className="workload-programs__heading" x="1150" y="130">
            THREE SHAPES. THREE PROGRAMS.
          </text>
          <text className="workload-programs__subheading" x="1150" y="174">
            MEASURED ROUTES, NOT MANUAL POLICY LABELS
          </text>

          {programExamples.map((program) => {
            const point = getPoint(program.id)
            const laneY = programLaneY[program.id]
            const labels = representativeLabels[program.id]
            if (!point || laneY === undefined || !labels) return null

            const inspected = inspectedId === program.id
            const muted = inspectedId !== null && !inspected

            return (
              <g
                key={program.id}
                className={[
                  `workload-program workload-program--${program.id}`,
                  isSpecialized ? 'is-routed' : 'is-pending',
                  inspected ? 'is-inspected' : '',
                  muted ? 'is-muted' : '',
                ].filter(Boolean).join(' ')}
              >
                <path
                  className="workload-program__route"
                  d={getRoutePath(program.id, point, laneY)}
                  markerEnd="url(#workload-arrow)"
                  pathLength="1"
                />
                <circle className="workload-program__anchor" cx="1162" cy={laneY} r="11" />
                <line
                  className="workload-program__rule"
                  x1="1188"
                  x2="1670"
                  y1={laneY}
                  y2={laneY}
                />
                <text className="workload-program__shape" x="1194" y={laneY - 66}>
                  S{program.id} · {labels.geometry}
                </text>
                <text className="workload-program__label" x="1194" y={laneY - 24}>
                  {labels.route}
                </text>
                <text className="workload-program__detail" x="1194" y={laneY + 42}>
                  {labels.tail}
                </text>
              </g>
            )
          })}
        </g>

        <g className="workload-figure__calibration" aria-hidden="true">
          <path d="M 1122 112 H 1146 M 1134 100 V 124" />
          <path d="M 1654 894 H 1678 M 1666 882 V 906" />
          <text x="1122" y="1004">GEOMETRY → BOTTLENECK → EXECUTABLE PROGRAM</text>
        </g>
      </svg>

      {inspectedPoint && (
        <aside
          className={[
            'workload-point-annotation',
            inspectedPoint.x > 740 ? 'workload-point-annotation--left' : '',
            inspectedPoint.y < 330 ? 'workload-point-annotation--below' : '',
          ].filter(Boolean).join(' ')}
          style={annotationStyle(inspectedPoint)}
          aria-live="polite"
        >
          <span className="workload-point-annotation__id">
            SHAPE {inspectedPoint.id}
          </span>
          <strong>
            BATCH {inspectedPoint.batch} · SEQUENCE {inspectedPoint.sequence} · WIDTH{' '}
            {inspectedPoint.width}
          </strong>
          <span>
            HEADS {inspectedPoint.heads} · FFN {inspectedPoint.ffn}
          </span>
          {inspectedProgram && (
            <span className="workload-point-annotation__program">
              {inspectedProgram.schedule} · {inspectedProgram.attention} ·{' '}
              {inspectedProgram.ffn}
            </span>
          )}
        </aside>
      )}
    </div>
  )
}
