import { useMemo, useState } from 'react'

import {
  programExamples,
  workloads,
  type Workload,
} from '../data/projectData'
import type { WorkloadStep } from '../motion/motionTypes'

type WorkloadFigureProps = {
  activeStep?: WorkloadStep
}

type PlotPoint = Workload & {
  x: number
  y: number
  radius: number
}

const residentWorkloads = workloads.filter((workload) => workload.id !== '14')
const representativeIds = new Set(['02', '08', '13'])

const representativeContext = {
  '02': {
    title: 'SMALL-BATCH WORKLOAD',
    geometry: 'B = 1 · S = 128 · D = 128',
    route: 'CUDA GRAPH · EFFICIENT SDPA · LINEAR + GELU FFN',
  },
  '08': {
    title: 'WIDE-MODEL WORKLOAD',
    geometry: 'B = 64 · S = 128 · D = 1,024',
    route: 'COMPILED · CAUSAL SDPA · TORCH FFN',
  },
  '13': {
    title: 'LONG-SEQUENCE WORKLOAD',
    geometry: 'B = 64 · S = 1,024 · D = 128',
    route: 'EAGER · TRITON S1024 · COMPILED FFN',
  },
} as const

const plot = {
  left: 154,
  right: 1372,
  top: 92,
  bottom: 602,
} as const

const logDomain = {
  tokens: [2, 6.2] as const,
  attention: [5, 9.6] as const,
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

  return residentWorkloads.map((workload) => {
    const key = `${workload.totalTokens}:${workload.attentionElements}`
    const cluster = clusters.get(key) ?? [workload]
    const clusterIndex = cluster.findIndex((item) => item.id === workload.id)
    const clusterAngle = -Math.PI / 2 + (clusterIndex / cluster.length) * Math.PI * 2
    const clusterRadius = cluster.length > 1 ? 28 + cluster.length * 2 : 0

    return {
      ...workload,
      x:
        scaleLog(workload.totalTokens, logDomain.tokens, [plot.left, plot.right]) +
        Math.cos(clusterAngle) * clusterRadius,
      y:
        scaleLog(
          workload.attentionElements,
          logDomain.attention,
          [plot.bottom, plot.top],
        ) + Math.sin(clusterAngle) * clusterRadius,
      radius: 10 + Math.max(0, Math.log2(workload.width / 32)) * 2,
    }
  })
}

function pointAriaLabel(point: PlotPoint) {
  return `Shape ${point.id}. Batch ${point.batch}, sequence ${point.sequence}, width ${point.width}, heads ${point.heads}, feed forward width ${point.ffn}.`
}

function formatPower(value: number) {
  const superscript = String(Math.round(Math.log10(value)))
    .replace('0', '⁰')
    .replace('1', '¹')
    .replace('2', '²')
    .replace('3', '³')
    .replace('4', '⁴')
    .replace('5', '⁵')
    .replace('6', '⁶')
    .replace('7', '⁷')
    .replace('8', '⁸')
    .replace('9', '⁹')

  return `10${superscript}`
}

export function WorkloadFigure({ activeStep = 'specialize' }: WorkloadFigureProps) {
  const plotPoints = useMemo(buildPlotPoints, [])
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [focusedId, setFocusedId] = useState<string | null>(null)
  const inspectedId = focusedId ?? hoveredId ?? '08'
  const inspectedPoint =
    plotPoints.find((point) => point.id === inspectedId) ?? plotPoints[0]
  const inspectedProgram = programExamples.find((program) => program.id === inspectedId)

  const tokenTicks = [100, 1_000, 10_000, 100_000, 1_000_000]
  const attentionTicks = [100_000, 10_000_000, 1_000_000_000]

  const inspectOnHover = (id: string) => setHoveredId(id)
  const clearHover = (id: string) => {
    setHoveredId((current) => (current === id ? null : current))
  }
  const inspectOnFocus = (id: string) => setFocusedId(id)
  const clearFocus = (id: string) => {
    setFocusedId((current) => (current === id ? null : current))
  }

  return (
    <div className="workload-plate" data-state={activeStep}>
      <figure className="workload-plate__figure">
        <div className="workload-plate__figure-index">
          <span>FIGURE 01 · RESIDENT WORKLOAD ATLAS</span>
          <span>X = B × S · Y = B × H × S²</span>
          <span>POINT AREA ∝ MODEL WIDTH D</span>
        </div>

        <svg
          className="workload-plate__plot"
          viewBox="0 0 1480 720"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-labelledby="workload-plot-title workload-plot-description"
        >
          <title id="workload-plot-title">Resident workload geometry</title>
          <desc id="workload-plot-description">
            Thirteen resident Transformer shapes plotted by total tokens and attention
            elements. Orange rings identify Shapes 02, 08, and 13 as representative
            measured program routes.
          </desc>

          <rect
            className="workload-plate__field"
            x={plot.left}
            y={plot.top}
            width={plot.right - plot.left}
            height={plot.bottom - plot.top}
          />

          <path
            className="workload-plate__zone workload-plate__zone--attention"
            d="M154 92 H1372 V224 C1108 214 886 236 686 292 C460 356 296 334 154 302 Z"
          />
          <path
            className="workload-plate__zone workload-plate__zone--launch"
            d="M154 602 V488 C424 434 646 462 852 420 C1052 380 1224 354 1372 370 V602 Z"
          />
          <text className="workload-plate__zone-label" x="184" y="138">
            HIGH ATTENTION-ELEMENT REGION
          </text>
          <text className="workload-plate__zone-label" x="184" y="564">
            LOW TOKEN-VOLUME REGION
          </text>

          {tokenTicks.map((tick) => {
            const x = scaleLog(tick, logDomain.tokens, [plot.left, plot.right])
            return (
              <g key={tick} className="workload-plate__axis">
                <line x1={x} x2={x} y1={plot.top} y2={plot.bottom} />
                <text x={x} y={plot.bottom + 38} textAnchor="middle">
                  {formatPower(tick)}
                </text>
              </g>
            )
          })}

          {attentionTicks.map((tick) => {
            const y = scaleLog(tick, logDomain.attention, [plot.bottom, plot.top])
            return (
              <g key={tick} className="workload-plate__axis">
                <line x1={plot.left} x2={plot.right} y1={y} y2={y} />
                <text x={plot.left - 26} y={y + 7} textAnchor="end">
                  {formatPower(tick)}
                </text>
              </g>
            )
          })}

          <line
            className="workload-plate__inspection-guide"
            x1={inspectedPoint.x}
            x2={inspectedPoint.x}
            y1={plot.top}
            y2={plot.bottom}
          />
          <line
            className="workload-plate__inspection-guide"
            x1={plot.left}
            x2={plot.right}
            y1={inspectedPoint.y}
            y2={inspectedPoint.y}
          />

          <text
            className="workload-plate__axis-label"
            x={(plot.left + plot.right) / 2}
            y="690"
            textAnchor="middle"
          >
            TOTAL TOKENS · B × S · LOG₁₀
          </text>
          <text
            className="workload-plate__axis-label"
            x="36"
            y={(plot.top + plot.bottom) / 2}
            textAnchor="middle"
            transform={`rotate(-90 36 ${(plot.top + plot.bottom) / 2})`}
          >
            ATTENTION ELEMENTS · B × H × S² · LOG₁₀
          </text>

          {plotPoints.map((point) => {
            const representative = representativeIds.has(point.id)
            const inspected = inspectedId === point.id
            const showLabel = inspected || (representative && !hoveredId && !focusedId)
            const labelAbove = point.y > plot.bottom - 68
            const labelToLeft = point.x > plot.right - 72

            return (
              <g
                key={point.id}
                className="workload-plate__point"
                data-representative={representative || undefined}
                data-inspected={inspected || undefined}
                transform={`translate(${point.x} ${point.y})`}
                tabIndex={0}
                aria-label={pointAriaLabel(point)}
                onPointerEnter={() => inspectOnHover(point.id)}
                onPointerLeave={() => clearHover(point.id)}
                onFocus={() => inspectOnFocus(point.id)}
                onBlur={() => clearFocus(point.id)}
              >
                {representative ? (
                  <circle className="workload-plate__point-halo" r={point.radius + 15} />
                ) : null}
                <circle className="workload-plate__point-mark" r={point.radius} />
                {showLabel ? (
                  <text
                    className="workload-plate__point-label"
                    x={labelAbove ? 0 : labelToLeft ? -point.radius - 13 : point.radius + 13}
                    y={labelAbove ? -point.radius - 12 : 7}
                    textAnchor={labelAbove ? 'middle' : labelToLeft ? 'end' : 'start'}
                  >
                    S{point.id}
                  </text>
                ) : null}
              </g>
            )
          })}
        </svg>

        <figcaption>
          Hover over or focus a mark to inspect its B / S / D / H / F geometry in the
          ledger below.
        </figcaption>
      </figure>

      <div className="workload-plate__regime-tabs" aria-label="Representative execution regimes">
        {programExamples.map((program) => {
          const context = representativeContext[program.id]
          const inspected = inspectedId === program.id

          return (
            <button
              key={program.id}
              type="button"
              className={`workload-plate__regime-tab workload-plate__regime-tab--${program.id}`}
              data-inspected={inspected || undefined}
              onPointerEnter={() => inspectOnHover(program.id)}
              onPointerLeave={() => clearHover(program.id)}
              onFocus={() => inspectOnFocus(program.id)}
              onBlur={() => clearFocus(program.id)}
            >
              <span>S{program.id}</span>
              <strong>{context.title}</strong>
              <small>{context.geometry}</small>
              <code>{context.route}</code>
            </button>
          )
        })}
      </div>

      <section className="workload-plate__inspection" aria-live="polite">
        <header>
          <span>INSPECTED SHAPE</span>
          <strong>S{inspectedPoint.id}</strong>
        </header>

        <dl aria-label={`Shape ${inspectedPoint.id} geometry`}>
          <div>
            <dt>B</dt>
            <dd>{inspectedPoint.batch.toLocaleString('en-US')}</dd>
          </div>
          <div>
            <dt>S</dt>
            <dd>{inspectedPoint.sequence.toLocaleString('en-US')}</dd>
          </div>
          <div>
            <dt>D</dt>
            <dd>{inspectedPoint.width.toLocaleString('en-US')}</dd>
          </div>
          <div>
            <dt>H</dt>
            <dd>{inspectedPoint.heads}</dd>
          </div>
          <div>
            <dt>F</dt>
            <dd>{inspectedPoint.ffn.toLocaleString('en-US')}</dd>
          </div>
        </dl>

        <div className="workload-plate__inspection-route">
          <span>{inspectedProgram ? 'REPRESENTATIVE MEASURED ROUTE' : 'RESIDENT ATLAS ENTRY'}</span>
          <strong>
            {inspectedProgram
              ? `${inspectedProgram.schedule} · ${inspectedProgram.attention} · ${inspectedProgram.ffn} FFN`
              : 'Route detail is retained in the complete deployment registry.'}
          </strong>
        </div>
      </section>
    </div>
  )
}
