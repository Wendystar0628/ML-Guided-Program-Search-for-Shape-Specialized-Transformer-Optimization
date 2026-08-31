import { equations, headline, performance } from '../data/projectData'

const VIEWBOX_WIDTH = 1680
const VIEWBOX_HEIGHT = 980
const PLOT_LEFT = 156
const PLOT_RIGHT = 1532
const PLOT_TOP = 238
const PLOT_BOTTOM = 804
const LOG_MIN = Math.log10(0.05)
const LOG_MAX = Math.log10(1000)

const majorTicks = [0.1, 1, 10, 100, 1000]
const minorTicks = [0.2, 0.5, 2, 5, 20, 50, 200, 500]
const highlightedShapes = new Set(['08', '11', '13'])

function latencyX(value: number) {
  const ratio = (Math.log10(value) - LOG_MIN) / (LOG_MAX - LOG_MIN)
  return PLOT_LEFT + ratio * (PLOT_RIGHT - PLOT_LEFT)
}

function shapeY(index: number) {
  return PLOT_TOP + (index * (PLOT_BOTTOM - PLOT_TOP)) / (performance.length - 1)
}

function speedupLabel(value: number) {
  return value.toFixed(2) + '×'
}

function latencyLabel(value: number) {
  return value >= 100 ? value.toFixed(1) : value.toFixed(3)
}

const speedupRange = performance.reduce(
  (range, point) => ({
    min: Math.min(range.min, point.speedup),
    max: Math.max(range.max, point.speedup),
  }),
  { min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY },
)

export function PerformanceFigure() {
  return (
    <figure className="performance-figure">
      <svg
        className="performance-figure__svg"
        viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
        role="img"
        aria-labelledby="performance-figure-title performance-figure-description"
      >
        <title id="performance-figure-title">Baseline and deployed latency by resident shape</title>
        <desc id="performance-figure-description">
          Log-scale dumbbell figure comparing baseline and deployed median latency for Shapes 01 through 13.
        </desc>

        <g className="performance-figure__heading">
          <text x={PLOT_LEFT} y="50" className="performance-figure__kicker">
            RESIDENT LATENCY · RTX 4080
          </text>
          <g className="performance-legend" transform="translate(1115 43)">
            <circle cx="0" cy="0" r="7" className="latency-marker baseline-marker" />
            <text x="18" y="5">BASELINE</text>
            <circle cx="184" cy="0" r="7" className="latency-marker deployed-marker" />
            <text x="202" y="5">DEPLOYED</text>
          </g>
        </g>

        <g className="performance-summary" aria-label="Aggregate performance">
          <g transform={`translate(${PLOT_LEFT} 112)`}>
            <text className="performance-summary__value">
              {headline.geomean.toFixed(2)}×
            </text>
            <text y="31" className="performance-summary__label">
              GEOMEAN
            </text>
          </g>
          <g transform="translate(594 112)">
            <text className="performance-summary__value">
              {speedupRange.min.toFixed(2)}×—{speedupRange.max.toFixed(2)}×
            </text>
            <text y="31" className="performance-summary__label">
              SPEEDUP RANGE
            </text>
          </g>
          <g transform={`translate(${PLOT_RIGHT} 112)`}>
            <text textAnchor="end" className="performance-summary__value">
              {headline.residentPassed} / {headline.residentTotal}
            </text>
            <text y="31" textAnchor="end" className="performance-summary__label">
              PASS
            </text>
          </g>
          <line x1={PLOT_LEFT} y1="169" x2={PLOT_RIGHT} y2="169" className="summary-rule" />
        </g>

        <g className="performance-grid" aria-hidden="true">
          {minorTicks.map((tick) => {
            const x = latencyX(tick)
            return (
              <line
                key={tick}
                x1={x}
                x2={x}
                y1={PLOT_TOP - 20}
                y2={PLOT_BOTTOM + 18}
                className="grid-line grid-line--minor"
              />
            )
          })}
          {majorTicks.map((tick) => {
            const x = latencyX(tick)
            return (
              <g key={tick}>
                <line
                  x1={x}
                  x2={x}
                  y1={PLOT_TOP - 20}
                  y2={PLOT_BOTTOM + 18}
                  className="grid-line grid-line--major"
                />
                <text x={x} y={PLOT_BOTTOM + 52} textAnchor="middle" className="axis-tick-label">
                  {tick}
                </text>
              </g>
            )
          })}
        </g>

        <g className="performance-rows">
          {performance.map((point, index) => {
            const y = shapeY(index)
            const baselineX = latencyX(point.baselineMs)
            const deployedX = latencyX(point.deployedMs)
            const isHighlighted = highlightedShapes.has(point.id)
            const rowClassName = isHighlighted
              ? 'performance-row performance-row--highlighted'
              : 'performance-row'

            return (
              <g key={point.id} className={rowClassName}>
                <line
                  x1={PLOT_LEFT - 12}
                  x2={PLOT_RIGHT}
                  y1={y}
                  y2={y}
                  className="performance-row__guide"
                />
                <text x={PLOT_LEFT - 29} y={y + 5} textAnchor="end" className="shape-label">
                  S{point.id}
                </text>
                <line
                  x1={deployedX}
                  x2={baselineX}
                  y1={y}
                  y2={y}
                  className="dumbbell-connector"
                />
                <circle cx={baselineX} cy={y} r={isHighlighted ? 8 : 6} className="latency-marker baseline-marker" />
                <circle cx={deployedX} cy={y} r={isHighlighted ? 9 : 7} className="latency-marker deployed-marker" />

                {isHighlighted ? (
                  <g className="performance-annotation">
                    <text
                      x={(deployedX + baselineX) / 2}
                      y={y - 13}
                      textAnchor="middle"
                      className="performance-annotation__speedup"
                    >
                      S{point.id} · {speedupLabel(point.speedup)}
                    </text>
                    <text
                      x={deployedX - 13}
                      y={y + 5}
                      textAnchor="end"
                      className="performance-annotation__latency"
                    >
                      {latencyLabel(point.deployedMs)}
                    </text>
                    <text
                      x={baselineX + 13}
                      y={y + 5}
                      textAnchor="start"
                      className="performance-annotation__latency"
                    >
                      {latencyLabel(point.baselineMs)}
                    </text>
                  </g>
                ) : null}
              </g>
            )
          })}
        </g>

        <g className="performance-axis">
          <line x1={PLOT_LEFT} x2={PLOT_RIGHT} y1={PLOT_BOTTOM + 18} y2={PLOT_BOTTOM + 18} className="axis-line" />
          <text
            x={(PLOT_LEFT + PLOT_RIGHT) / 2}
            y={PLOT_BOTTOM + 86}
            textAnchor="middle"
            className="axis-title"
          >
            MEDIAN LATENCY · ms · LOG SCALE
          </text>
        </g>

        <g className="performance-formulas">
          <text x={PLOT_LEFT} y="936" className="performance-formula performance-formula--speedup">
            {equations.speedup}
          </text>
          <text x={PLOT_RIGHT} y="936" textAnchor="end" className="performance-formula performance-formula--geomean">
            {equations.geomean}
          </text>
        </g>
      </svg>
    </figure>
  )
}
