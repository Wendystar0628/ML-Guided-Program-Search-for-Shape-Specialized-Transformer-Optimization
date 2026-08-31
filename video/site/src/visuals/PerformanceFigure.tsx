import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { equations, performance } from '../data/projectData'
import { EvidenceAggregate } from './EvidenceAggregate'

const LOG_MIN = Math.log10(0.05)
const LOG_MAX = Math.log10(1000)
const highlightedShapes = new Set(['02', '08', '13'])

type RailStyle = CSSProperties & {
  '--baseline-position': string
  '--deployed-position': string
}

function latencyPosition(value: number) {
  const ratio = (Math.log10(value) - LOG_MIN) / (LOG_MAX - LOG_MIN)
  return `${Math.max(0, Math.min(1, ratio)) * 100}%`
}

function latencyLabel(value: number) {
  if (value >= 100) return value.toFixed(3)
  return value.toFixed(4)
}

function speedupLabel(value: number) {
  return `${value.toFixed(2)}×`
}

export function PerformanceFigure() {
  const figureRef = useRef<HTMLElement>(null)
  const [isVisible, setIsVisible] = useState(false)
  const [activeShape, setActiveShape] = useState<string | null>(null)

  useEffect(() => {
    const figure = figureRef.current
    if (!figure) return

    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.18 },
    )

    observer.observe(figure)
    return () => observer.disconnect()
  }, [])

  return (
    <figure
      ref={figureRef}
      className="performance-figure evidence-ledger"
      data-reveal={isVisible ? 'settled' : 'waiting'}
      data-active-shape={activeShape ?? undefined}
      aria-labelledby="evidence-title"
      aria-describedby="evidence-method evidence-boundary"
    >
      <header className="evidence-ledger__header">
        <div className="evidence-ledger__title-lockup">
          <span className="evidence-ledger__kicker">RESIDENT LATENCY LEDGER · RTX 4080</span>
          <h2 id="evidence-title">MEASURED SHAPE BY SHAPE</h2>
          <p>Thirteen correct ratios form one equal-shape aggregate.</p>
        </div>

        <div className="evidence-ledger__comparator" aria-label="Comparison direction">
          <span>SAME SUPPLIED COMPARATOR</span>
          <b>BASELINE</b>
          <i aria-hidden="true">→</i>
          <b>DEPLOYED</b>
        </div>
      </header>

      <div className="measurement-contract" id="evidence-method">
        <div className="measurement-contract__primary">
          <span>MEASUREMENT CONTRACT</span>
          <strong>FIXED FP32 INPUTS · CUDA EVENT MEDIANS · ABSOLUTE-OR-RELATIVE CORRECTNESS</strong>
        </div>
        <div>
          <span>ISOLATION</span>
          <strong>EXCLUSIVE PROJECT GPU LEASE · FRESH PROCESS PER SHAPE</strong>
        </div>
        <div>
          <span>STANDARD RESIDENT PRESET</span>
          <strong>5 CORRECTNESS · 20 WARMUPS · 100 REPEATS · 3 ROUNDS</strong>
        </div>
        <p id="evidence-boundary">S06 uses a reduced protocol for its large batch; the standard preset is not claimed for every Shape.</p>
      </div>

      <div className="evidence-ledger__plot">
        <div className="evidence-ledger__axis" aria-hidden="true">
          <span>0.05</span>
          <span>0.1</span>
          <span>1</span>
          <span>10</span>
          <span>100</span>
          <span>1000 ms</span>
        </div>

        <div className="evidence-ledger__columns" aria-hidden="true">
          <span>SHAPE</span>
          <span>BASELINE</span>
          <span>LOG LATENCY · ms</span>
          <span>DEPLOYED</span>
          <span>SPEEDUP</span>
        </div>

        <div className="evidence-ledger__rows">
          {performance.map((point) => {
            const isHighlighted = highlightedShapes.has(point.id)
            const isActive = activeShape === point.id
            const isMuted = activeShape !== null && !isActive
            const style: RailStyle = {
              '--baseline-position': latencyPosition(point.baselineMs),
              '--deployed-position': latencyPosition(point.deployedMs),
            }

            return (
              <div
                key={point.id}
                className={`performance-ledger-row${isHighlighted ? ' performance-ledger-row--representative' : ''}`}
                data-active={isActive ? 'true' : undefined}
                data-muted={isMuted ? 'true' : undefined}
                tabIndex={0}
                role="group"
                aria-label={`Shape ${point.id}: baseline ${latencyLabel(point.baselineMs)} milliseconds, deployed ${latencyLabel(point.deployedMs)} milliseconds, speedup ${speedupLabel(point.speedup)}. Correct by the supplied comparator.`}
                onPointerEnter={() => setActiveShape(point.id)}
                onPointerLeave={() => setActiveShape(null)}
                onFocus={() => setActiveShape(point.id)}
                onBlur={() => setActiveShape(null)}
              >
                <span className="performance-ledger-row__shape">S{point.id}</span>
                <span className="performance-ledger-row__baseline">
                  {latencyLabel(point.baselineMs)} <small>ms</small>
                </span>

                <span className="performance-ledger-row__rail" style={style} aria-hidden="true">
                  <span className="performance-ledger-row__grid" />
                  <span className="performance-ledger-row__connector" />
                  <span className="performance-ledger-row__marker performance-ledger-row__marker--baseline" />
                  <span className="performance-ledger-row__marker performance-ledger-row__marker--deployed" />
                </span>

                <span className="performance-ledger-row__deployed">
                  {latencyLabel(point.deployedMs)} <small>ms</small>
                </span>
                <strong className="performance-ledger-row__speedup">{speedupLabel(point.speedup)}</strong>
                <span className="performance-ledger-row__inspection" aria-hidden={isActive ? undefined : true}>
                  BASELINE {latencyLabel(point.baselineMs)} ms · DEPLOYED {latencyLabel(point.deployedMs)} ms · SPEEDUP{' '}
                  {speedupLabel(point.speedup)} · PASS
                </span>
              </div>
            )
          })}
        </div>

        <span className="evidence-ledger__sweep" aria-hidden="true" />
        <EvidenceAggregate />
      </div>

      <figcaption className="evidence-ledger__formulas">
        <span>{equations.speedup}</span>
        <span>{equations.geomean}</span>
        <small>Equal weight per resident Shape in log-speedup space.</small>
      </figcaption>
    </figure>
  )
}
