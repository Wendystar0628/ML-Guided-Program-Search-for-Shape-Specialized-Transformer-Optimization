import type { CSSProperties } from 'react'
import { equations, measurementProtocol, performance } from '../data/projectData'
import { EvidenceAggregate } from './EvidenceAggregate'

const LOG_MIN = Math.log10(0.05)
const LOG_MAX = Math.log10(1000)
const axisTicks = [0.05, 0.1, 1, 10, 100, 1000]
const highlightedShapes = new Set(['08', '11', '13'])

type PositionedStyle = CSSProperties & {
  '--baseline-position'?: string
  '--deployed-position'?: string
  '--tick-position'?: string
}

function latencyPosition(value: number) {
  const ratio = (Math.log10(value) - LOG_MIN) / (LOG_MAX - LOG_MIN)
  return `${Math.max(0, Math.min(1, ratio)) * 100}%`
}

function latencyLabel(value: number) {
  return value >= 100 ? value.toFixed(3) : value.toFixed(4)
}

function speedupLabel(value: number) {
  return `${value.toFixed(2)}×`
}

function technicalSentence(value: string) {
  const sentence = value.toLowerCase().replace(/\b(fp32|cuda|gpu|rtx)\b/g, (term) => term.toUpperCase())
  return sentence.charAt(0).toUpperCase() + sentence.slice(1)
}

export function PerformanceFigure() {
  return (
    <figure
      className="performance-figure evidence-ledger"
      aria-labelledby="evidence-title"
      aria-describedby="evidence-method evidence-boundary"
    >
      <header className="evidence-ledger__header">
        <div className="evidence-ledger__title-lockup">
          <span className="evidence-ledger__kicker">05 · EVALUATION RESULTS</span>
          <h2 id="evidence-title">Measured deployment performance</h2>
          <p className="evidence-ledger__subtitle">
            Baseline and deployed median latency for Shapes 01–13 on the validated NVIDIA GeForce RTX 4080 system.
          </p>
        </div>

        <div className="evidence-ledger__legend" aria-label="Latency markers">
          <span>
            <i className="evidence-ledger__legend-marker evidence-ledger__legend-marker--baseline" aria-hidden="true" />
            BASELINE
          </span>
          <span>
            <i className="evidence-ledger__legend-marker evidence-ledger__legend-marker--deployed" aria-hidden="true" />
            DEPLOYED
          </span>
        </div>
      </header>

      <section className="measurement-contract" id="evidence-method" aria-labelledby="measurement-contract-title">
        <div className="measurement-contract__heading">
          <span>MEASUREMENT PROTOCOL</span>
          <h3 id="measurement-contract-title">Controlled resident-shape timing</h3>
        </div>

        <dl className="measurement-contract__grid">
          <div>
            <dt>PRECISION</dt>
            <dd>{technicalSentence(measurementProtocol.precision)}</dd>
          </div>
          <div>
            <dt>TIMING</dt>
            <dd>Median latency from CUDA Events</dd>
          </div>
          <div>
            <dt>CORRECTNESS</dt>
            <dd>{technicalSentence(measurementProtocol.comparator)}</dd>
          </div>
          <div>
            <dt>ISOLATION</dt>
            <dd>{technicalSentence(measurementProtocol.isolation.join(' · '))}</dd>
          </div>
          <div className="measurement-contract__preset">
            <dt>STANDARD RESIDENT PRESET</dt>
            <dd>{technicalSentence(measurementProtocol.residentPreset)}</dd>
          </div>
        </dl>

        <p className="measurement-contract__boundary" id="evidence-boundary">
          S06 · 1 CORRECTNESS · 2 WARMUPS · 5 REPEATS · 3 ROUNDS
        </p>
      </section>

      <header className="evidence-ledger__results-heading">
        <span>PER-SHAPE RESULTS</span>
        <p>Median latency comparison · logarithmic scale</p>
      </header>

      <div className="evidence-ledger__table" role="table" aria-label="Resident baseline and deployed latency">
        <div className="evidence-ledger__head" role="row">
          <span className="evidence-ledger__head-shape" role="columnheader">SHAPE</span>
          <span className="evidence-ledger__head-baseline" role="columnheader">BASELINE</span>
          <span className="evidence-ledger__head-rail" role="columnheader">LOG LATENCY · ms</span>
          <span className="evidence-ledger__head-deployed" role="columnheader">DEPLOYED</span>
          <span className="evidence-ledger__head-speedup" role="columnheader">SPEEDUP</span>
        </div>

        <div className="evidence-ledger__axis" aria-hidden="true">
          <span className="evidence-ledger__axis-rail">
            {axisTicks.map((tick) => (
              <i
                key={tick}
                className="evidence-ledger__axis-tick"
                style={{ '--tick-position': latencyPosition(tick) } as PositionedStyle}
              >
                {tick}
              </i>
            ))}
          </span>
        </div>

        <div className="evidence-ledger__rows" role="rowgroup">
          {performance.map((point) => {
            const style: PositionedStyle = {
              '--baseline-position': latencyPosition(point.baselineMs),
              '--deployed-position': latencyPosition(point.deployedMs),
            }
            const rowClassName = highlightedShapes.has(point.id)
              ? 'performance-ledger-row performance-ledger-row--representative'
              : 'performance-ledger-row'

            return (
              <div
                key={point.id}
                className={rowClassName}
                role="row"
                aria-label={`Shape ${point.id}: baseline ${latencyLabel(point.baselineMs)} milliseconds; deployed ${latencyLabel(point.deployedMs)} milliseconds; ${speedupLabel(point.speedup)} speedup; pass.`}
              >
                <strong className="performance-ledger-row__shape" role="cell">S{point.id}</strong>
                <span className="performance-ledger-row__baseline" role="cell">
                  <b>{latencyLabel(point.baselineMs)}</b>
                  <small>ms</small>
                </span>

                <span className="performance-ledger-row__rail" style={style} role="cell" aria-hidden="true">
                  <span className="performance-ledger-row__grid" />
                  <span className="performance-ledger-row__connector" />
                  <span className="performance-ledger-row__marker performance-ledger-row__marker--baseline" />
                  <span className="performance-ledger-row__marker performance-ledger-row__marker--deployed" />
                </span>

                <span className="performance-ledger-row__deployed" role="cell">
                  <b>{latencyLabel(point.deployedMs)}</b>
                  <small>ms</small>
                </span>
                <strong className="performance-ledger-row__speedup" role="cell">
                  {speedupLabel(point.speedup)}
                </strong>
              </div>
            )
          })}
        </div>
      </div>

      <EvidenceAggregate />

      <figcaption className="evidence-ledger__formulas">
        <code>{equations.speedup}</code>
        <code>{equations.geomean}</code>
        <small>EQUAL-WEIGHT GEOMETRIC MEAN ACROSS 13 RESIDENT SHAPES</small>
      </figcaption>
    </figure>
  )
}
