import { useRef } from 'react'

import { useNarrativeStep } from '../hooks/useNarrativeStep'
import { WorkloadFigure } from '../visuals/WorkloadFigure'

const workloadRanges = [
  { symbol: 'B', name: 'BATCH', range: '1 — 10,000' },
  { symbol: 'S', name: 'SEQUENCE', range: '32 — 100,000' },
  { symbol: 'D', name: 'WIDTH', range: '32 — 1,024' },
  { symbol: 'H', name: 'HEADS', range: '1 — 16' },
  { symbol: 'F', name: 'FFN', range: '32 — 1,024' },
] as const

export function WorkloadSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const activeStep = useNarrativeStep(sectionRef, 'map')

  return (
    <section
      ref={sectionRef}
      id="workloads"
      className="narrative-section workload-section"
      data-active-step={activeStep}
      aria-labelledby="workload-heading"
    >
      <div className="workload-atlas">
        <span
          className="workload-step-marker workload-step-marker--map"
          data-step="map"
          aria-hidden="true"
        />

        <div className="workload-atlas__heading">
          <p className="workload-atlas__index">02 · WORKLOAD SPECIALIZATION</p>
          <h2 id="workload-heading">
            ONE MODEL.
            <span>MANY GPU REGIMES.</span>
          </h2>
          <p className="workload-atlas__lead">
            Fourteen official workloads preserve one pre-normalized Transformer
            meaning while shifting launch overhead, matrix throughput, memory traffic,
            and device capacity.
          </p>
        </div>

        <dl className="workload-ranges" aria-label="Official workload parameter ranges">
          {workloadRanges.map((dimension) => (
            <div key={dimension.symbol} className="workload-range">
              <dt>
                <span>{dimension.symbol}</span>
                {dimension.name}
              </dt>
              <dd>{dimension.range}</dd>
            </div>
          ))}
        </dl>

        <p className="workload-atlas__resident-count">
          13 RESIDENT SHAPES · ONE SUPPLIED COMPARATOR PER SHAPE
        </p>

        <div className="workload-atlas__visual">
          <span
            className="workload-step-marker workload-step-marker--specialize"
            data-step="specialize"
            aria-hidden="true"
          />
          <WorkloadFigure activeStep={activeStep} />
        </div>

        <div className="workload-atlas__thesis">
          <p>SHAPE CHANGES THE BOTTLENECK.</p>
          <strong>THE PROGRAM MUST CHANGE WITH IT.</strong>
          <span>
            Inspect any Shape with pointer or keyboard focus. S02, S08, and S13 expose
            three different measured execution routes.
          </span>
        </div>

        <div className="workload-config-ribbon" aria-label="Program configuration handoff">
          <span>SEARCHABLE PROGRAM OBJECT</span>
          <strong>ConfigSpec = ProgramConfig + ScheduleConfig</strong>
          <p>
            ATTENTION · PROJECTIONS · FFN · NORMS · LAYOUT · PRECISION · FUSION ·
            RUNTIME · SCHEDULE
          </p>
        </div>
      </div>
    </section>
  )
}
