import { useRef } from 'react'

import { useNarrativeStep } from '../hooks/useNarrativeStep'
import type { WorkloadStep } from '../motion/motionTypes'
import '../styles/workload.css'
import { WorkloadFigure } from '../visuals/WorkloadFigure'

const workloadDimensions = [
  { symbol: 'B', name: 'BATCH', range: '1 — 10,000' },
  { symbol: 'S', name: 'SEQUENCE', range: '32 — 100,000' },
  { symbol: 'D', name: 'MODEL WIDTH', range: '32 — 1,024' },
  { symbol: 'H', name: 'HEAD COUNT', range: '1 — 16' },
  { symbol: 'F', name: 'FFN WIDTH', range: '32 — 1,024' },
] as const

const workloadFacts = [
  { label: 'OFFICIAL WORKLOADS', value: '14' },
  { label: 'RESIDENT SHAPES', value: '13' },
  { label: 'CORRECTNESS CHECK', value: 'SUPPLIED ELEMENTWISE COMPARATOR' },
] as const

export function WorkloadSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const activeStep = useNarrativeStep<WorkloadStep>(sectionRef, 'map')

  return (
    <section
      ref={sectionRef}
      id="workloads"
      className="narrative-section workload-flow"
      data-active-step={activeStep}
      aria-labelledby="workload-heading"
    >
      <div className="workload-flow__shell">
        <header className="workload-flow__header" data-step="map">
          <div className="workload-flow__heading">
            <p className="workload-flow__eyebrow">02 · PROBLEM SETTING</p>
            <h2 id="workload-heading">
              <span>Workload Geometry</span>
              <span>and Execution Regimes</span>
            </h2>
            <p className="workload-flow__lead">
              The resident atlas positions Shapes 01–13 by token volume and attention
              elements; point area encodes model width. Three representative Shapes expose
              the deployed runtime, attention, and FFN selections.
            </p>
          </div>

          <dl className="workload-flow__facts" aria-label="Workload study scope">
            {workloadFacts.map((fact) => (
              <div key={fact.label}>
                <dt>{fact.label}</dt>
                <dd>{fact.value}</dd>
              </div>
            ))}
          </dl>
        </header>

        <dl
          className="workload-flow__dimensions"
          aria-label="Official workload parameter ranges"
        >
          {workloadDimensions.map((dimension) => (
            <div key={dimension.symbol} className="workload-flow__dimension">
              <dt>
                <span>{dimension.symbol}</span>
                {dimension.name}
              </dt>
              <dd>{dimension.range}</dd>
            </div>
          ))}
        </dl>

        <div className="workload-flow__figure" data-step="specialize">
          <WorkloadFigure activeStep={activeStep} />
        </div>

        <div className="workload-flow__program-object" aria-label="Searchable program definition">
          <span>SEARCH UNIT</span>
          <code>ConfigSpec = ProgramConfig + ScheduleConfig</code>
          <p>
            ProgramConfig selects operators, layout, precision, and fusion;
            ScheduleConfig selects runtime, compile mode, tiling, and microbatch.
          </p>
        </div>
      </div>
    </section>
  )
}
