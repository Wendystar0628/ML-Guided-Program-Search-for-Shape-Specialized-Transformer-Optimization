import { useRef } from 'react'
import { equations, promotionThresholds, searchEvidence } from '../data/projectData'
import { useNarrativeStep } from '../hooks/useNarrativeStep'
import type { SearchStep } from '../motion/motionTypes'
import { ProgramSearchFigure } from '../visuals/ProgramSearchFigure'

const stageAnnotations: ReadonlyArray<{
  step: SearchStep
  eyebrow: string
  detail: string
  left: string
  top: string
  sensorTop: string
}> = [
  {
    step: 'reject',
    eyebrow: 'REJECT',
    detail: 'Invalid plans stop before GPU work.',
    left: '22%',
    top: '47%',
    sensorTop: '14%',
  },
  {
    step: 'screen',
    eyebrow: 'SCREEN',
    detail: 'Only Screen observations train the constraint-aware TPE.',
    left: '38%',
    top: '8%',
    sensorTop: '30%',
  },
  {
    step: 'enhanced',
    eyebrow: 'ENHANCED',
    detail: 'Remeasure the feasible frontier.',
    left: '54%',
    top: '74%',
    sensorTop: '46%',
  },
  {
    step: 'formal',
    eyebrow: 'FORMAL',
    detail: 'Lock one challenger against the incumbent.',
    left: '68%',
    top: '9%',
    sensorTop: '62%',
  },
  {
    step: 'registry',
    eyebrow: 'REGISTRY',
    detail: 'Commit the applicable Formal result as a complete ConfigSpec.',
    left: '82%',
    top: '57%',
    sensorTop: '78%',
  },
]

export function SearchSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const activeStep = useNarrativeStep<SearchStep>(sectionRef, 'sample')

  return (
    <section
      ref={sectionRef}
      id="search"
      className="narrative-section search-corridor"
      data-state={activeStep}
      aria-labelledby="search-heading"
    >
      <header className="search-corridor__origin" data-step="sample">
        <p className="search-corridor__handoff">EXECUTIONPLAN → SCREEN OBSERVATIONS</p>
        <h2 id="search-heading">SEARCH THE WHOLE PROGRAM</h2>
        <p className="search-corridor__lead">
          Each compatible structure owns a persistent, constraint-aware TPE study. Learning ranks
          the next executable program; GPU evidence decides whether it survives.
        </p>
      </header>

      <div className="search-corridor__method" aria-label="Conditional search method">
        <div className="search-method__equations">
          <span>CONDITIONAL TPE</span>
          <strong>ℓ(x) / g(x)</strong>
          <small>Only active schedule parameters enter each branch.</small>
        </div>
        <dl className="search-method__budget">
          <div>
            <dt>STRUCTURE BRANCHES</dt>
            <dd>≤ 36</dd>
          </div>
          <div>
            <dt>STARTUP</dt>
            <dd>min(10, |Xbranch|)</dd>
          </div>
          <div>
            <dt>EXPLORATION RESERVE</dt>
            <dd>≈ 10%</dd>
          </div>
        </dl>
      </div>

      <div className="search-corridor__visual">
        <ProgramSearchFigure
          activeStep={activeStep}
          evidence={searchEvidence}
          thresholds={promotionThresholds}
          tpeEquation={equations.tpe}
        />

        <div className="search-corridor__annotations" aria-label="Search stage explanations">
          {stageAnnotations.map((annotation) => (
            <div key={annotation.step}>
              <span
                className="search-corridor__step-sensor"
                data-step={annotation.step}
                aria-hidden="true"
                style={{
                  position: 'absolute',
                  top: annotation.sensorTop,
                  left: 0,
                  width: 1,
                  height: 1,
                }}
              />
              <div
                className={`search-corridor__annotation search-corridor__annotation--${annotation.step}`}
                style={{ left: annotation.left, top: annotation.top }}
              >
                <span>{annotation.eyebrow}</span>
                <p>{annotation.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="search-corridor__formal-contract">
        <p className="search-formal__claim">ONLY PAIRED FORMAL EVIDENCE CAN REPLACE AN INCUMBENT</p>
        <div className="search-formal__thresholds" aria-label="Formal promotion thresholds">
          {promotionThresholds.map((threshold) => (
            <span key={threshold.blocks}>
              <strong>{threshold.blocks}</strong>
              {threshold.ratio}
            </span>
          ))}
        </div>
        <p className="search-formal__bound">
          A first deployment requires a feasible Formal result. Paired blocks alternate incumbent
          and challenger order · per-comparison false-promotion bound ≤ 0.0464 under the stated null.
        </p>
      </div>

      <footer className="search-corridor__output">
        <span>6 DEPLOYMENT UPDATES</span>
        <p>
          The registry stores an approved ConfigSpec for the measured stack identity × Shape
          variant; runtime resolution sends that ConfigSpec through PlanBuilder again.
        </p>
      </footer>
    </section>
  )
}
