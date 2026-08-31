import { headline } from '../data/projectData'
import { GpuProgramField } from '../visuals/GpuProgramField'

export function OutcomeScene() {
  return (
    <section
      id="outcome"
      className="scene scene-outcome"
      aria-labelledby="outcome-title"
    >
      <div className="outcome-layout">
        <header className="outcome-heading">
          <p className="instrument-label outcome-kicker">{headline.kicker}</p>
          <h1 id="outcome-title" className="outcome-title">
            {headline.title.map((line) => (
              <span key={line}>{line}</span>
            ))}
          </h1>
          <p className="outcome-subject">{headline.subject}</p>
        </header>

        <div className="outcome-stage">
          <div className="gpu-field-frame">
            <GpuProgramField />
          </div>

          <div className="outcome-stage-labels" aria-hidden="true">
            <span className="stage-label stage-label--candidates">CANDIDATES</span>
            <span className="stage-label stage-label--measured">GPU MEASUREMENT</span>
            <span className="stage-label stage-label--winner">WINNER</span>
          </div>
        </div>

        <div className="outcome-metrics" aria-label="Measured outcome">
          <div className="outcome-metric outcome-metric--speedup">
            <strong>{headline.geomean.toFixed(2)}×</strong>
            <span>GEOMEAN SPEEDUP</span>
          </div>
          <div className="outcome-metric outcome-metric--correctness">
            <strong>
              {headline.residentPassed} / {headline.residentTotal}
            </strong>
            <span>RESIDENT SHAPES PASS</span>
          </div>
        </div>
      </div>
    </section>
  )
}
