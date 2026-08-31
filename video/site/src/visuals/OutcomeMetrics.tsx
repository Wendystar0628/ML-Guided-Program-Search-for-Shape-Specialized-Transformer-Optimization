import { headline } from '../data/projectData'

export function OutcomeMetrics() {
  return (
    <aside className="hero-fluid__metrics" aria-label="Measured project outcome">
      <div className="hero-fluid__metric">
        <strong>{headline.geomean.toFixed(2)}×</strong>
        <span>EQUAL-SHAPE GEOMEAN SPEEDUP</span>
      </div>

      <div className="hero-fluid__metric">
        <strong>
          {headline.residentPassed}/{headline.residentTotal}
        </strong>
        <span>RESIDENT SHAPES PASS</span>
      </div>

      <p className="hero-fluid__claim">
        <span>SELECTION BASIS</span>
        <strong>{headline.resultClaim}</strong>
      </p>
    </aside>
  )
}
