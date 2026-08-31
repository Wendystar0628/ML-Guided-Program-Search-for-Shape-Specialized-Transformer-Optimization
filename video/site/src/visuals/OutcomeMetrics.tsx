import { headline } from '../data/projectData'

export function OutcomeMetrics() {
  return (
    <aside className="outcome-metrics" aria-label="Measured project outcome">
      <div className="outcome-metric outcome-metric--speedup">
        <strong>{headline.geomean.toFixed(2)}×</strong>
        <span>EQUAL-SHAPE GEOMEAN</span>
      </div>

      <p className="outcome-measured-claim">
        <span>MEASURED PROGRAMS</span>
        <strong>— NOT HAND-PICKED POLICY LABELS</strong>
      </p>

      <div className="outcome-metric outcome-metric--correctness">
        <strong>
          {headline.residentPassed} / {headline.residentTotal}
        </strong>
        <span>RESIDENT SHAPES PASS</span>
      </div>
    </aside>
  )
}
