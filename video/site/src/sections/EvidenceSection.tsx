import { PerformanceFigure } from '../visuals/PerformanceFigure'

export function EvidenceSection() {
  return (
    <section id="evidence" className="narrative-section evidence-section" aria-labelledby="evidence-title">
      <PerformanceFigure />

      <footer className="evidence-closing" aria-label="Project conclusion and evidence boundaries">
        <span className="evidence-closing__feedback" aria-hidden="true" />

        <div className="evidence-closing__claim">
          <p className="evidence-closing__kicker">MEASURE, THEN DEPLOY</p>
          <p className="evidence-closing__statement">
            Shape-specialized search turns workload and hardware differences into auditable executable choices without relaxing correctness.
          </p>
          <p className="evidence-closing__result">14.49× EQUAL-SHAPE GEOMEAN · 13 / 13 PASS</p>
        </div>

        <div className="evidence-closing__method">
          <span>REAL GPU MEASUREMENT</span>
          <span>REPRODUCIBLE PROGRAM IDENTITY</span>
          <span>EXACT-DEVICE DEPLOYMENT</span>
        </div>

        <div className="evidence-closing__boundaries">
          <span>MEASURED ON ONE NVIDIA GEFORCE RTX 4080</span>
          <span>≈ 12 HOURS OF CUMULATIVE SEARCH</span>
          <span>BEST FOUND · NOT A GLOBAL OPTIMUM</span>
          <span>LOCAL ENGINEERING RESULT · NOT AN OFFICIAL SCORE</span>
        </div>

        <p className="evidence-closing__chain" aria-label="Shape to program to evidence to deployment">
          <span>SHAPE</span>
          <i aria-hidden="true">→</i>
          <span>PROGRAM</span>
          <i aria-hidden="true">→</i>
          <span>EVIDENCE</span>
          <i aria-hidden="true">→</i>
          <span>DEPLOYMENT</span>
        </p>
      </footer>
    </section>
  )
}
