import '../styles/evidence.css'
import { PerformanceFigure } from '../visuals/PerformanceFigure'

export function EvidenceSection() {
  return (
    <section id="evidence" className="narrative-section evidence-section" aria-labelledby="evidence-title">
      <PerformanceFigure />

      <footer className="evidence-result-note" aria-label="Statistical result summary">
        <div className="evidence-result-note__sample" aria-hidden="true">
          {Array.from({ length: 13 }, (_, index) => <i key={index} />)}
          <span />
        </div>

        <div className="evidence-result-note__statement">
          <span>RESULT INTERPRETATION</span>
          <strong>Deployed median latency was lower than the baseline for all 13 resident Shapes.</strong>
        </div>

        <div className="evidence-result-note__annotation">
          <span>AGGREGATION</span>
          <p>Equal weight per shape in log-speedup space.</p>
        </div>
      </footer>
    </section>
  )
}
