import { shape14Note } from '../data/projectData'
import { PerformanceFigure } from '../visuals/PerformanceFigure'

export function EvidenceScene() {
  return (
    <section id="evidence" className="scene evidence-scene" aria-labelledby="evidence-title">
      <header className="scene-heading evidence-scene__heading">
        <div className="evidence-scene__meta">
          <span className="scene-index">04</span>
          <span className="shape14-note" role="note">
            {shape14Note}
          </span>
        </div>
        <h2 id="evidence-title">MEASURED SHAPE BY SHAPE</h2>
      </header>

      <PerformanceFigure />
    </section>
  )
}
