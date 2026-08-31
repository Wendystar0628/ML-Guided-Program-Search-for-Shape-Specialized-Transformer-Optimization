import {
  equations,
  promotionThresholds,
  searchEvidence,
} from '../data/projectData'
import { ProgramSearchFigure } from '../visuals/ProgramSearchFigure'

export function ProgramSearchScene() {
  return (
    <section
      id="search"
      className="scene scene--search program-search-scene"
      aria-labelledby="program-search-heading"
    >
      <div className="scene-shell program-search-scene__shell">
        <header className="scene-heading program-search-scene__heading">
          <span className="scene-index" aria-hidden="true">
            03
          </span>
          <h2 id="program-search-heading">SEARCH THE WHOLE PROGRAM</h2>
        </header>

        <ProgramSearchFigure
          evidence={searchEvidence}
          thresholds={promotionThresholds}
          tpeEquation={equations.tpe}
        />
      </div>
    </section>
  )
}
