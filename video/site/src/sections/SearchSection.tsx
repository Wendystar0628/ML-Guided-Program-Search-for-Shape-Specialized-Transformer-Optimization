import { equations, promotionThresholds, searchEvidence } from '../data/projectData'
import { ProgramSearchFigure } from '../visuals/ProgramSearchFigure'

const searchBudget = [
  { label: 'STRUCTURE BRANCHES', value: '≤ 36', note: 'conditional parameter domains' },
  { label: 'STARTUP PER BRANCH', value: 'min(10, |Xbranch|)', note: 'branch-local initialization' },
  { label: 'EXPLORATION RESERVE', value: '≈ 10%', note: 'allocated to least-sampled branches' },
] as const

export function SearchSection() {
  return (
    <section
      id="search"
      className="narrative-section search-system"
      aria-labelledby="search-heading"
    >
      <div className="search-system__inner">
        <header className="search-system__header">
          <p>CONFIGSPEC SEARCH · STATIC CHECKS · GPU MEASUREMENT</p>
          <h2 id="search-heading">Conditional program and schedule search</h2>
          <div className="search-system__introduction">
            <div className="search-system__brief">
              <span>SEARCH OBJECT</span>
              <p>
                Each program structure defines a conditional parameter domain. The optimizer
                samples one complete ConfigSpec, applies static legality checks, then measures
                legal candidates at increasing fidelity.
              </p>
            </div>
            <div className="search-system__domain">
              <span>CONFIGURATION DOMAIN</span>
              <strong>ProgramConfig + active ScheduleConfig</strong>
            </div>
          </div>
        </header>

        <dl className="search-system__budget" aria-label="Conditional search budget">
          {searchBudget.map((item) => (
            <div key={item.label}>
              <dt>{item.label}</dt>
              <dd>{item.value}</dd>
              <small>{item.note}</small>
            </div>
          ))}
        </dl>

        <ProgramSearchFigure
          activeStep="registry"
          evidence={searchEvidence}
          thresholds={promotionThresholds}
          tpeEquation={equations.tpe}
        />

        <footer className="search-system__output">
          <strong>6 registry updates</strong>
          <p>
            Registry key: measured-stack identity × Shape variant. Stored value: approved
            ConfigSpec. Runtime resolution passes that ConfigSpec through PlanBuilder again.
          </p>
        </footer>
      </div>
    </section>
  )
}
