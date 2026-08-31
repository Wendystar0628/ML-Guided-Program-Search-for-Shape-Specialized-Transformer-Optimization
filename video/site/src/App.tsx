import { EvidenceScene } from './scenes/EvidenceScene'
import { OutcomeScene } from './scenes/OutcomeScene'
import { ProgramSearchScene } from './scenes/ProgramSearchScene'
import { WorkloadScene } from './scenes/WorkloadScene'
import { EvidenceTrace } from './visuals/EvidenceTrace'

const sections = [
  { id: 'outcome', index: '01', label: 'OUTCOME' },
  { id: 'workloads', index: '02', label: 'WORKLOADS' },
  { id: 'search', index: '03', label: 'SEARCH' },
  { id: 'evidence', index: '04', label: 'EVIDENCE' },
] as const

export function App() {
  return (
    <main className="research-exhibit">
      <div className="engineering-grid" aria-hidden="true" />
      <EvidenceTrace />
      <nav className="section-rail" aria-label="Exhibit sections">
        {sections.map((section) => (
          <a key={section.id} href={'#' + section.id}>
            <span>{section.index}</span>
            {section.label}
          </a>
        ))}
      </nav>
      <OutcomeScene />
      <WorkloadScene />
      <ProgramSearchScene />
      <EvidenceScene />
    </main>
  )
}
