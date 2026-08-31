import { FlowProgress } from './components/FlowProgress'
import { relayLabels } from './data/narrativeData'
import { ArchitectureSection } from './sections/ArchitectureSection'
import { EvidenceSection } from './sections/EvidenceSection'
import { HeroSection } from './sections/HeroSection'
import { SearchSection } from './sections/SearchSection'
import { WorkloadSection } from './sections/WorkloadSection'
import { EvidenceTrace } from './visuals/EvidenceTrace'

export function App() {
  return (
    <main className="story-flow">
      <div className="engineering-grid" aria-hidden="true" />
      <FlowProgress />

      <HeroSection />
      <EvidenceTrace
        variant="outcome-workloads"
        label={relayLabels.outcomeWorkloads}
      />

      <WorkloadSection />
      <EvidenceTrace
        variant="workloads-architecture"
        label={relayLabels.workloadsArchitecture}
      />

      <ArchitectureSection />
      <EvidenceTrace
        variant="architecture-search"
        label={relayLabels.architectureSearch}
      />

      <SearchSection />
      <EvidenceTrace
        variant="search-evidence"
        label={relayLabels.searchEvidence}
      />

      <EvidenceSection />
    </main>
  )
}
