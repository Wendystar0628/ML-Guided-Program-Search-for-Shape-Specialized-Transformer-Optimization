import { useRef, useState } from 'react'
import { headline } from '../data/projectData'
import { useNarrativeProgress } from '../hooks/useNarrativeProgress'
import { PROGRAM_FIELD_RANGES } from '../motion/programFieldRanges'
import { GpuProgramField } from '../visuals/GpuProgramField'
import { OutcomeMetrics } from '../visuals/OutcomeMetrics'

const representativeCandidates = [
  { index: 1, id: 'C02', anatomy: 'PROGRAM CONFIG · SCHEDULE CONFIG' },
  { index: 3, id: 'C04', anatomy: 'TYPED OPERATORS · RUNTIME CHOICES' },
  { index: 5, id: 'C06', anatomy: 'LAYOUT · PRECISION · FUSION' },
] as const

function phaseFor(progress: number) {
  if (progress < PROGRAM_FIELD_RANGES.programSpace.end) return 'candidates'
  if (progress < PROGRAM_FIELD_RANGES.converge.end) return 'converging'
  if (progress < PROGRAM_FIELD_RANGES.measure.end) return 'measuring'
  if (progress < PROGRAM_FIELD_RANGES.winner.end) return 'winner'
  return 'handoff'
}

export function HeroSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const progress = useNarrativeProgress(sectionRef)
  const [inspectedCandidate, setInspectedCandidate] = useState<number | null>(null)
  const phase = phaseFor(progress)

  return (
    <section
      ref={sectionRef}
      id="outcome"
      className="hero-section narrative-section"
      aria-labelledby="hero-title"
      data-phase={phase}
    >
      <div className="hero-sticky">
        <header className="hero-headline">
          <p className="hero-kicker">TIKTOK TECHJAM 2026 · HARDWARE EFFICIENCY</p>
          <h1 id="hero-title">
            {headline.title.map((line) => (
              <span key={line}>{line}</span>
            ))}
          </h1>
          <p className="hero-definition">
            Shape-specialized Transformer execution on an NVIDIA GeForce RTX 4080.
          </p>
        </header>

        <div className="hero-program-stage" data-phase={phase}>
          <GpuProgramField
            progress={progress}
            inspectedCandidate={inspectedCandidate}
            onInspectCandidate={setInspectedCandidate}
          />

          <div className="hero-stage-label hero-stage-label--candidates">
            <span>PROGRAM CANDIDATES</span>
            <strong>SEARCH COMPLETE PROGRAMS</strong>
          </div>

          <div
            className="hero-candidate-inspector"
            aria-label="Representative program candidates"
          >
            {representativeCandidates.map((candidate) => {
              const active = inspectedCandidate === candidate.index

              return (
                <button
                  key={candidate.id}
                  type="button"
                  data-active={active || undefined}
                  onPointerEnter={() => setInspectedCandidate(candidate.index)}
                  onPointerLeave={() => setInspectedCandidate(null)}
                  onFocus={() => setInspectedCandidate(candidate.index)}
                  onBlur={() => setInspectedCandidate(null)}
                >
                  <span>{candidate.id} · COMPLETE CONFIG SPEC</span>
                  <strong>{candidate.anatomy}</strong>
                </button>
              )
            })}
          </div>

          <div className="hero-measurement-overlay" aria-label="GPU measurement channels">
            <span>MEASURED ON RTX 4080</span>
            <div className="hero-measurement-channels">
              <strong>ACCURACY</strong>
              <strong>PATH</strong>
              <strong>LATENCY</strong>
              <strong>MEMORY</strong>
            </div>
          </div>

          <div className="hero-winner-overlay">
            <span>ONE MEASURED WINNER PER SHAPE</span>
            <strong>COMPILED FORWARD · CAUSAL SDPA · TORCH FFN</strong>
            <small>MEASURED PROGRAM · RTX 4080 · SHAPE S08</small>
          </div>

          <span className="hero-handoff-anchor" aria-hidden="true" />
        </div>

        <OutcomeMetrics />

        <p className="hero-bridge-question">
          WHY DOES EACH SHAPE NEED A DIFFERENT PATH?
        </p>
      </div>
    </section>
  )
}
