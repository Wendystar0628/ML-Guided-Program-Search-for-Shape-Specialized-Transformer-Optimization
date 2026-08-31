import { useRef } from 'react'
import { headline } from '../data/projectData'
import { useNarrativeProgress } from '../hooks/useNarrativeProgress'
import { PROGRAM_FIELD_RANGES } from '../motion/programFieldRanges'
import { GpuProgramField } from '../visuals/GpuProgramField'
import { OutcomeMetrics } from '../visuals/OutcomeMetrics'
import '../styles/hero.css'

const phaseLabels = {
  candidates: 'CANDIDATE CONFIGURATIONS',
  converging: 'TPE-GUIDED REFINEMENT',
  measuring: 'GPU MEASUREMENT',
  winner: 'PAIRED COMPARISON',
  handoff: 'EXACT-DEVICE REGISTRY UPDATE',
} as const

type HeroPhase = keyof typeof phaseLabels

function phaseFor(progress: number): HeroPhase {
  if (progress < PROGRAM_FIELD_RANGES.programSpace.end) return 'candidates'
  if (progress < PROGRAM_FIELD_RANGES.converge.end) return 'converging'
  if (progress < PROGRAM_FIELD_RANGES.measure.end) return 'measuring'
  if (progress < PROGRAM_FIELD_RANGES.winner.end) return 'winner'
  return 'handoff'
}

export function HeroSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const progress = useNarrativeProgress(sectionRef)
  const phase = phaseFor(progress)

  return (
    <section
      ref={sectionRef}
      id="outcome"
      className="hero-section hero-section--fluid narrative-section"
      aria-labelledby="hero-title"
    >
      <div className="hero-fluid">
        <header className="hero-fluid__headline">
          <p className="hero-fluid__kicker">{headline.kicker}</p>

          <h1 id="hero-title">
            {headline.title.map((line) => (
              <span key={line}>{line}</span>
            ))}
          </h1>

          <section className="hero-fluid__abstract" aria-label="Research scope">
            <div className="hero-fluid__abstract-copy">
              <span>RESEARCH SCOPE</span>
              <p>{headline.subject}</p>
            </div>

            <dl className="hero-fluid__metadata">
              <div>
                <dt>SEARCH OBJECT</dt>
                <dd>Complete execution program</dd>
              </div>
              <div>
                <dt>HARDWARE</dt>
                <dd>NVIDIA RTX 4080</dd>
              </div>
              <div>
                <dt>EVALUATION</dt>
                <dd>SHAPES 01–13</dd>
              </div>
            </dl>
          </section>

          <div className="hero-fluid__calibration" aria-hidden="true">
            <span>TYPED PROGRAM SPACE</span>
            <i />
            <span>GPU-MEASURED EXECUTION</span>
          </div>
        </header>

        <OutcomeMetrics />

        <figure className="hero-fluid__visual">
          <figcaption className="hero-fluid__visual-caption">
            <span>CANDIDATE PROGRAM SPACE</span>
            <strong>{phaseLabels[phase]}</strong>
          </figcaption>

          <div className="hero-fluid__canvas-frame">
            <GpuProgramField progress={progress} />
          </div>

          <dl className="hero-fluid__program" aria-label="Measured Shape 08 program">
            <div>
              <dt>MEASURED PROGRAM · S08</dt>
              <dd>COMPILED RUNTIME</dd>
            </div>
            <div>
              <dt>ATTENTION</dt>
              <dd>CAUSAL SDPA</dd>
            </div>
            <div>
              <dt>FFN</dt>
              <dd>TORCH</dd>
            </div>
          </dl>
        </figure>

        <p className="hero-fluid__bridge">
          <span>OUTCOME → WORKLOADS</span>
          Workload geometry defines the execution regime to be optimized.
        </p>
      </div>
    </section>
  )
}
