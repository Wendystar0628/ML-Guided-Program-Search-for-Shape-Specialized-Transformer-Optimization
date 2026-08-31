import { useState } from 'react'
import type { SearchStep } from '../motion/motionTypes'
import '../styles/search.css'

type SearchEvidenceItem = {
  label: string
  value: number
}

type PromotionThreshold = {
  blocks: string
  ratio: string
}

type ProgramSearchFigureProps = {
  activeStep?: SearchStep
  evidence: readonly SearchEvidenceItem[]
  thresholds: readonly PromotionThreshold[]
  tpeEquation: string
}

type SearchStage = {
  id: SearchStep
  index: string
  title: string
  value: string
  unit: string
  description: string
  tokens: number
}

const stageRank: Record<SearchStep, number> = {
  sample: 0,
  reject: 1,
  screen: 2,
  enhanced: 3,
  formal: 4,
  registry: 5,
}

const edgeLabels = ['Construct', 'Validate', 'Select frontier', 'Lock challenger', 'Promote'] as const
const rejectedMarks = [1, 4, 3, 3, 2] as const

function evidenceValue(
  evidence: readonly SearchEvidenceItem[],
  label: string,
): string {
  return (evidence.find((item) => item.label === label)?.value ?? 0).toLocaleString('en-US')
}

export function ProgramSearchFigure({
  activeStep = 'registry',
  evidence,
  thresholds,
  tpeEquation,
}: ProgramSearchFigureProps) {
  const [inspectedStage, setInspectedStage] = useState<SearchStep | null>(null)
  const selectedStage = inspectedStage ?? activeStep

  const stages: readonly SearchStage[] = [
    {
      id: 'sample',
      index: '01',
      title: 'Compatible structure branches',
      value: '≤ 36',
      unit: 'STRUCTURE BRANCHES',
      description: 'A sampled ConfigSpec contains one program structure and its active schedule and runtime parameters.',
      tokens: 24,
    },
    {
      id: 'reject',
      index: '02',
      title: 'Static legality',
      value: 'CPU',
      unit: 'NO GPU TIME',
      description: 'Static constraints reject incompatible shape, capability, layout, precision, fusion and runtime combinations before GPU execution.',
      tokens: 19,
    },
    {
      id: 'screen',
      index: '03',
      title: 'Screen',
      value: evidenceValue(evidence, 'SCREEN'),
      unit: 'ENTRIES · DEADLINE 65%',
      description: 'Screen measures legal candidates until the 65% cumulative soft deadline; only these observations train the conditional TPE.',
      tokens: 14,
    },
    {
      id: 'enhanced',
      index: '04',
      title: 'Enhanced',
      value: evidenceValue(evidence, 'ENHANCED'),
      unit: 'ENTRIES · DEADLINE 82%',
      description: 'Enhanced remeasures the selected frontier until the 82% cumulative soft deadline; these observations do not train TPE.',
      tokens: 8,
    },
    {
      id: 'formal',
      index: '05',
      title: 'Formal',
      value: evidenceValue(evidence, 'FORMAL'),
      unit: 'PAIRED COMPARISONS',
      description: 'Formal alternates incumbent and challenger within the remaining budget through the 100% soft deadline.',
      tokens: 3,
    },
    {
      id: 'registry',
      index: '06',
      title: 'Registry',
      value: evidenceValue(evidence, 'UPDATES'),
      unit: 'DEPLOYMENT UPDATES',
      description: 'The approved ConfigSpec is stored under measured-stack identity × Shape variant.',
      tokens: 1,
    },
  ]

  const selected = stages.find((stage) => stage.id === selectedStage) ?? stages[0]
  const selectedRank = stageRank[selectedStage]

  return (
    <figure
      className="search-system-flow"
      aria-labelledby="search-system-flow-title"
      data-active-step={selectedStage}
    >
      <div className="search-system-flow__controller">
        <div className="search-system-flow__formula">
          <span>Branch-local conditional TPE</span>
          <strong id="search-system-flow-title">TPE ranking signal · high {tpeEquation}</strong>
        </div>
        <div className="search-system-flow__proposal" aria-hidden="true">
          <span>Branch-local TPE</span>
          <b>→</b>
          <span className="search-system-flow__proposal-score">ℓ(x) / g(x)</span>
          <b>→</b>
          <span>Candidate ConfigSpec</span>
        </div>
        <div className="search-system-flow__learning-rule">
          <span>Learning signal</span>
          <p>Screen observations train the conditional TPE. Enhanced and Formal observations are excluded from model updates.</p>
        </div>
      </div>

      <div className="search-system-flow__branch-lattice" aria-hidden="true">
        <div className="search-branch-lattice__heading">
          <span>Compatible branch set</span>
          <strong>≤ 36 conditional domains</strong>
        </div>
        <div className="search-branch-lattice__cells">
          {Array.from({ length: 36 }, (_, index) => (
            <i
              key={index}
              data-state={index % 11 === 0 ? 'frontier' : index % 4 === 0 ? 'active' : 'available'}
            />
          ))}
        </div>
        <b className="search-branch-lattice__arrow">→</b>
        <div className="search-branch-lattice__outlet">
          <span>Selected branch</span>
          <strong>Complete ConfigSpec</strong>
        </div>
      </div>

      <div className="search-system-flow__legend" aria-hidden="true">
        <span>Schematic candidate count</span>
        <i />
        <span>Orange markers denote filtered candidates.</span>
      </div>

      <ol className="search-system-flow__pipeline" aria-label="Program search and promotion pipeline">
        {stages.map((stage, index) => {
          const rank = stageRank[stage.id]
          const status = rank < selectedRank ? 'complete' : rank === selectedRank ? 'active' : 'ready'

          return (
            <li className="search-system-flow__pair" key={stage.id}>
              <button
                type="button"
                className="search-system-stage"
                data-status={status}
                data-step={stage.id}
                aria-pressed={stage.id === selectedStage}
                onPointerEnter={() => setInspectedStage(stage.id)}
                onPointerLeave={() => setInspectedStage(null)}
                onFocus={() => setInspectedStage(stage.id)}
                onBlur={() => setInspectedStage(null)}
              >
                <span className="search-system-stage__index">{stage.index}</span>
                <span className="search-system-stage__title">{stage.title}</span>
                <span className="search-system-stage__population" aria-hidden="true">
                  {Array.from({ length: stage.tokens }, (_, tokenIndex) => (
                    <i key={tokenIndex} data-frontier={tokenIndex === stage.tokens - 1} />
                  ))}
                </span>
                <strong>{stage.value}</strong>
                <small>{stage.unit}</small>
              </button>

              {index < edgeLabels.length ? (
                <div className="search-system-edge" aria-hidden="true">
                  <span>{edgeLabels[index]}</span>
                  <b>→</b>
                  <span className="search-system-edge__rejects">
                    {Array.from({ length: rejectedMarks[index] }, (_, markIndex) => (
                      <i key={markIndex} />
                    ))}
                  </span>
                </div>
              ) : null}
            </li>
          )
        })}
      </ol>

      <div className="search-system-flow__detail" aria-live="polite">
        <span>{selected.index} · {selected.title}</span>
        <p>{selected.description}</p>
      </div>

      <div className="search-system-flow__formal">
        <div>
          <span>Formal promotion rule</span>
          <strong>Incumbent replacement requires paired Formal evidence.</strong>
        </div>
        <ul aria-label="Formal promotion thresholds">
          {thresholds.map((threshold) => (
            <li key={threshold.blocks}>
              <span>{threshold.blocks}</span>
              <strong>{threshold.ratio}</strong>
            </li>
          ))}
        </ul>
        <small>A first deployment requires a feasible Formal result. Per-comparison false-promotion bound ≤ 0.0464 under the stated null.</small>
      </div>

      <figcaption>
        <span>Search procedure</span>
        <p>Each compatible structure branch owns a persistent TPE study; fixed-budget allocation precedes Screen, Enhanced, Formal and registry update.</p>
      </figcaption>
    </figure>
  )
}
