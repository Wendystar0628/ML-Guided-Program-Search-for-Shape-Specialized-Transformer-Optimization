import { useState, type CSSProperties } from 'react'
import type { SearchStep } from '../motion/motionTypes'

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

type GateId = 'legality' | 'screen' | 'enhanced' | 'formal' | 'registry'

const stepRank: Record<SearchStep, number> = {
  sample: 0,
  reject: 1,
  screen: 2,
  enhanced: 3,
  formal: 4,
  registry: 5,
}

const candidatePaths = [
  { y: 278, gateY: 302, legal: true },
  { y: 302, gateY: 326, legal: false },
  { y: 326, gateY: 342, legal: true },
  { y: 350, gateY: 366, legal: false },
  { y: 374, gateY: 382, legal: true },
  { y: 398, gateY: 406, legal: true },
  { y: 422, gateY: 422, legal: false },
  { y: 446, gateY: 438, legal: true },
  { y: 470, gateY: 462, legal: false },
  { y: 494, gateY: 478, legal: true },
  { y: 518, gateY: 494, legal: false },
  { y: 542, gateY: 518, legal: true },
  { y: 566, gateY: 534, legal: false },
  { y: 590, gateY: 558, legal: true },
  { y: 614, gateY: 574, legal: false },
] as const

const screenPaths = [
  { startY: 302, endY: 360, survives: true },
  { startY: 342, endY: 384, survives: false },
  { startY: 382, endY: 408, survives: true },
  { startY: 406, endY: 432, survives: true },
  { startY: 438, endY: 456, survives: false },
  { startY: 478, endY: 480, survives: true },
  { startY: 518, endY: 504, survives: false },
  { startY: 558, endY: 528, survives: true },
] as const

const enhancedPaths = [
  { startY: 360, endY: 398, survives: false },
  { startY: 408, endY: 422, survives: true },
  { startY: 432, endY: 446, survives: false },
  { startY: 480, endY: 470, survives: true },
  { startY: 528, endY: 494, survives: false },
] as const

const formalPaths = [
  { startY: 422, endY: 438, wins: false },
  { startY: 470, endY: 454, wins: true },
] as const

const gates: ReadonlyArray<{
  id: GateId
  x: number
  label: string
  noteTitle: string
  noteLines: readonly string[]
  noteX: number
  noteY: number
}> = [
  {
    id: 'legality',
    x: 620,
    label: 'STATIC LEGALITY',
    noteTitle: 'REJECT BEFORE GPU WORK',
    noteLines: ['Shape · capability · layout', 'Precision · fusion · runtime'],
    noteX: 456,
    noteY: 664,
  },
  {
    id: 'screen',
    x: 980,
    label: 'SCREEN · 65%',
    noteTitle: 'SCREEN TRAINS TPE',
    noteLines: ['All COMPLETE observations', 'Accuracy · path · runtime constraints'],
    noteX: 826,
    noteY: 178,
  },
  {
    id: 'enhanced',
    x: 1320,
    label: 'ENHANCED · 82%',
    noteTitle: 'REMEASURE THE FRONTIER',
    noteLines: ['Fastest eligible 20%', 'Maximum eight resident candidates'],
    noteX: 1166,
    noteY: 664,
  },
  {
    id: 'formal',
    x: 1630,
    label: 'FORMAL · 100%',
    noteTitle: 'LOCK ONE CHALLENGER',
    noteLines: ['Alternating paired blocks', 'Sequential gate is preset'],
    noteX: 1476,
    noteY: 178,
  },
  {
    id: 'registry',
    x: 2040,
    label: 'REGISTRY',
    noteTitle: 'EXACT MATCH ONLY',
    noteLines: ['Stores the approved ConfigSpec', 'Measured stack identity + Shape variant'],
    noteX: 1702,
    noteY: 664,
  },
]

const evidenceUnits = ['SCREEN ENTRIES', 'ENHANCED ENTRIES', 'FORMAL COMPARISONS', 'DEPLOYMENT UPDATES'] as const

function pathStyle(index: number): CSSProperties {
  return { '--path-index': index } as CSSProperties
}

export function ProgramSearchFigure({
  activeStep = 'sample',
  evidence,
  thresholds,
  tpeEquation,
}: ProgramSearchFigureProps) {
  const [activeGate, setActiveGate] = useState<GateId | null>(null)
  const rank = stepRank[activeStep]
  const inspectedGate = gates.find((gate) => gate.id === activeGate)

  return (
    <figure className="program-search-figure" data-active-step={activeStep}>
      <svg
        className="program-search-figure__svg"
        viewBox="0 0 2200 860"
        role="img"
        aria-labelledby="program-search-figure-title program-search-figure-description"
      >
        <title id="program-search-figure-title">Conditional program search corridor</title>
        <desc id="program-search-figure-description">
          Typed execution programs are proposed inside compatible branches, filtered by static
          legality, narrowed through Screen, Enhanced and Formal GPU evidence, then committed to an
          measured-stack-and-Shape registry under the applicable Formal rule.
        </desc>

        <defs>
          <linearGradient id="search-corridor-wash" x1="0" x2="1">
            <stop offset="0" stopColor="#2457d6" stopOpacity="0.055" />
            <stop offset="0.74" stopColor="#2457d6" stopOpacity="0.015" />
            <stop offset="1" stopColor="#f06432" stopOpacity="0.075" />
          </linearGradient>
          <marker id="search-winner-arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
            <path d="M0 0 L12 6 L0 12 Z" className="search-winner-arrow" />
          </marker>
        </defs>

        <path
          className="search-corridor-field"
          d="M92 220 H620 L980 294 L1320 354 L1630 402 L2110 424 V488 L1630 506 L1320 566 L980 626 L620 666 H92 Z"
          fill="url(#search-corridor-wash)"
        />
        <path className="search-corridor-boundary" d="M92 220 H620 L980 294 L1320 354 L1630 402 L2110 424" />
        <path className="search-corridor-boundary" d="M92 666 H620 L980 626 L1320 566 L1630 506 L2110 488" />

        <g className="search-calibration" aria-hidden="true">
          <path className="search-calibration__rail" d="M94 202 H2110" />
          {Array.from({ length: 34 }, (_, index) => {
            const x = 94 + index * 61
            return (
              <line key={x} className="search-calibration__tick" x1={x} y1="190" x2={x} y2={index % 5 === 0 ? 210 : 202} />
            )
          })}
        </g>

        <g className="search-tpe" aria-label={`Conditional TPE score ${tpeEquation}`}>
          <text className="search-tpe__label" x="96" y="72">BRANCH-LOCAL CONDITIONAL TPE</text>
          <text className="search-tpe__equation" x="96" y="132">ℓ(x) = p(x | y &lt; y*)</text>
          <text className="search-tpe__equation search-tpe__equation--secondary" x="96" y="174">g(x) = p(x | y ≥ y*)</text>
          <text className="search-tpe__ratio" x="476" y="132">FAVOR HIGH {tpeEquation}</text>
          <path className="search-density search-density--l" d="M478 172 C526 172 530 110 578 110 C626 110 636 184 690 184" />
          <path className="search-density search-density--g" d="M478 184 C528 184 546 140 600 140 C646 140 660 166 690 166" />
          <line className="search-density__baseline" x1="478" y1="194" x2="690" y2="194" />
        </g>

        <g className="search-stage search-stage--origin">
          <text className="search-stage__label" x="96" y="256">EXECUTIONPLAN CANDIDATES</text>
          <text className="search-stage__purpose" x="96" y="696">FIXED-BUDGET SURVIVOR TPE</text>
        </g>

        <g className="search-paths search-paths--candidates" fill="none">
          {candidatePaths.map((path, index) => {
            const rejected = !path.legal && rank >= stepRank.reject
            return (
              <path
                key={`${path.y}-${path.gateY}`}
                className={`search-path search-path--candidate ${path.legal ? 'search-path--legal' : 'search-path--illegal'} ${rejected ? 'is-rejected' : 'is-revealed'}`}
                d={`M96 ${path.y} C248 ${path.y} 344 ${path.gateY} 620 ${path.gateY}`}
                pathLength="1"
                style={pathStyle(index)}
              />
            )
          })}
        </g>

        <g className={`search-paths search-paths--screen ${rank >= stepRank.screen ? 'is-revealed' : 'is-pending'}`} fill="none">
          {screenPaths.map((path, index) => (
            <path
              key={`${path.startY}-${path.endY}`}
              className={`search-path search-path--measured ${path.survives ? 'search-path--survivor' : 'search-path--rejected'}`}
              d={`M620 ${path.startY} C760 ${path.startY} 842 ${path.endY} 980 ${path.endY}`}
              pathLength="1"
              style={pathStyle(index)}
            />
          ))}
        </g>

        <g className={`search-paths search-paths--enhanced ${rank >= stepRank.enhanced ? 'is-revealed' : 'is-pending'}`} fill="none">
          {enhancedPaths.map((path, index) => (
            <path
              key={`${path.startY}-${path.endY}`}
              className={`search-path ${path.survives ? 'search-path--survivor' : 'search-path--rejected'}`}
              d={`M980 ${path.startY} C1092 ${path.startY} 1202 ${path.endY} 1320 ${path.endY}`}
              pathLength="1"
              style={pathStyle(index)}
            />
          ))}
        </g>

        <g className={`search-paths search-paths--formal ${rank >= stepRank.formal ? 'is-revealed' : 'is-pending'}`} fill="none">
          {formalPaths.map((path, index) => (
            <path
              key={`${path.startY}-${path.endY}`}
              className={`search-path ${path.wins ? 'search-path--formal-winner' : 'search-path--formal-candidate'}`}
              d={`M1320 ${path.startY} C1434 ${path.startY} 1518 ${path.endY} 1630 ${path.endY}`}
              pathLength="1"
              style={pathStyle(index)}
            />
          ))}
        </g>

        <g className={`search-paths search-paths--registry ${rank >= stepRank.registry ? 'is-revealed' : 'is-pending'}`} fill="none">
          <path className="search-path search-path--winner" d="M1630 454 C1764 454 1850 456 1994 456" pathLength="1" markerEnd="url(#search-winner-arrow)" />
        </g>

        <g className={`search-rejections ${rank >= stepRank.reject ? 'is-revealed' : 'is-pending'}`} aria-hidden="true">
          {candidatePaths.filter((path) => !path.legal).map((path) => (
            <path key={path.gateY} className="search-rejection-mark" d={`M604 ${path.gateY - 11} L636 ${path.gateY + 11}`} />
          ))}
          {screenPaths.filter((path) => !path.survives).map((path) => (
            <path key={path.endY} className="search-rejection-mark" d={`M964 ${path.endY - 10} L996 ${path.endY + 10}`} />
          ))}
          {enhancedPaths.filter((path) => !path.survives).map((path) => (
            <path key={path.endY} className="search-rejection-mark" d={`M1304 ${path.endY - 10} L1336 ${path.endY + 10}`} />
          ))}
          <path className="search-rejection-mark" d="M1614 426 L1646 450" />
        </g>

        <g className="search-stage-gates">
          {gates.map((gate) => {
            const isInspected = gate.id === activeGate
            return (
              <g
                key={gate.id}
                className={`search-gate search-gate--${gate.id} ${isInspected ? 'is-inspected' : ''}`}
                tabIndex={0}
                role="button"
                aria-label={`${gate.label}. ${gate.noteTitle}. ${gate.noteLines.join('. ')}`}
                onPointerEnter={() => setActiveGate(gate.id)}
                onPointerLeave={() => setActiveGate(null)}
                onFocus={() => setActiveGate(gate.id)}
                onBlur={() => setActiveGate(null)}
              >
                <line className="search-gate__line" x1={gate.x} y1="242" x2={gate.x} y2="642" />
                <line className="search-gate__cap" x1={gate.x - 15} y1="242" x2={gate.x + 15} y2="242" />
                <line className="search-gate__cap" x1={gate.x - 15} y1="642" x2={gate.x + 15} y2="642" />
                <circle className="search-gate__focus-target" cx={gate.x} cy="454" r="24" />
                <text className="search-gate__label" x={gate.x} y="228" textAnchor="middle">{gate.label}</text>
              </g>
            )
          })}
        </g>

        <g className="search-evidence-labels">
          {evidence.map((item, index) => {
            const positions = [
              { x: 980, valueY: 98, unitY: 138 },
              { x: 1320, valueY: 720, unitY: 760 },
              { x: 1630, valueY: 98, unitY: 138 },
              { x: 2040, valueY: 720, unitY: 760 },
            ]
            const position = positions[index] ?? positions[positions.length - 1]
            const requiredRank = index + stepRank.screen
            return (
              <g key={item.label} className={`search-evidence search-evidence--${item.label.toLowerCase()} ${rank >= requiredRank ? 'is-revealed' : 'is-pending'}`}>
                <text className="search-evidence__value" x={position.x} y={position.valueY} textAnchor="middle">{item.value.toLocaleString('en-US')}</text>
                <text className="search-evidence__unit" x={position.x} y={position.unitY} textAnchor="middle">{evidenceUnits[index] ?? item.label}</text>
              </g>
            )
          })}
        </g>

        <g className={`promotion-gate ${rank >= stepRank.formal ? 'is-revealed' : 'is-pending'}`}>
          <text className="promotion-gate__label" x="1424" y="590">PRESET PAIRED PROMOTION GATE</text>
          {thresholds.map((threshold, index) => {
            const x = 1424 + index * 202
            return (
              <g key={threshold.blocks} className="promotion-threshold">
                <path className="promotion-threshold__trace" d={`M${x} 610 H${x + 166}`} />
                <text className="promotion-threshold__blocks" x={x} y="640">{threshold.blocks}</text>
                <text className="promotion-threshold__ratio" x={x + 80} y="640">{threshold.ratio}</text>
              </g>
            )
          })}
          <text className="promotion-gate__bound" x="1424" y="684">PER-COMPARISON FALSE-PROMOTION BOUND ≤ 0.0464</text>
        </g>

        <g className="search-registry" transform="translate(2040 456)">
          <circle className="search-registry__orbit search-registry__orbit--outer" r="68" />
          <circle className="search-registry__orbit search-registry__orbit--inner" r="40" />
          <path className="search-registry__bracket" d="M-86 -24 H-104 V24 H-86 M86 -24 H104 V24 H86" />
          <circle className="search-registry__core" r="11" />
        </g>

        {inspectedGate ? (
          <g className="search-gate-note" transform={`translate(${inspectedGate.noteX} ${inspectedGate.noteY})`} aria-live="polite">
            <rect className="search-gate-note__surface" width="326" height="132" rx="4" />
            <path className="search-gate-note__rule" d="M0 0 H326" />
            <text className="search-gate-note__title" x="22" y="38">{inspectedGate.noteTitle}</text>
            {inspectedGate.noteLines.map((line, index) => (
              <text key={line} className="search-gate-note__line" x="22" y={78 + index * 30}>{line}</text>
            ))}
          </g>
        ) : null}
      </svg>
    </figure>
  )
}
