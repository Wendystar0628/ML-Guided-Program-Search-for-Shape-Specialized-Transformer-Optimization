type SearchEvidenceItem = {
  label: string
  value: number
}

type PromotionThreshold = {
  blocks: string
  ratio: string
}

type ProgramSearchFigureProps = {
  evidence: readonly SearchEvidenceItem[]
  thresholds: readonly PromotionThreshold[]
  tpeEquation: string
}

const candidatePaths = [
  { d: 'M76 224 C174 224 194 286 294 286 S430 252 608 252', legal: true },
  { d: 'M76 252 C170 252 210 320 302 320 S446 274 608 274', legal: false },
  { d: 'M76 280 C168 280 212 238 314 238 S456 296 608 296', legal: true },
  { d: 'M76 308 C182 308 212 356 328 356 S468 318 608 318', legal: false },
  { d: 'M76 336 C170 336 230 274 340 274 S472 340 608 340', legal: true },
  { d: 'M76 364 C180 364 234 392 346 392 S486 362 608 362', legal: true },
  { d: 'M76 392 C176 392 230 330 330 330 S478 384 608 384', legal: false },
  { d: 'M76 420 C176 420 236 438 348 438 S488 406 608 406', legal: true },
  { d: 'M76 448 C166 448 222 382 326 382 S468 428 608 428', legal: false },
  { d: 'M76 476 C180 476 238 472 350 472 S490 450 608 450', legal: true },
  { d: 'M76 504 C168 504 218 426 314 426 S458 472 608 472', legal: false },
  { d: 'M76 532 C174 532 226 512 342 512 S486 494 608 494', legal: true },
  { d: 'M76 560 C176 560 212 462 304 462 S448 516 608 516', legal: false },
] as const

const legalPaths = [
  'M608 252 C690 252 726 314 818 342',
  'M608 296 C692 296 732 334 818 358',
  'M608 340 C694 340 740 350 818 374',
  'M608 362 C696 362 746 374 818 390',
  'M608 406 C696 406 746 410 818 406',
  'M608 450 C692 450 738 434 818 422',
  'M608 494 C690 494 730 456 818 438',
] as const

const screenSurvivors = [
  'M818 350 C876 350 928 368 1008 382',
  'M818 374 C882 374 932 382 1008 394',
  'M818 406 C882 406 934 410 1008 406',
  'M818 438 C878 438 930 424 1008 418',
] as const

const screenRejects = [
  'M818 358 C840 358 854 360 878 364',
  'M818 390 C840 390 854 392 878 394',
  'M818 422 C840 422 854 420 878 416',
] as const

const enhancedSurvivors = [
  'M1008 382 C1072 382 1126 394 1218 400',
  'M1008 418 C1072 418 1126 412 1218 408',
] as const

const enhancedRejects = [
  'M1008 394 C1034 394 1050 396 1078 398',
  'M1008 406 C1034 406 1050 404 1078 402',
] as const

const stageTicks = [
  { x: 608, label: 'LEGAL?' },
  { x: 818, label: 'SCREEN' },
  { x: 1008, label: 'ENHANCED' },
  { x: 1218, label: 'FORMAL' },
] as const

const rejectionMarks = [274, 318, 384, 428, 472, 516] as const

const screenRejectionMarks = [364, 394, 416] as const

const enhancedRejectionMarks = [398, 402] as const

const evidenceUnits = ['ENTRIES', 'ENTRIES', 'COMPARISONS', 'UPDATES'] as const

export function ProgramSearchFigure({
  evidence,
  thresholds,
  tpeEquation,
}: ProgramSearchFigureProps) {
  return (
    <figure className="program-search-figure">
      <svg
        className="program-search-figure__svg"
        viewBox="0 0 1600 820"
        role="img"
        aria-labelledby="program-search-figure-title program-search-figure-description"
      >
        <title id="program-search-figure-title">Program search convergence</title>
        <desc id="program-search-figure-description">
          Candidate configurations are rejected by legality and progressively narrowed by screen,
          enhanced, and formal measurement before one program enters the registry.
        </desc>

        <g className="search-calibration" aria-hidden="true">
          <path className="search-calibration__rail" d="M76 184 H1518" />
          {Array.from({ length: 25 }, (_, index) => {
            const x = 76 + index * 60
            return (
              <line
                key={x}
                className="search-calibration__tick"
                x1={x}
                y1="176"
                x2={x}
                y2={index % 4 === 0 ? 194 : 188}
              />
            )
          })}
        </g>

        <g className="search-tpe" aria-label={`Conditional TPE score ${tpeEquation}`}>
          <text className="search-tpe__label" x="78" y="78">
            CONDITIONAL TPE
          </text>
          <text className="search-tpe__equation" x="78" y="124">
            {tpeEquation}
          </text>
          <path
            className="search-density search-density--l"
            d="M252 132 C282 132 286 84 322 84 C358 84 364 146 402 146"
          />
          <path
            className="search-density search-density--g"
            d="M252 146 C292 146 304 112 344 112 C378 112 384 132 402 132"
          />
          <line className="search-density__baseline" x1="252" y1="154" x2="402" y2="154" />
        </g>

        <g className="search-stage search-stage--input">
          <text className="search-stage__label" x="76" y="650">
            WORKLOAD
          </text>
          <path className="search-stage__rule" d="M76 622 H214" />
        </g>

        <g className="search-stage search-stage--config">
          <text className="search-stage__label" x="330" y="650" textAnchor="middle">
            CONFIG
          </text>
          <path className="search-stage__rule" d="M248 622 H412" />
          <circle className="search-config-node" cx="286" cy="622" r="5" />
          <circle className="search-config-node" cx="330" cy="622" r="5" />
          <circle className="search-config-node" cx="374" cy="622" r="5" />
        </g>

        <g className="search-stage-gates">
          {stageTicks.map((stage) => (
            <g key={stage.label} className={`search-gate search-gate--${stage.label.toLowerCase().replace('?', '')}`}>
              <line className="search-gate__line" x1={stage.x} y1="210" x2={stage.x} y2="566" />
              <line className="search-gate__cap" x1={stage.x - 11} y1="210" x2={stage.x + 11} y2="210" />
              <line className="search-gate__cap" x1={stage.x - 11} y1="566" x2={stage.x + 11} y2="566" />
              <text className="search-gate__label" x={stage.x} y="162" textAnchor="middle">
                {stage.label}
              </text>
            </g>
          ))}
        </g>

        <g className="search-paths search-paths--candidates" fill="none">
          {candidatePaths.map((path, index) => (
            <path
              key={path.d}
              className={`search-path search-path--candidate ${
                path.legal ? 'search-path--legal' : 'search-path--illegal'
              }`}
              d={path.d}
              pathLength="1"
              style={{ '--path-index': index } as React.CSSProperties}
            />
          ))}
        </g>

        <g className="search-paths search-paths--legal" fill="none">
          {legalPaths.map((path, index) => (
            <path
              key={path}
              className="search-path search-path--measured"
              d={path}
              pathLength="1"
              style={{ '--path-index': index } as React.CSSProperties}
            />
          ))}
        </g>

        <g className="search-paths search-paths--screen" fill="none">
          {screenSurvivors.map((path, index) => (
            <path
              key={path}
              className="search-path search-path--survivor"
              d={path}
              pathLength="1"
              style={{ '--path-index': index } as React.CSSProperties}
            />
          ))}
          {screenRejects.map((path) => (
            <path key={path} className="search-path search-path--rejected" d={path} />
          ))}
        </g>

        <g className="search-paths search-paths--enhanced" fill="none">
          {enhancedSurvivors.map((path, index) => (
            <path
              key={path}
              className="search-path search-path--survivor"
              d={path}
              pathLength="1"
              style={{ '--path-index': index } as React.CSSProperties}
            />
          ))}
          {enhancedRejects.map((path) => (
            <path key={path} className="search-path search-path--rejected" d={path} />
          ))}
        </g>

        <g className="search-paths search-paths--formal" fill="none">
          <path
            className="search-path search-path--formal-candidate"
            d="M1218 400 C1254 400 1276 402 1310 404"
          />
          <path
            className="search-path search-path--winner"
            d="M1218 408 C1302 408 1378 407 1462 407"
            pathLength="1"
          />
        </g>

        <g className="search-rejections" aria-hidden="true">
          {rejectionMarks.map((y) => (
            <path key={y} className="search-rejection-mark" d={`M596 ${y - 7} L620 ${y + 7}`} />
          ))}
          {screenRejectionMarks.map((y) => (
            <path key={y} className="search-rejection-mark" d={`M866 ${y - 7} L884 ${y + 7}`} />
          ))}
          {enhancedRejectionMarks.map((y, index) => (
            <path
              key={`${y}-${index}`}
              className="search-rejection-mark"
              d={`M1068 ${y - 7 - index * 5} L1086 ${y + 7 + index * 5}`}
            />
          ))}
          <path className="search-rejection-mark" d="M1300 394 L1318 414" />
        </g>

        <g className="search-funnel" fill="none" aria-hidden="true">
          <path className="search-funnel__edge" d="M608 224 C770 244 1048 328 1218 382" />
          <path className="search-funnel__edge" d="M608 540 C770 520 1048 448 1218 426" />
          <path className="search-funnel__terminal" d="M1218 382 L1262 404 L1218 426" />
        </g>

        <g className="search-evidence-labels">
          {evidence.map((item, index) => {
            const x = [818, 1008, 1218, 1462][index] ?? 1462
            return (
              <g key={item.label} className={`search-evidence search-evidence--${item.label.toLowerCase()}`}>
                <text className="search-evidence__value" x={x} y="104" textAnchor="middle">
                  {item.value.toLocaleString('en-US')}
                </text>
                <text className="search-evidence__unit" x={x} y="128" textAnchor="middle">
                  {evidenceUnits[index] ?? item.label}
                </text>
              </g>
            )
          })}
        </g>

        <g className="promotion-gate">
          <text className="promotion-gate__label" x="1030" y="640">
            PROMOTION GATE
          </text>
          {thresholds.map((threshold, index) => {
            const y = 680 + index * 42
            const gateY = 548 + index * 5
            return (
              <g key={threshold.blocks} className="promotion-threshold">
                <path
                  className="promotion-threshold__trace"
                  d={`M1218 ${gateY} C1172 ${gateY + 28} 1150 ${y} 1098 ${y} H1030`}
                />
                <text className="promotion-threshold__blocks" x="1030" y={y - 8}>
                  {threshold.blocks}
                </text>
                <text className="promotion-threshold__ratio" x="1120" y={y - 8}>
                  {threshold.ratio}
                </text>
              </g>
            )
          })}
        </g>

        <g className="search-registry" transform="translate(1462 407)">
          <circle className="search-registry__orbit search-registry__orbit--outer" r="70" />
          <circle className="search-registry__orbit search-registry__orbit--inner" r="42" />
          <path className="search-registry__bracket" d="M-86 -22 H-100 V22 H-86 M86 -22 H100 V22 H86" />
          <circle className="search-registry__core" r="10" />
          <text className="search-registry__label" x="0" y="112" textAnchor="middle">
            REGISTRY
          </text>
        </g>
      </svg>
    </figure>
  )
}
