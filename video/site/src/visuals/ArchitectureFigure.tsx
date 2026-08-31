import { useMemo } from 'react'
import {
  architectureEdges,
  architectureNodes,
  architectureSteps,
  evidenceChannels,
  evidenceStores,
  promotionThresholds,
  runtimeLanes,
  type ArchitectureNodeId,
} from '../data/architectureData'
import type { ArchitectureStep } from '../motion/motionTypes'

type ArchitectureFigureProps = {
  activeStep: ArchitectureStep
  activeNodeId: ArchitectureNodeId | null
  pinnedNodeId: ArchitectureNodeId | null
  onNodeEnter: (nodeId: ArchitectureNodeId) => void
  onNodeLeave: () => void
  onNodeToggle: (nodeId: ArchitectureNodeId) => void
}

function stageStatus(stage: ArchitectureStep, activeStep: ArchitectureStep) {
  const stageIndex = architectureSteps.indexOf(stage)
  const activeIndex = architectureSteps.indexOf(activeStep)

  if (stageIndex < activeIndex) return 'complete'
  if (stageIndex === activeIndex) return 'active'
  return 'future'
}

function nodeRelation(nodeId: ArchitectureNodeId, activeNodeId: ArchitectureNodeId | null) {
  if (!activeNodeId) return 'context'
  if (nodeId === activeNodeId) return 'selected'

  const connected = architectureEdges.some(
    (edge) =>
      (edge.from === activeNodeId && edge.to === nodeId) ||
      (edge.to === activeNodeId && edge.from === nodeId),
  )

  return connected ? 'related' : 'muted'
}

export function ArchitectureFigure({
  activeStep,
  activeNodeId,
  pinnedNodeId,
  onNodeEnter,
  onNodeLeave,
  onNodeToggle,
}: ArchitectureFigureProps) {
  const relatedEdgeIds = useMemo(() => {
    if (!activeNodeId) return new Set<string>()

    return new Set(
      architectureEdges
        .filter((edge) => edge.from === activeNodeId || edge.to === activeNodeId)
        .map((edge) => edge.id),
    )
  }, [activeNodeId])

  return (
    <figure
      className="architecture-figure"
      data-active-step={activeStep}
      aria-label="Closed evidence loop from workload identity through program construction, isolated measurement, formal promotion and runtime resolution"
    >
      <svg
        className="architecture-figure__connectors"
        viewBox="0 0 1920 920"
        role="img"
        aria-label="Architecture connectors and evidence feedback loop"
      >
        <defs>
          <marker
            id="architecture-arrow-blue"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" />
          </marker>
          <marker
            id="architecture-arrow-orange"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" />
          </marker>
          <filter id="architecture-packet-glow" x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <path
          className="architecture-figure__system-plane"
          d="M 340 470 C 500 180 1440 130 1690 250 C 1890 350 1860 650 1660 720 C 1300 850 570 855 340 510"
        />

        {architectureEdges.map((edge) => {
          const status = stageStatus(edge.stage, activeStep)
          const relation = activeNodeId
            ? relatedEdgeIds.has(edge.id)
              ? 'related'
              : 'muted'
            : 'context'

          return (
            <g
              key={edge.id}
              className="architecture-edge"
              data-kind={edge.kind}
              data-status={status}
              data-relation={relation}
            >
              <path
                className="architecture-edge__context"
                d={edge.path}
                pathLength="1"
              />
              <path
                className="architecture-edge__signal"
                d={edge.path}
                pathLength="1"
                markerEnd={
                  edge.kind === 'promotion' || edge.kind === 'output'
                    ? 'url(#architecture-arrow-orange)'
                    : 'url(#architecture-arrow-blue)'
                }
              />
              {status === 'active' ? (
                <circle className="architecture-edge__packet" r="6" filter="url(#architecture-packet-glow)">
                  <animateMotion dur="620ms" fill="freeze" path={edge.path} />
                </circle>
              ) : null}
            </g>
          )
        })}

        <g className="architecture-merge-label">
          <text x="175" y="464">SEARCH REQUEST</text>
          <path d="M 178 476 L 294 476" />
        </g>

        <g className="architecture-gate-label architecture-gate-label--accepted">
          <text x="1055" y="447">ACCEPTED</text>
        </g>
        <g className="architecture-gate-label architecture-gate-label--rejected">
          <path d="M 975 552 L 1005 582 M 1005 552 L 975 582" />
          <text x="1020" y="577">NO GPU TIME</text>
        </g>

        <g className="architecture-trace-pair" data-active={activeStep === 'measure' || activeStep === 'resolve'}>
          <path d="M 1240 440 C 1290 416 1320 382 1365 368" />
          <path d="M 1240 457 C 1295 438 1330 404 1372 393" />
          <text x="1260" y="398">EXPECTED TRACE</text>
          <text x="1260" y="430">OBSERVED TRACE</text>
          <text className="architecture-trace-pair__match" x="1350" y="340">PATH MATCH</text>
        </g>

        <g className="architecture-feedback-label" data-active={stageStatus('measure', activeStep)}>
          <text x="720" y="850">LATENCY · FAILURES · COVERAGE</text>
          <text x="720" y="880">UPDATE THE SAMPLER</text>
        </g>

        <g className="architecture-output-rail" data-active={activeStep === 'resolve'}>
          <path d="M 1465 352 L 1785 352" />
          <text x="1490" y="330">SAME MEASURED PROGRAM</text>
        </g>
      </svg>

      <div className="architecture-figure__nodes">
        {architectureNodes.map((node) => {
          const status = stageStatus(node.step, activeStep)
          const relation = nodeRelation(node.id, activeNodeId)

          return (
            <button
              key={node.id}
              type="button"
              className="architecture-node"
              data-node-id={node.id}
              data-kind={node.kind}
              data-status={status}
              data-relation={relation}
              aria-pressed={pinnedNodeId === node.id}
              aria-label={`${node.label}. ${node.detail}`}
              style={{ left: `${node.x}%`, top: `${node.y}%` }}
              onPointerEnter={() => onNodeEnter(node.id)}
              onPointerLeave={onNodeLeave}
              onFocus={() => onNodeEnter(node.id)}
              onBlur={onNodeLeave}
              onClick={() => onNodeToggle(node.id)}
            >
              {node.eyebrow ? <span className="architecture-node__eyebrow">{node.eyebrow}</span> : null}
              <strong className="architecture-node__label">{node.label}</strong>
            </button>
          )
        })}
      </div>

      <div
        className="architecture-runtime-lanes"
        data-active={stageStatus('measure', activeStep)}
        aria-label="Execution plan runtime lanes"
      >
        {runtimeLanes.map((lane) => (
          <span key={lane.id} className="architecture-runtime-lane" data-lane={lane.id}>
            <strong>{lane.label}</strong>
            <small>{lane.repository}</small>
          </span>
        ))}
      </div>

      <div
        className="architecture-evidence-channels"
        data-active={stageStatus('measure', activeStep)}
        aria-label="Measured GPU evidence channels"
      >
        <span className="architecture-evidence-isolation">SINGLE-GPU LEASE · PER-SHAPE FRESH PROCESS</span>
        {evidenceChannels.map((channel) => (
          <span key={channel} className="architecture-evidence-channel">{channel}</span>
        ))}
      </div>

      <div
        className="architecture-promotion-rule"
        data-active={stageStatus('promote', activeStep)}
        aria-label="Formal paired promotion thresholds"
      >
        <span className="architecture-promotion-rule__order">AB ↔ BA</span>
        {promotionThresholds.map((threshold) => (
          <span key={threshold}>{threshold}</span>
        ))}
      </div>

      <div
        className="architecture-registry-grid"
        data-active={activeStep === 'promote' || activeStep === 'resolve'}
        aria-label="Environment by shape deployment registry"
      >
        <span className="architecture-registry-grid__axis architecture-registry-grid__axis--environment">
          ENVIRONMENT
        </span>
        <span className="architecture-registry-grid__axis architecture-registry-grid__axis--shape">
          SHAPE
        </span>
        {Array.from({ length: 12 }, (_, index) => (
          <i key={index} data-winner={index === 6 ? 'true' : 'false'} />
        ))}
        <span className="architecture-registry-grid__winner">APPROVED CONFIG SPEC</span>
        <span className="architecture-registry-grid__fallback">NO MATCH → PORTABLE FALLBACK</span>
      </div>

      <div className="architecture-evidence-stores" aria-label="Evidence and deployment stores">
        {evidenceStores.map((store) => (
          <span key={store.label} className="architecture-evidence-store">
            <strong>{store.label}</strong>
            <small>{store.detail}</small>
          </span>
        ))}
      </div>
    </figure>
  )
}
