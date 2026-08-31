import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react'
import {
  architectureNodes,
  architecturePhases,
  workflowCommands,
  type ArchitectureNodeId,
} from '../data/architectureData'
import { useNarrativeStep } from '../hooks/useNarrativeStep'
import type { ArchitectureStep } from '../motion/motionTypes'
import { ArchitectureFigure } from '../visuals/ArchitectureFigure'

export function ArchitectureSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const activeStep = useNarrativeStep<ArchitectureStep>(sectionRef, 'construct')
  const [hoveredNodeId, setHoveredNodeId] = useState<ArchitectureNodeId | null>(null)
  const [pinnedNodeId, setPinnedNodeId] = useState<ArchitectureNodeId | null>(null)

  const activeNodeId = pinnedNodeId ?? hoveredNodeId
  const activeNode = useMemo(
    () => architectureNodes.find((node) => node.id === activeNodeId) ?? null,
    [activeNodeId],
  )

  useEffect(() => {
    const clearPinnedNode = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') setPinnedNodeId(null)
    }

    window.addEventListener('keydown', clearPinnedNode)
    return () => window.removeEventListener('keydown', clearPinnedNode)
  }, [])

  const togglePinnedNode = (nodeId: ArchitectureNodeId) => {
    setPinnedNodeId((currentNodeId) => (currentNodeId === nodeId ? null : nodeId))
  }

  const clearPinnedNodeFromPlane = (event: MouseEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement
    if (target.closest('[data-node-id], [data-architecture-interactive]')) return
    setPinnedNodeId(null)
  }

  return (
    <section
      ref={sectionRef}
      id="architecture"
      className="architecture-section"
      data-active-step={activeStep}
      aria-labelledby="architecture-title"
    >
      <div className="architecture-sticky" onClick={clearPinnedNodeFromPlane}>
        <header className="architecture-heading">
          <div className="architecture-heading__copy">
            <p className="architecture-heading__overline">END-TO-END ARCHITECTURE</p>
            <h2 id="architecture-title">ONE CLOSED EVIDENCE LOOP</h2>
            <p className="architecture-heading__summary">
              The program we search is the program we measure, promote and execute.
            </p>
          </div>

          <div className="architecture-command-rail" aria-label="Bounded workflow entry points">
            {workflowCommands.map((command) => (
              <span key={command.label} className="architecture-command">
                <strong>{command.label}</strong>
                <small>{command.scope}</small>
              </span>
            ))}
          </div>
        </header>

        <div className="architecture-system-plane">
          <ArchitectureFigure
            activeStep={activeStep}
            activeNodeId={activeNodeId}
            pinnedNodeId={pinnedNodeId}
            onNodeEnter={setHoveredNodeId}
            onNodeLeave={() => setHoveredNodeId(null)}
            onNodeToggle={togglePinnedNode}
          />

          <ol className="architecture-phase-index" aria-label="Architecture phases">
            {architecturePhases.map((phase) => (
              <li
                key={phase.id}
                data-step-name={phase.id}
                data-active={phase.id === activeStep}
                aria-current={phase.id === activeStep ? 'step' : undefined}
              >
                <span>{phase.index}</span>
                <strong>{phase.label}</strong>
              </li>
            ))}
          </ol>

          <div className="architecture-phase-facts" aria-live="polite">
            {architecturePhases.map((phase) => (
              <p
                key={phase.id}
                className={`architecture-phase-fact architecture-phase-fact--${phase.side}`}
                data-step-name={phase.id}
                data-active={phase.id === activeStep}
              >
                <span>{phase.label}</span>
                {phase.sentence}
              </p>
            ))}
          </div>

          {activeNode ? (
            <aside
              className={`architecture-inspector architecture-inspector--${activeNode.inspectorSide}`}
              data-architecture-interactive
              data-pinned={pinnedNodeId === activeNode.id}
              aria-label={`${activeNode.label} module details`}
            >
              <div className="architecture-inspector__heading">
                <span>{pinnedNodeId === activeNode.id ? 'PINNED MODULE' : 'MODULE TRACE'}</span>
                <strong>{activeNode.label}</strong>
              </div>
              <code>{activeNode.repository}</code>
              <dl>
                <div>
                  <dt>INPUT</dt>
                  <dd>{activeNode.input}</dd>
                </div>
                <div>
                  <dt>OUTPUT</dt>
                  <dd>{activeNode.output}</dd>
                </div>
              </dl>
              <p>{activeNode.detail}</p>
              {activeNode.evidence ? <small>{activeNode.evidence}</small> : null}
              {pinnedNodeId === activeNode.id ? (
                <button type="button" onClick={() => setPinnedNodeId(null)}>
                  RELEASE · ESC
                </button>
              ) : null}
            </aside>
          ) : null}
        </div>
      </div>

      <div className="architecture-scroll-track" aria-hidden="true">
        {architecturePhases.map((phase) => (
          <div key={phase.id} className="architecture-sentinel" data-step={phase.id} />
        ))}
      </div>
    </section>
  )
}
