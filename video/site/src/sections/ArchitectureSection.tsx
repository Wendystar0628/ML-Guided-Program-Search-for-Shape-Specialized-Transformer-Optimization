import { useEffect, useMemo, useState } from 'react'
import { architectureNodes, type ArchitectureNodeId } from '../data/architectureData'
import type { ArchitectureStep } from '../motion/motionTypes'
import { ArchitectureFigure } from '../visuals/ArchitectureFigure'

export function ArchitectureSection() {
  const [activeStep, setActiveStep] = useState<ArchitectureStep>('resolve')
  const [hoveredNodeId, setHoveredNodeId] = useState<ArchitectureNodeId | null>(null)
  const [pinnedNodeId, setPinnedNodeId] = useState<ArchitectureNodeId | null>(null)

  const activeNodeId = pinnedNodeId ?? hoveredNodeId
  const activeNode = useMemo(
    () => architectureNodes.find((node) => node.id === activeNodeId) ?? null,
    [activeNodeId],
  )

  useEffect(() => {
    const clearPinnedNode = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setPinnedNodeId(null)
        setActiveStep('resolve')
      }
    }

    window.addEventListener('keydown', clearPinnedNode)
    return () => window.removeEventListener('keydown', clearPinnedNode)
  }, [])

  const inspectNode = (nodeId: ArchitectureNodeId) => {
    if (pinnedNodeId) return
    setHoveredNodeId(nodeId)
    const node = architectureNodes.find((candidate) => candidate.id === nodeId)
    if (node) setActiveStep(node.step)
  }

  const leaveNode = () => {
    setHoveredNodeId(null)
    if (!pinnedNodeId) setActiveStep('resolve')
  }

  const togglePinnedNode = (nodeId: ArchitectureNodeId) => {
    if (pinnedNodeId === nodeId) {
      setPinnedNodeId(null)
      setActiveStep('resolve')
      return
    }

    setPinnedNodeId(nodeId)
    const node = architectureNodes.find((candidate) => candidate.id === nodeId)
    if (node) setActiveStep(node.step)
  }

  return (
    <section
      id="architecture"
      className="architecture-section architecture-section--flow"
      aria-labelledby="architecture-title"
    >
      <div className="architecture-flow-shell">
        <header className="architecture-flow-heading">
          <div>
            <p className="architecture-flow-heading__overline">Implementation overview</p>
            <h2 id="architecture-title">System Architecture and Execution Lifecycle</h2>
          </div>
          <div className="architecture-flow-heading__aside">
            <p>
              Three system layers construct a legal ExecutionPlan, collect multi-fidelity
              GPU evidence, and resolve the approved ConfigSpec for runtime execution.
            </p>
            <small>Hover to inspect · click to pin · press Esc to reset</small>
          </div>
        </header>

        <ArchitectureFigure
          activeStep={activeStep}
          activeNodeId={activeNodeId}
          pinnedNodeId={pinnedNodeId}
          onNodeEnter={inspectNode}
          onNodeLeave={leaveNode}
          onNodeToggle={togglePinnedNode}
        />

        <div className="architecture-flow-detail" aria-live="polite">
          {activeNode ? (
            <article className="architecture-flow-inspector">
              <div className="architecture-flow-inspector__title">
                <span>{pinnedNodeId === activeNode.id ? 'Selected module' : activeNode.eyebrow}</span>
                <strong>{activeNode.label}</strong>
              </div>
              <code>{activeNode.repository}</code>
              <dl>
                <div>
                  <dt>Input</dt>
                  <dd>{activeNode.input}</dd>
                </div>
                <div>
                  <dt>Output</dt>
                  <dd>{activeNode.output}</dd>
                </div>
              </dl>
              <div className="architecture-flow-inspector__explanation">
                <p>{activeNode.detail}</p>
                {activeNode.evidence ? <small>{activeNode.evidence}</small> : null}
              </div>
            </article>
          ) : (
            <div className="architecture-flow-principles">
              <p><span>01</span><strong>Program construction</strong><small>Workload identity → ConfigSpec → ExecutionPlan</small></p>
              <p><span>02</span><strong>Evidence collection</strong><small>Screen → Enhanced → paired Formal evidence</small></p>
              <p><span>03</span><strong>Deployment resolution</strong><small>Registry key → runtime branch → official output</small></p>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
