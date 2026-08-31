import { useEffect, useMemo, useRef, useState } from 'react'
import dagre from '@dagrejs/dagre'
import {
  Background,
  BackgroundVariant,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type ReactFlowInstance,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import '../styles/architecture.css'
import {
  architectureConnectors,
  architectureEdges,
  architectureLanes,
  architectureNodes,
  architectureSteps,
  type ArchitectureLane,
  type ArchitectureLaneId,
  type ArchitectureNode,
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

type NodeStatus = 'complete' | 'active' | 'future'
type NodeRelation = 'context' | 'selected' | 'related' | 'muted'
type FlowNodeData = ArchitectureNode & {
  status: NodeStatus
  relation: NodeRelation
  pinned: boolean
  [key: string]: unknown
}
type ArchitectureFlowNode = Node<FlowNodeData, 'architecture'>
type ArchitectureFlowEdge = Edge<Record<string, never>>

const NODE_HEIGHT = 142

function nodeWidth(node: ArchitectureNode) {
  if (node.id === 'conditional-search') return 320
  if (node.id === 'plan-builder' || node.id === 'official-context') return 292
  if (node.id === 'formal-paired') return 320
  if (node.id === 'sequential-promotion' || node.id === 'official-output') return 292
  if (node.id === 'exact-device-registry') return 310
  return 250
}

function createLaneLayout(lane: ArchitectureLane) {
  const graph = new dagre.graphlib.Graph()
    .setDefaultEdgeLabel(() => ({}))
    .setGraph({
      rankdir: 'LR',
      ranker: 'network-simplex',
      ranksep: 64,
      nodesep: 34,
      marginx: 24,
      marginy: 24,
    })

  const laneNodes = architectureNodes.filter((node) => lane.nodeIds.includes(node.id))
  const laneEdges = architectureEdges.filter((edge) => edge.lane === lane.id)

  laneNodes.forEach((node) => {
    graph.setNode(node.id, { width: nodeWidth(node), height: NODE_HEIGHT })
  })
  laneEdges.forEach((edge) => graph.setEdge(edge.from, edge.to))
  dagre.layout(graph)

  return new Map(
    laneNodes.map((node) => {
      const point = graph.node(node.id)
      return [
        node.id,
        {
          x: point.x - nodeWidth(node) / 2,
          y: point.y - NODE_HEIGHT / 2,
        },
      ]
    }),
  )
}

const laneLayouts = new Map(
  architectureLanes.map((lane) => [lane.id, createLaneLayout(lane)]),
)

function stageStatus(stage: ArchitectureStep, activeStep: ArchitectureStep): NodeStatus {
  const stageIndex = architectureSteps.indexOf(stage)
  const activeIndex = architectureSteps.indexOf(activeStep)

  if (stageIndex < activeIndex) return 'complete'
  if (stageIndex === activeIndex) return 'active'
  return 'future'
}

function laneIsActive(laneId: ArchitectureLaneId, activeStep: ArchitectureStep) {
  if (laneId === 'synthesis') return activeStep === 'construct' || activeStep === 'validate'
  if (laneId === 'evidence') return activeStep === 'measure'
  return activeStep === 'promote' || activeStep === 'resolve'
}

function ArchitectureFlowNode({ data }: NodeProps<ArchitectureFlowNode>) {
  return (
    <div
      className="architecture-flow-node"
      data-kind={data.kind}
      data-status={data.status}
      data-relation={data.relation}
      data-pinned={data.pinned}
    >
      <Handle id="target" type="target" position={Position.Left} />
      <span className="architecture-flow-node__eyebrow">{data.eyebrow}</span>
      <strong>{data.label}</strong>
      <small>{data.note}</small>
      <Handle id="source" type="source" position={Position.Right} />
    </div>
  )
}

const nodeTypes = { architecture: ArchitectureFlowNode }

type ArchitectureLaneFlowProps = ArchitectureFigureProps & {
  lane: ArchitectureLane
  connectedNodeIds: Set<ArchitectureNodeId>
}

function ArchitectureLaneFlow({
  lane,
  activeStep,
  activeNodeId,
  pinnedNodeId,
  connectedNodeIds,
  onNodeEnter,
  onNodeLeave,
  onNodeToggle,
}: ArchitectureLaneFlowProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [flowInstance, setFlowInstance] = useState<
    ReactFlowInstance<ArchitectureFlowNode, ArchitectureFlowEdge> | undefined
  >()

  const nodes = useMemo<ArchitectureFlowNode[]>(() => {
    const positions = laneLayouts.get(lane.id)
    return architectureNodes
      .filter((node) => lane.nodeIds.includes(node.id))
      .map((node) => {
        const relation: NodeRelation = !activeNodeId
          ? 'context'
          : node.id === activeNodeId
            ? 'selected'
            : connectedNodeIds.has(node.id)
              ? 'related'
              : 'muted'

        return {
          id: node.id,
          type: 'architecture',
          position: positions?.get(node.id) ?? { x: 0, y: 0 },
          width: nodeWidth(node),
          height: NODE_HEIGHT,
          style: { width: nodeWidth(node), height: NODE_HEIGHT },
          data: {
            ...node,
            status: stageStatus(node.step, activeStep),
            relation,
            pinned: pinnedNodeId === node.id,
          },
          ariaLabel: `${node.label}. ${node.detail}`,
          draggable: false,
          selectable: false,
        }
      })
  }, [activeNodeId, activeStep, connectedNodeIds, lane, pinnedNodeId])

  const edges = useMemo<ArchitectureFlowEdge[]>(
    () =>
      architectureEdges
        .filter((edge) => edge.lane === lane.id)
        .map((edge) => {
          const status = stageStatus(edge.stage, activeStep)
          const related = !activeNodeId || edge.from === activeNodeId || edge.to === activeNodeId
          const isOrange = edge.kind === 'promotion'
          const stroke = isOrange ? '#f05a2a' : '#2457d6'

          return {
            id: edge.id,
            source: edge.from,
            target: edge.to,
            sourceHandle: 'source',
            targetHandle: 'target',
            type: 'smoothstep',
            animated: status === 'active',
            className: [
              'architecture-flow__edge',
              `architecture-flow__edge--${status}`,
              related ? 'architecture-flow__edge--related' : 'architecture-flow__edge--muted',
            ].join(' '),
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: stroke,
              width: 22,
              height: 22,
            },
            style: {
              stroke,
              strokeWidth: status === 'active' ? 4.2 : 3.2,
            },
            pathOptions: { borderRadius: 18 },
            focusable: false,
            selectable: false,
          }
        }),
    [activeNodeId, activeStep, lane.id],
  )

  useEffect(() => {
    const container = containerRef.current
    if (!container || !flowInstance) return

    let animationFrame = 0
    const fitGraph = () => {
      cancelAnimationFrame(animationFrame)
      animationFrame = requestAnimationFrame(() => {
        void flowInstance.fitView({
          padding: lane.id === 'deployment' ? 0.13 : 0.09,
          duration: 180,
        })
      })
    }
    const resizeObserver = new ResizeObserver(fitGraph)
    resizeObserver.observe(container)
    fitGraph()

    return () => {
      cancelAnimationFrame(animationFrame)
      resizeObserver.disconnect()
    }
  }, [flowInstance])

  return (
    <div ref={containerRef} className="architecture-band__canvas">
      <ReactFlow<ArchitectureFlowNode, ArchitectureFlowEdge>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onInit={setFlowInstance}
        onNodeMouseEnter={(_event, node) => onNodeEnter(node.id as ArchitectureNodeId)}
        onNodeMouseLeave={onNodeLeave}
        onNodeClick={(_event, node) => onNodeToggle(node.id as ArchitectureNodeId)}
        fitView
        fitViewOptions={{ padding: lane.id === 'deployment' ? 0.13 : 0.09 }}
        minZoom={0.35}
        maxZoom={1.08}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        panOnScroll={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        preventScrolling={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={26}
          size={1.25}
          color="rgba(73, 88, 103, 0.16)"
        />
      </ReactFlow>
    </div>
  )
}

export function ArchitectureFigure(props: ArchitectureFigureProps) {
  const connectedNodeIds = useMemo(() => {
    if (!props.activeNodeId) return new Set<ArchitectureNodeId>()

    const ids = new Set<ArchitectureNodeId>([props.activeNodeId])
    architectureEdges.forEach((edge) => {
      if (edge.from === props.activeNodeId) ids.add(edge.to)
      if (edge.to === props.activeNodeId) ids.add(edge.from)
    })
    return ids
  }, [props.activeNodeId])

  return (
    <figure
      className="architecture-figure architecture-bands"
      data-active-step={props.activeStep}
      aria-label="System architecture from program synthesis through measurement and deployment execution"
    >
      {architectureLanes.map((lane) => {
        const connector = architectureConnectors.find((item) => item.afterLane === lane.id)

        return (
          <div className="architecture-band-group" key={lane.id}>
            <section
              className="architecture-band"
              data-lane={lane.id}
              data-active={laneIsActive(lane.id, props.activeStep)}
              aria-labelledby={`architecture-lane-${lane.id}`}
            >
              <header className="architecture-band__heading">
                <span>{lane.index}</span>
                <div>
                  <h3 id={`architecture-lane-${lane.id}`}>{lane.title}</h3>
                  <p>{lane.summary}</p>
                </div>
              </header>

              <ArchitectureLaneFlow
                {...props}
                lane={lane}
                connectedNodeIds={connectedNodeIds}
              />
            </section>

            {connector ? (
              <div className="architecture-band-connector" aria-label={connector.primary}>
                <span className="architecture-band-connector__primary">
                  <i aria-hidden="true">↓</i>
                  <strong>{connector.primary}</strong>
                </span>
                {connector.feedback ? (
                  <span className="architecture-band-connector__feedback">
                    <i aria-hidden="true">↺</i>
                    {connector.feedback}
                  </span>
                ) : null}
              </div>
            ) : null}
          </div>
        )
      })}
    </figure>
  )
}
