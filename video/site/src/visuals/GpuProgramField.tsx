import { Canvas, useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import * as THREE from 'three'
import {
  PROGRAM_FIELD_RANGES,
  clamp01,
  rangeProgress,
} from '../motion/programFieldRanges'

type Point = readonly [number, number, number]

type TraceProps = {
  points: readonly Point[]
  color: string
  opacity?: number
  radius?: number
  emissive?: string
  emissiveIntensity?: number
}

type GpuProgramFieldProps = {
  progress: number
  inspectedCandidate: number | null
  onInspectCandidate: (candidate: number | null) => void
}

const candidateY = [-2.55, -1.7, -0.9, 0, 0.9, 1.7, 2.55] as const

const winnerPath = [
  [1.76, 0, 0.18],
  [2.68, 0, 0.16],
  [3.72, 0.04, 0.12],
  [4.74, -0.26, 0.08],
  [5.52, -1.08, 0.04],
] as const

function ProgramTrace({
  points,
  color,
  opacity = 1,
  radius = 0.045,
  emissive = color,
  emissiveIntensity = 0,
}: TraceProps) {
  const curve = useMemo(
    () =>
      new THREE.CatmullRomCurve3(
        points.map(([x, y, z]) => new THREE.Vector3(x, y, z)),
        false,
        'catmullrom',
        0.24,
      ),
    [points],
  )

  return (
    <mesh>
      <tubeGeometry args={[curve, 56, radius, 8, false]} />
      <meshStandardMaterial
        color={color}
        emissive={emissive}
        emissiveIntensity={emissiveIntensity}
        roughness={0.68}
        metalness={0.08}
        transparent
        opacity={opacity}
      />
    </mesh>
  )
}

function CameraRig({ progress }: { progress: number }) {
  const { camera, invalidate } = useThree()

  useEffect(() => {
    if (!(camera instanceof THREE.OrthographicCamera)) return

    const focus = rangeProgress(progress, PROGRAM_FIELD_RANGES.winner)
    camera.position.x = 0.34 * focus
    camera.position.y = -0.12 * focus
    camera.zoom = 94 * (0.96 + focus * 0.08)
    camera.updateProjectionMatrix()
    invalidate()
  }, [camera, invalidate, progress])

  return null
}

function CandidateInputs({
  progress,
  inspectedCandidate,
  onInspectCandidate,
}: GpuProgramFieldProps) {
  const convergence = rangeProgress(progress, PROGRAM_FIELD_RANGES.converge)
  const measurement = rangeProgress(progress, PROGRAM_FIELD_RANGES.measure)

  const candidatePaths = useMemo(
    () =>
      candidateY.map((y, index) => {
        const bend = index % 2 === 0 ? 0.12 : -0.12

        return [
          [-5.5, y, -0.12],
          [-4.35 + convergence * 0.14, y + bend, -0.08],
          [-3.2 + convergence * 0.24, y * (0.88 - convergence * 0.08), -0.04],
          [-2.18 + convergence * 0.34, y * (0.56 - convergence * 0.12), 0],
          [-1.28 + convergence * 0.34, y * (0.28 - convergence * 0.13), 0.08],
        ] as const
      }),
    [convergence],
  )

  return (
    <group>
      {candidatePaths.map((points, index) => {
        const selected = index === 3
        const inspected = inspectedCandidate === index
        const anotherInspected = inspectedCandidate !== null && !inspected
        const baseOpacity = selected ? 1 : 0.48
        const opacity = anotherInspected
          ? 0.13
          : inspected
            ? 1
            : baseOpacity * (1 - measurement * (selected ? 0.05 : 0.48))
        const radius = inspected ? 0.092 : selected ? 0.068 : 0.042
        const y = candidateY[index]

        return (
          <group
            key={y}
            position={[0, 0, inspected ? 0.2 : 0]}
            onPointerOver={(event) => {
              event.stopPropagation()
              onInspectCandidate(index)
            }}
            onPointerOut={(event) => {
              event.stopPropagation()
              onInspectCandidate(null)
            }}
          >
            <ProgramTrace
              points={points}
              color={inspected ? '#164CD6' : '#2457D6'}
              opacity={opacity}
              radius={radius}
              emissiveIntensity={inspected ? 0.3 : selected ? 0.08 : 0}
            />

            <mesh position={[-5.56, y, -0.12]}>
              <boxGeometry args={[0.3, 0.2, 0.18]} />
              <meshStandardMaterial
                color={inspected || selected ? '#2457D6' : '#9AA9C9'}
                emissive="#2457D6"
                emissiveIntensity={inspected ? 0.26 : 0}
                roughness={0.82}
                metalness={0.04}
                transparent
                opacity={anotherInspected ? 0.22 : 1}
              />
            </mesh>

            {[-5.18, -4.82, -4.46].map((x, nodeIndex) => (
              <mesh key={x} position={[x, y + (nodeIndex - 1) * 0.06, -0.06]}>
                <boxGeometry args={[0.16, 0.1, 0.1]} />
                <meshStandardMaterial
                  color={inspected ? '#F5F8FF' : '#B7C5E4'}
                  roughness={0.78}
                  transparent
                  opacity={anotherInspected ? 0.18 : 0.9}
                />
              </mesh>
            ))}
          </group>
        )
      })}
    </group>
  )
}

function MeasurementSurface({ progress }: { progress: number }) {
  const measurement = rangeProgress(progress, PROGRAM_FIELD_RANGES.measure)
  const rails = [-0.66, -0.22, 0.22, 0.66]
  const contacts = [-1.28, -0.9, -0.52, -0.14, 0.24, 0.62, 1, 1.38]

  return (
    <group position={[0.25, 0, 0.04 + measurement * 0.06]}>
      <mesh position={[0, 0, -0.04]}>
        <boxGeometry args={[3.12, 3.84, 0.22]} />
        <meshStandardMaterial
          color="#E7EBEA"
          emissive="#2457D6"
          emissiveIntensity={measurement * 0.05}
          roughness={0.9}
          metalness={0.02}
        />
      </mesh>

      <mesh position={[0, 0, 0.13]}>
        <boxGeometry args={[2.36, 2.9, 0.2]} />
        <meshStandardMaterial color="#36424B" roughness={0.76} metalness={0.16} />
      </mesh>

      <mesh position={[0, 0, 0.28]}>
        <boxGeometry args={[1.58, 2.18, 0.1]} />
        <meshStandardMaterial color="#F5F7F6" roughness={0.96} metalness={0} />
      </mesh>

      {rails.map((y, index) => {
        const railActivation = clamp01(measurement * 4 - index)
        const color = index === 2 ? '#F06432' : '#2457D6'

        return (
          <mesh key={y} position={[0, y, 0.37 + railActivation * 0.035]}>
            <boxGeometry args={[1.28, 0.095, 0.06]} />
            <meshStandardMaterial
              color={color}
              emissive={color}
              emissiveIntensity={railActivation * 0.56}
              roughness={0.68}
              metalness={0.04}
            />
          </mesh>
        )
      })}

      {contacts.map((y) => (
        <group key={y}>
          <mesh position={[-1.7, y, -0.01]}>
            <boxGeometry args={[0.32, 0.095, 0.08]} />
            <meshStandardMaterial color="#78838A" roughness={0.74} metalness={0.24} />
          </mesh>
          <mesh position={[1.7, y, -0.01]}>
            <boxGeometry args={[0.32, 0.095, 0.08]} />
            <meshStandardMaterial color="#78838A" roughness={0.74} metalness={0.24} />
          </mesh>
        </group>
      ))}
    </group>
  )
}

function WinnerOutput({ progress }: { progress: number }) {
  const winner = rangeProgress(progress, PROGRAM_FIELD_RANGES.winner)
  const dock = rangeProgress(progress, PROGRAM_FIELD_RANGES.dock)
  const curve = useMemo(
    () =>
      new THREE.CatmullRomCurve3(
        winnerPath.map(([x, y, z]) => new THREE.Vector3(x, y, z)),
        false,
        'catmullrom',
        0.24,
      ),
    [],
  )
  const tokenPosition = curve.getPoint(clamp01(winner * 0.82 + dock * 0.18))
  const tokenScale = Math.max(0.001, winner * (1 + dock * 0.18))

  return (
    <group>
      <ProgramTrace
        points={winnerPath}
        color="#F06432"
        opacity={0.08 + winner * 0.92}
        radius={0.035 + winner * 0.055}
        emissive="#F06432"
        emissiveIntensity={winner * 0.2 + dock * 0.42}
      />

      <group position={tokenPosition} scale={tokenScale}>
        <mesh>
          <boxGeometry args={[0.52, 0.52, 0.26]} />
          <meshStandardMaterial
            color="#F06432"
            emissive="#F06432"
            emissiveIntensity={0.16 + dock * 0.42}
            roughness={0.72}
            metalness={0.06}
          />
        </mesh>
        <mesh position={[0, 0, 0.17]}>
          <boxGeometry args={[0.2, 0.2, 0.08]} />
          <meshStandardMaterial color="#FFF8F2" roughness={0.92} metalness={0} />
        </mesh>
      </group>
    </group>
  )
}

function ProgramFieldScene(props: GpuProgramFieldProps) {
  const { progress } = props
  const sceneRotation =
    -0.16 + rangeProgress(progress, PROGRAM_FIELD_RANGES.winner) * 0.12

  return (
    <>
      <ambientLight intensity={1.65} />
      <directionalLight position={[4, 6, 10]} intensity={2.2} color="#FFFFFF" />
      <directionalLight position={[-5, -3, 6]} intensity={0.62} color="#C9D7F4" />
      <CameraRig progress={progress} />

      <group rotation={[-0.08, sceneRotation, -0.018]}>
        <CandidateInputs {...props} />
        <MeasurementSurface progress={progress} />
        <WinnerOutput progress={progress} />
      </group>
    </>
  )
}

export function GpuProgramField(props: GpuProgramFieldProps) {
  return (
    <Canvas
      className="gpu-program-field"
      frameloop="demand"
      orthographic
      camera={{ position: [0, 0, 12], zoom: 94, near: 0.1, far: 40 }}
      dpr={1}
      gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      role="img"
      aria-label="Complete program candidates converge through GPU measurement into one measured winner."
      onPointerMissed={() => props.onInspectCandidate(null)}
    >
      <ProgramFieldScene {...props} />
    </Canvas>
  )
}
