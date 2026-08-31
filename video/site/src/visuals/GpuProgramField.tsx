import { Canvas } from '@react-three/fiber'
import { useMemo } from 'react'
import * as THREE from 'three'

type Point = readonly [number, number, number]

type TraceProps = {
  points: readonly Point[]
  color: string
  opacity?: number
  radius?: number
}

const candidateY = [-2.55, -1.7, -0.9, 0, 0.9, 1.7, 2.55] as const

const candidatePaths = candidateY.map(
  (y, index) =>
    [
      [-5.5, y, -0.12],
      [-4.35, y + (index % 2 === 0 ? 0.12 : -0.12), -0.08],
      [-3.2, y * 0.88, -0.04],
      [-2.18, y * 0.56, 0],
      [-1.28, y * 0.28, 0.08],
    ] as const,
)

const winnerPath = [
  [1.78, 0, 0.18],
  [2.55, 0, 0.16],
  [3.45, 0.08, 0.12],
  [4.4, 0.08, 0.08],
  [5.25, 0.08, 0.04],
] as const

function ProgramTrace({
  points,
  color,
  opacity = 1,
  radius = 0.045,
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
        roughness={0.68}
        metalness={0.08}
        transparent={opacity < 1}
        opacity={opacity}
      />
    </mesh>
  )
}

function CandidateInputs() {
  return (
    <group>
      {candidatePaths.map((points, index) => {
        const selected = index === 3
        const y = candidateY[index]

        return (
          <group key={y}>
            <ProgramTrace
              points={points}
              color="#2457D6"
              opacity={selected ? 1 : 0.32 + index * 0.025}
              radius={selected ? 0.068 : 0.038}
            />
            <mesh position={[-5.56, y, -0.12]}>
              <boxGeometry args={[0.26, 0.18, 0.16]} />
              <meshStandardMaterial
                color={selected ? '#2457D6' : '#9AA9C9'}
                roughness={0.82}
                metalness={0.04}
              />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}

function MeasurementSurface() {
  const contacts = [-1.28, -0.9, -0.52, -0.14, 0.24, 0.62, 1, 1.38]
  const lanes = [-0.82, -0.41, 0, 0.41, 0.82]

  return (
    <group position={[0.25, 0, 0]}>
      <mesh position={[0, 0, -0.04]}>
        <boxGeometry args={[3.12, 3.84, 0.22]} />
        <meshStandardMaterial
          color="#E7EBEA"
          roughness={0.9}
          metalness={0.02}
        />
      </mesh>

      <mesh position={[0, 0, 0.13]}>
        <boxGeometry args={[2.36, 2.9, 0.2]} />
        <meshStandardMaterial
          color="#36424B"
          roughness={0.76}
          metalness={0.16}
        />
      </mesh>

      <mesh position={[0, 0, 0.28]}>
        <boxGeometry args={[1.58, 2.18, 0.1]} />
        <meshStandardMaterial
          color="#F5F7F6"
          roughness={0.96}
          metalness={0}
        />
      </mesh>

      {lanes.map((y, index) => (
        <mesh key={y} position={[0, y, 0.36]}>
          <boxGeometry args={[1.18 - index * 0.06, 0.075, 0.055]} />
          <meshStandardMaterial
            color={index === 2 ? '#F06432' : '#2457D6'}
            roughness={0.72}
            metalness={0.04}
          />
        </mesh>
      ))}

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

      {[[-1.2, -1.55], [1.2, -1.55], [-1.2, 1.55], [1.2, 1.55]].map(
        ([x, y]) => (
          <mesh key={`${x}-${y}`} position={[x, y, 0.22]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.08, 0.08, 0.08, 20]} />
            <meshStandardMaterial color="#A7AFB1" roughness={0.8} metalness={0.12} />
          </mesh>
        ),
      )}
    </group>
  )
}

function WinnerOutput() {
  return (
    <group>
      <ProgramTrace points={winnerPath} color="#F06432" radius={0.082} />
      <mesh position={[5.48, 0.08, 0.04]}>
        <boxGeometry args={[0.48, 0.48, 0.24]} />
        <meshStandardMaterial color="#F06432" roughness={0.76} metalness={0.06} />
      </mesh>
      <mesh position={[5.48, 0.08, 0.18]}>
        <boxGeometry args={[0.2, 0.2, 0.08]} />
        <meshStandardMaterial color="#FFF8F2" roughness={0.92} metalness={0} />
      </mesh>
    </group>
  )
}

function ProgramFieldScene() {
  return (
    <>
      <ambientLight intensity={1.65} />
      <directionalLight position={[4, 6, 10]} intensity={2.2} color="#FFFFFF" />
      <directionalLight position={[-5, -3, 6]} intensity={0.62} color="#C9D7F4" />

      <group rotation={[-0.08, -0.16, -0.018]}>
        <CandidateInputs />
        <MeasurementSurface />
        <WinnerOutput />
      </group>
    </>
  )
}

export function GpuProgramField() {
  return (
    <Canvas
      className="gpu-program-field"
      frameloop="demand"
      orthographic
      camera={{ position: [0, 0, 12], zoom: 88, near: 0.1, far: 40 }}
      gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
      role="img"
      aria-label="Candidate programs converge through GPU measurement into one measured winner."
    >
      <ProgramFieldScene />
    </Canvas>
  )
}
