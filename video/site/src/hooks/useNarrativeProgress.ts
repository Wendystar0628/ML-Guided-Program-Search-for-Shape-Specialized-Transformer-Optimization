import { useEffect, useRef, useState, type RefObject } from 'react'
import { clamp01 } from '../motion/programFieldRanges'

export function useNarrativeProgress(
  containerRef: RefObject<HTMLElement | null>,
): number {
  const [progress, setProgress] = useState(0)
  const frameRef = useRef<number | null>(null)
  const lastProgressRef = useRef(-1)

  useEffect(() => {
    const update = () => {
      frameRef.current = null
      const container = containerRef.current
      if (!container) {
        return
      }

      const rect = container.getBoundingClientRect()
      const travel = Math.max(rect.height - window.innerHeight, 1)
      const nextProgress = clamp01(-rect.top / travel)

      if (Math.abs(nextProgress - lastProgressRef.current) >= 0.001) {
        lastProgressRef.current = nextProgress
        setProgress(nextProgress)
      }
    }

    const scheduleUpdate = () => {
      if (frameRef.current === null) {
        frameRef.current = window.requestAnimationFrame(update)
      }
    }

    window.addEventListener('scroll', scheduleUpdate, { passive: true })
    window.addEventListener('resize', scheduleUpdate)
    scheduleUpdate()

    return () => {
      window.removeEventListener('scroll', scheduleUpdate)
      window.removeEventListener('resize', scheduleUpdate)
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current)
      }
    }
  }, [containerRef])

  return progress
}
