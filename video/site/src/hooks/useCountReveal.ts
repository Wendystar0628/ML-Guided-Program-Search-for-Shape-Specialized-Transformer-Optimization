import { useEffect, useRef, useState, type RefObject } from 'react'

export type CountRevealOptions = {
  from?: number
  duration?: number
}

export function useCountReveal(
  targetRef: RefObject<Element | null>,
  target: number,
  options: CountRevealOptions = {},
): number {
  const { from = 0, duration = 620 } = options
  const [value, setValue] = useState(from)
  const hasRevealedRef = useRef(false)

  useEffect(() => {
    const element = targetRef.current
    if (!element) {
      return
    }

    if (hasRevealedRef.current || duration <= 0) {
      setValue(target)
      hasRevealedRef.current = true
      return
    }

    let animationFrame: number | null = null

    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting) || hasRevealedRef.current) {
          return
        }

        hasRevealedRef.current = true
        observer.disconnect()
        const startedAt = performance.now()

        const animate = (now: number) => {
          const linear = Math.min((now - startedAt) / duration, 1)
          const eased = linear * linear * (3 - 2 * linear)
          setValue(from + (target - from) * eased)

          if (linear < 1) {
            animationFrame = window.requestAnimationFrame(animate)
          } else {
            animationFrame = null
          }
        }

        animationFrame = window.requestAnimationFrame(animate)
      },
      { rootMargin: '0px 0px -10% 0px', threshold: 0.15 },
    )

    observer.observe(element)

    return () => {
      observer.disconnect()
      if (animationFrame !== null) {
        window.cancelAnimationFrame(animationFrame)
      }
    }
  }, [duration, from, target, targetRef])

  return value
}
