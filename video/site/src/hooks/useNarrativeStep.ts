import { useEffect, useState, type RefObject } from 'react'

const STEP_SELECTOR = '[data-step], [data-narrative-step]'

function readStep<T extends string>(element: HTMLElement): T | null {
  const step = element.dataset.step ?? element.dataset.narrativeStep
  return step ? (step as T) : null
}

export function useNarrativeStep<T extends string>(
  containerRef: RefObject<HTMLElement | null>,
  initialStep: T,
): T {
  const [activeStep, setActiveStep] = useState<T>(initialStep)

  useEffect(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    const sentinels = Array.from(container.querySelectorAll<HTMLElement>(STEP_SELECTOR))
    if (sentinels.length === 0) {
      return
    }

    const readingLine = () => window.innerHeight * 0.48

    const publishNearest = (elements: HTMLElement[]) => {
      const line = readingLine()
      const nearest = elements.reduce((best, element) => {
        const bestDistance = Math.abs(best.getBoundingClientRect().top - line)
        const distance = Math.abs(element.getBoundingClientRect().top - line)
        return distance < bestDistance ? element : best
      })
      const step = readStep<T>(nearest)
      if (step) {
        setActiveStep(step)
      }
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const intersecting = entries
          .filter((entry) => entry.isIntersecting)
          .map((entry) => entry.target as HTMLElement)

        if (intersecting.length > 0) {
          publishNearest(intersecting)
        }
      },
      { rootMargin: '-38% 0px -52% 0px', threshold: 0 },
    )

    sentinels.forEach((sentinel) => observer.observe(sentinel))

    const initialFrame = window.requestAnimationFrame(() => {
      const line = readingLine()
      const passed = sentinels.filter(
        (sentinel) => sentinel.getBoundingClientRect().top <= line,
      )
      const initialSentinel = passed.at(-1)
      const step = initialSentinel ? readStep<T>(initialSentinel) : null
      setActiveStep(step ?? initialStep)
    })

    return () => {
      window.cancelAnimationFrame(initialFrame)
      observer.disconnect()
    }
  }, [containerRef, initialStep])

  return activeStep
}
