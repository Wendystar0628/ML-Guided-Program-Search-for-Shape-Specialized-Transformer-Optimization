import { useEffect, useState } from 'react'
import { narrativeAnchors } from '../data/narrativeData'

type NarrativeAnchorId = (typeof narrativeAnchors)[number]['id']

export function FlowProgress() {
  const [activeId, setActiveId] = useState<NarrativeAnchorId>('outcome')

  useEffect(() => {
    const sections = narrativeAnchors
      .map((anchor) => document.getElementById(anchor.id))
      .filter((section): section is HTMLElement => section !== null)

    if (sections.length === 0) {
      return
    }

    const setNearestToReadingLine = (candidates: HTMLElement[]) => {
      const readingLine = window.innerHeight * 0.48
      const nearest = candidates.reduce((best, section) => {
        const bestDistance = Math.abs(best.getBoundingClientRect().top - readingLine)
        const distance = Math.abs(section.getBoundingClientRect().top - readingLine)
        return distance < bestDistance ? section : best
      })

      setActiveId(nearest.id as NarrativeAnchorId)
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const intersecting = entries
          .filter((entry) => entry.isIntersecting)
          .map((entry) => entry.target as HTMLElement)

        if (intersecting.length > 0) {
          setNearestToReadingLine(intersecting)
        }
      },
      { rootMargin: '-42% 0px -42% 0px', threshold: 0 },
    )

    sections.forEach((section) => observer.observe(section))

    const initialFrame = window.requestAnimationFrame(() => {
      const readingLine = window.innerHeight * 0.48
      const passed = sections.filter(
        (section) => section.getBoundingClientRect().top <= readingLine,
      )
      setActiveId((passed.at(-1) ?? sections[0]).id as NarrativeAnchorId)
    })

    return () => {
      window.cancelAnimationFrame(initialFrame)
      observer.disconnect()
    }
  }, [])

  const activeIndex = narrativeAnchors.findIndex((anchor) => anchor.id === activeId)
  return (
    <nav
      className="flow-progress"
      aria-label="Project narrative"
      data-active-section={activeId}
    >
      <ol className="flow-progress__anchors">
        {narrativeAnchors.map((anchor, index) => {
          const distance = index - activeIndex
          const position =
            distance === 0 ? 'current' : Math.abs(distance) === 1 ? 'adjacent' : 'remote'

          return (
            <li
              key={anchor.id}
              className={`flow-progress__anchor flow-progress__anchor--${position}`}
              data-position={position}
            >
              <a
                href={`#${anchor.id}`}
                aria-current={position === 'current' ? 'location' : undefined}
                aria-label={`Go to ${anchor.navLabel.toLowerCase()}`}
              >
                <span className="flow-progress__index" aria-hidden="true">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <span className="flow-progress__label">{anchor.navLabel}</span>
              </a>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
