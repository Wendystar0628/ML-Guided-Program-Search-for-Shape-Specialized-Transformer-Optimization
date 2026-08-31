import { useEffect, useRef, useState } from 'react'
import { headline, performance } from '../data/projectData'

const speedupRange = performance.reduce(
  (range, point) => ({
    min: Math.min(range.min, point.speedup),
    max: Math.max(range.max, point.speedup),
  }),
  { min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY },
)

export function EvidenceAggregate() {
  const aggregateRef = useRef<HTMLElement>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const aggregate = aggregateRef.current
    if (!aggregate) return

    const observer = new IntersectionObserver(
      ([entry]) => setIsVisible(entry.isIntersecting),
      { threshold: 0.45 },
    )

    observer.observe(aggregate)
    return () => observer.disconnect()
  }, [])

  return (
    <aside
      ref={aggregateRef}
      className="evidence-aggregate"
      data-reveal={isVisible ? 'visible' : 'waiting'}
      aria-label="Aggregate of Shapes 01 through 13"
    >
      <span className="evidence-aggregate__convergence" aria-hidden="true" />
      <p className="evidence-aggregate__source">SHAPES 01–13 · EQUAL LOG WEIGHT</p>

      <div className="evidence-aggregate__geomean">
        <strong>{headline.geomean.toFixed(2)}×</strong>
        <span>EQUAL-SHAPE GEOMEAN</span>
      </div>

      <div className="evidence-aggregate__range">
        <strong>{speedupRange.min.toFixed(2)}×—{speedupRange.max.toFixed(2)}×</strong>
        <span>MEASURED SPEEDUP RANGE</span>
      </div>

      <div className="evidence-aggregate__correctness">
        <strong>
          {headline.residentPassed} / {headline.residentTotal}
        </strong>
        <span>RESIDENT SHAPES PASS</span>
        <small>SUPPLIED COMPARATOR</small>
      </div>

      <p className="evidence-aggregate__shape14" role="note">
        S14 · STREAMED EXECUTION · EXCLUDED FROM GEOMEAN
      </p>
    </aside>
  )
}
