import { headline, performance, shape14Note } from '../data/projectData'

const speedupRange = performance.reduce(
  (range, point) => ({
    min: Math.min(range.min, point.speedup),
    max: Math.max(range.max, point.speedup),
  }),
  { min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY },
)

export function EvidenceAggregate() {
  return (
    <aside className="evidence-aggregate" aria-label="Aggregate of resident Shapes 01 through 13">
      <p className="evidence-aggregate__source">AGGREGATE SPEEDUP · SHAPES 01–13</p>

      <dl className="evidence-aggregate__metrics">
        <div className="evidence-aggregate__metric evidence-aggregate__metric--primary">
          <dd>{headline.geomean.toFixed(2)}×</dd>
          <dt>GEOMETRIC MEAN SPEEDUP</dt>
        </div>
        <div className="evidence-aggregate__metric">
          <dd>
            {speedupRange.min.toFixed(2)}×—{speedupRange.max.toFixed(2)}×
          </dd>
          <dt>MINIMUM—MAXIMUM SPEEDUP</dt>
        </div>
        <div className="evidence-aggregate__metric">
          <dd>
            {headline.residentPassed} / {headline.residentTotal}
          </dd>
          <dt>SHAPES PASSING THE SUPPLIED COMPARATOR</dt>
        </div>
      </dl>

      <p className="evidence-aggregate__shape14" role="note">
        {shape14Note}
      </p>
    </aside>
  )
}
