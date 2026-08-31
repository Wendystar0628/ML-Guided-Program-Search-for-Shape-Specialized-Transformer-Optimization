import { WorkloadFigure } from '../visuals/WorkloadFigure'

const dimensions = ['B', 'S', 'D', 'H', 'F'] as const

export function WorkloadScene() {
  return (
    <section
      id="workloads"
      className="scene scene--workload workload-scene"
      aria-labelledby="workload-heading"
    >
      <header className="scene__header workload-scene__header">
        <div className="scene__eyebrow">02 · WORKLOAD DIVERSITY</div>
        <h2 id="workload-heading" className="scene__title workload-scene__title">
          NO UNIVERSAL FAST PATH
        </h2>
        <div className="workload-dimensions" aria-label="Workload dimensions">
          <span className="workload-dimensions__count">13 RESIDENT SHAPES</span>
          {dimensions.map((dimension) => (
            <span key={dimension} className="workload-dimensions__item">
              {dimension}
            </span>
          ))}
        </div>
      </header>

      <div className="workload-scene__visual">
        <WorkloadFigure />
      </div>
    </section>
  )
}
