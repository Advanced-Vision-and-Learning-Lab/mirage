export const RankComparison = ({ metric: onlyMetric } = {}) => {
  const metrics = [
    {
      title: "POPE",
      columns: [["POPE Accuracy"], ["Claude Opus 5.5", "Mod-POPE"], ["GPT-6.1 Sol", "Mod-POPE"]],
      rows: [
        ["Gemma-4-12B", 1, 1, 1],
        ["Qwen2-VL-7B", 2, 3, 3],
        ["Gemma-4-E4B", 3, 2, 2],
        ["Qwen2-VL-2B", 4, 4, 6],
        ["Muse-Glimmer-30B", 5, 7, 9],
        ["PaliGemma-3B", 6, 5, 5],
        ["Aya-Vision-8B", 7, 6, 4],
        ["Ministral-3-8B", 8, 8, 7],
        ["Ministral-3-14B", 9, 9, 8],
      ],
    },
    {
      title: "ROPE",
      columns: [["Strict ROPE"], ["Claude Opus 5.5", "Mod-ROPE"], ["GPT-6.1 Sol", "Mod-ROPE"]],
      rows: [
        ["Gemma-4-12B", 1, 1, 2],
        ["Qwen2-VL-7B", 2, 3, 3],
        ["Gemma-4-E4B", 3, 4, 4],
        ["Aya-Vision-8B", 4, 5, 6],
        ["Qwen2-VL-2B", 5, 6, 5],
        ["PaliGemma-3B", 6, 8, 8],
        ["Muse-Glimmer-30B", 7, 2, 1],
        ["Ministral-3-14B", 8, 7, 7],
        ["Ministral-3-8B", 9, 9, 9],
      ],
    },
    {
      title: "CHAIR",
      columns: [["Raw CHAIR-I", "lower is better"], ["Claude Opus 5.5", "Mod-CHAIR F0.5"], ["GPT-6.1 Sol", "Mod-CHAIR F0.5"]],
      rows: [
        ["Gemma-4-12B", 4, 4, 4],
        ["Qwen2-VL-7B", 2, 9, 9],
        ["Gemma-4-E4B", 6, 6, 6],
        ["Qwen2-VL-2B", 3, 8, 8],
        ["Muse-Glimmer-30B", 5, 1, 1],
        ["PaliGemma-3B", 1, 7, 7],
        ["Aya-Vision-8B", 7, 2, 2],
        ["Ministral-3-8B", 9, 3, 3],
        ["Ministral-3-14B", 8, 5, 5],
      ],
    },
    {
      title: "THRONE",
      columns: [["Paper-style THRONE"], ["Claude Opus 5.5", "THRONE F0.5"], ["GPT-6.1 Sol", "THRONE F0.5"]],
      rows: [
        ["Gemma-4-12B", 2, 3, 3],
        ["Qwen2-VL-7B", 8, 8, 8],
        ["Gemma-4-E4B", 6, 5, 5],
        ["Qwen2-VL-2B", 4, 6, 6],
        ["Muse-Glimmer-30B", 1, 1, 1],
        ["PaliGemma-3B", 9, 9, 9],
        ["Aya-Vision-8B", 3, 2, 2],
        ["Ministral-3-8B", 7, 7, 7],
        ["Ministral-3-14B", 5, 4, 4],
      ],
    },
  ];

  const style = `
    .rank-comparison { margin: 1rem 0; }
    .rank-comparison__metric { padding: 1.1rem 0; }
    .rank-comparison__metric + .rank-comparison__metric { border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent); }
    .rank-comparison__metric h4 { margin: 0 0 .4rem; font-size: 1rem; }
    .rank-comparison__chart { display: block; width: 100%; height: auto; color: inherit; }
    .rank-comparison__chart text { fill: currentColor; font-family: inherit; }
    .rank-comparison__chart .rank-muted { opacity: .68; }
    .rank-comparison__chart .rank-guide { stroke: currentColor; stroke-width: 1; opacity: .15; }
    .rank-comparison__chart .rank-axis { stroke: currentColor; stroke-width: 1; opacity: .42; }
    .rank-comparison__chart .rank-flow { fill: none; stroke: currentColor; stroke-width: 1.5; opacity: .4; }
    .rank-comparison__chart .rank-flow--moved { stroke: #5369cf; stroke-width: 2.5; opacity: 1; }
    .rank-comparison__chart .rank-dot { fill: currentColor; opacity: .62; }
    .rank-comparison__chart .rank-dot--moved { fill: #5369cf; opacity: 1; }
    .rank-comparison__legend { display: flex; align-items: center; gap: .5rem; margin: .2rem 0 .5rem; font-size: .85rem; opacity: .78; }
    .rank-comparison__legend-mark { width: 1.4rem; border-top: 2px solid #5369cf; }
    .rank-comparison__mobile { display: none; }
    @media (max-width: 600px) {
      .rank-comparison__chart { display: none; }
      .rank-comparison__mobile { display: block; }
      .rank-comparison__mobile table { width: 100%; font-size: .8rem; }
    }
  `;

  const shownMetrics = onlyMetric ? metrics.filter(({ title }) => title === onlyMetric) : metrics;
  const rankY = (rank) => 64 + (rank - 1) * 22;
  const xs = [290, 350, 650, 700];

  return (
    <>
      <style>{style}</style>
      <div className="rank-comparison">
        <p className="rank-comparison__legend"><span className="rank-comparison__legend-mark" aria-hidden="true" />Blue links moved two or more ranks between adjacent columns.</p>
        {shownMetrics.map((metric) => {
          const slug = metric.title.toLowerCase();
          return (
            <section className="rank-comparison__metric" key={metric.title}>
              <h4>{metric.title}</h4>
              <svg
                className="rank-comparison__chart"
                viewBox="0 0 1000 264"
                role="img"
                aria-labelledby={`rank-title-${slug} rank-description-${slug}`}
              >
                <title id={`rank-title-${slug}`}>{metric.title} model rankings</title>
                <desc id={`rank-description-${slug}`}>Three independent rank columns: original metric, Claude Opus 5.5 Mod metric, and GPT-6.1 Sol Mod metric. Rank 1 is best. Links connect the same model between adjacent columns.</desc>
                {Array.from({ length: 9 }, (_, index) => {
                  const rank = index + 1;
                  const y = rankY(rank);
                  return (
                    <g key={`guide-${rank}`}>
                      <line className="rank-guide" x1="0" y1={y} x2="1000" y2={y} />
                    </g>
                  );
                })}
                {[
                  { x: 140, labelX: 12 },
                  { x: 500, labelX: 365 },
                  { x: 860, labelX: 715 },
                ].map(({ x, labelX }, columnIndex) => {
                  const sorted = metric.rows
                    .map(([model, ...values]) => ({ model, rank: values[columnIndex] }))
                    .sort((a, b) => a.rank - b.rank);
                  const nodeX = columnIndex === 0 ? xs[0] : columnIndex === 1 ? xs[1] : xs[3];
                  const rightNodeX = columnIndex === 1 ? xs[2] : null;
                  return (
                    <g key={`column-${columnIndex}`}>
                      <text x={x} y="18" textAnchor="middle" fontSize="13" fontWeight="500">
                        {metric.columns[columnIndex].map((line, lineIndex) => (
                          <tspan key={line} x={x} dy={lineIndex === 0 ? 0 : 16}>{line}</tspan>
                        ))}
                      </text>
                      {sorted.map(({ model, rank }) => (
                        <g key={`${columnIndex}-${model}`}>
                          <circle className="rank-dot" cx={nodeX} cy={rankY(rank)} r="3" />
                          {rightNodeX !== null && <circle className="rank-dot" cx={rightNodeX} cy={rankY(rank)} r="3" />}
                          <text x={labelX} y={rankY(rank) + 4} fontSize="13">{rank}. {model}</text>
                        </g>
                      ))}
                    </g>
                  );
                })}
                {metric.rows.map(([model, original, opus, sol]) => {
                  const firstMoved = Math.abs(original - opus) >= 2;
                  const secondMoved = Math.abs(opus - sol) >= 2;
                  return (
                    <g key={model}>
                      <line className={firstMoved ? "rank-flow rank-flow--moved" : "rank-flow"} x1={xs[0]} y1={rankY(original)} x2={xs[1]} y2={rankY(opus)} />
                      <line className={secondMoved ? "rank-flow rank-flow--moved" : "rank-flow"} x1={xs[2]} y1={rankY(opus)} x2={xs[3]} y2={rankY(sol)} />
                    </g>
                  );
                })}
              </svg>
              <div className="rank-comparison__mobile">
                <table>
                  <thead>
                    <tr><th>Rank</th><th>{metric.columns[0].join(" ")}</th><th>{metric.columns[1].join(" ")}</th><th>{metric.columns[2].join(" ")}</th></tr>
                  </thead>
                  <tbody>
                    {Array.from({ length: 9 }, (_, index) => index + 1).map((rank) => (
                      <tr key={rank}>
                        <td>{rank}</td>
                        {[0, 1, 2].map((columnIndex) => {
                          const [model] = metric.rows.find((row) => row[columnIndex + 1] === rank);
                          return <td key={columnIndex}>{model}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          );
        })}
      </div>
    </>
  );
};
