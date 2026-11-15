export const RankComparison = () => {
  const metrics = [
    {
      title: "POPE",
      columns: [["POPE Accuracy"], ["Claude Opus", "Mod-POPE"], ["GPT-6.1 Sol", "Mod-POPE"]],
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
      columns: [["Strict ROPE"], ["Claude Opus", "Mod-ROPE"], ["GPT-6.1 Sol", "Mod-ROPE"]],
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
      columns: [["Raw CHAIR-I", "lower is better"], ["Claude Opus", "Mod-CHAIR F0.5"], ["GPT-6.1 Sol", "Mod-CHAIR F0.5"]],
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
      columns: [["Paper-style THRONE"], ["Claude Opus", "THRONE F0.5"], ["GPT-6.1 Sol", "THRONE F0.5"]],
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
      .rank-comparison__mobile th:not(:first-child), .rank-comparison__mobile td:not(:first-child) { text-align: center; }
    }
  `;

  const rankY = (rank) => 64 + (rank - 1) * 22;
  const xs = [112, 300, 488];

  return (
    <>
      <style>{style}</style>
      <div className="rank-comparison">
        <p className="rank-comparison__legend"><span className="rank-comparison__legend-mark" aria-hidden="true" />Blue lines moved two or more ranks between columns.</p>
        {metrics.map((metric) => {
          const slug = metric.title.toLowerCase();
          return (
            <section className="rank-comparison__metric" key={metric.title}>
              <h4>{metric.title}</h4>
              <svg
                className="rank-comparison__chart"
                viewBox="0 0 760 264"
                role="img"
                aria-labelledby={`rank-title-${slug} rank-description-${slug}`}
              >
                <title id={`rank-title-${slug}`}>{metric.title} model rankings</title>
                <desc id={`rank-description-${slug}`}>Rankings by original metric, Claude Opus, and GPT-6.1 Sol. Rank 1 is best. Lines connect the same model across columns.</desc>
                {Array.from({ length: 9 }, (_, index) => {
                  const rank = index + 1;
                  const y = rankY(rank);
                  return (
                    <g key={`guide-${rank}`}>
                      <line className="rank-guide" x1="68" y1={y} x2="488" y2={y} />
                      <text className="rank-muted" x="46" y={y + 4} textAnchor="end" fontSize="13">{rank}</text>
                    </g>
                  );
                })}
                {xs.map((x, columnIndex) => (
                  <g key={`column-${columnIndex}`}>
                    <line className="rank-axis" x1={x} y1="54" x2={x} y2={rankY(9) + 8} />
                    <text x={x} y="18" textAnchor="middle" fontSize="13" fontWeight="500">
                      {metric.columns[columnIndex].map((line, lineIndex) => (
                        <tspan key={line} x={x} dy={lineIndex === 0 ? 0 : 16}>
                          {line}
                        </tspan>
                      ))}
                    </text>
                  </g>
                ))}
                {metric.rows.map(([model, original, opus, sol]) => {
                  const ranks = [original, opus, sol];
                  const moved = Math.max(
                    Math.abs(original - opus),
                    Math.abs(original - sol),
                    Math.abs(opus - sol),
                  ) >= 2;
                  const points = ranks.map((rank, index) => `${xs[index]},${rankY(rank)}`).join(" ");
                  return (
                    <g key={model}>
                      <polyline className={moved ? "rank-flow rank-flow--moved" : "rank-flow"} points={points} />
                      {ranks.map((rank, index) => (
                        <circle
                          key={`${model}-${index}`}
                          className={moved ? "rank-dot rank-dot--moved" : "rank-dot"}
                          cx={xs[index]}
                          cy={rankY(rank)}
                          r={moved ? 3.5 : 3}
                        />
                      ))}
                      <text x="508" y={rankY(sol) + 4} fontSize="13">{model}</text>
                    </g>
                  );
                })}
              </svg>
              <div className="rank-comparison__mobile">
                <table>
                  <thead>
                    <tr><th>Model</th><th>Original</th><th>Opus</th><th>Sol</th></tr>
                  </thead>
                  <tbody>
                    {metric.rows.map(([model, original, opus, sol]) => (
                      <tr key={model}><td>{model}</td><td>{original}</td><td>{opus}</td><td>{sol}</td></tr>
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
