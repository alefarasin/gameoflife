export type Series = { label: string; color: string; values: number[] };

const W = 520, H = 200, PAD = { l: 48, r: 12, t: 10, b: 22 };

export default function LineChart({ title, xs, series }: { title: string; xs: number[]; series: Series[] }) {
  const all = series.flatMap((s) => s.values);
  const ymin = all.length ? Math.min(...all) : 0;
  const ymax = all.length ? Math.max(...all) : 1;
  const span = ymax - ymin || 1;
  const xmin = xs[0] ?? 0;
  const xspan = (xs[xs.length - 1] ?? 1) - xmin || 1;
  const px = (x: number) => PAD.l + ((x - xmin) / xspan) * (W - PAD.l - PAD.r);
  const py = (y: number) => H - PAD.b - ((y - ymin) / span) * (H - PAD.t - PAD.b);
  const fmt = (n: number) => (Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(2));

  return (
    <figure className="chart">
      <figcaption>
        {title}
        {series.map((s) => (
          <span key={s.label} className="legend" style={{ color: s.color }}>● {s.label}</span>
        ))}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} className="axis" />
        <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={H - PAD.b} className="axis" />
        <text x={PAD.l - 6} y={PAD.t + 8} textAnchor="end" className="tick">{fmt(ymax)}</text>
        <text x={PAD.l - 6} y={H - PAD.b} textAnchor="end" className="tick">{fmt(ymin)}</text>
        <text x={PAD.l} y={H - 6} className="tick">{xmin}</text>
        <text x={W - PAD.r} y={H - 6} textAnchor="end" className="tick">{xs[xs.length - 1] ?? ""}</text>
        {series.map((s) => (
          <polyline
            key={s.label}
            fill="none"
            stroke={s.color}
            strokeWidth={1.8}
            points={s.values.map((v, i) => `${px(xs[i])},${py(v)}`).join(" ")}
          />
        ))}
      </svg>
    </figure>
  );
}
