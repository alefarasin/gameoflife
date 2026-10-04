import { useEffect, useState } from "react";
import LineChart from "./LineChart";

type Run = { run_id: string; learning: string; started: number; reports: number; tick: number; last_seen: number };
type Metric = {
  id: number; tick: number; generation: number; population: number;
  avg_energy: number; avg_fitness: number; max_fitness: number;
};

const MAX_POINTS = 2000;

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [live, setLive] = useState(false);

  useEffect(() => {
    const load = () =>
      fetch("/api/runs").then((r) => r.json()).then((rs: Run[]) => {
        setRuns(rs);
        setRunId((cur) => cur ?? rs[0]?.run_id ?? null);
      }).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!runId) return;
    setMetrics([]);
    let es: EventSource | null = null;
    let cancelled = false;
    fetch(`/api/runs/${encodeURIComponent(runId)}/metrics`)
      .then((r) => r.json())
      .then((hist: Metric[]) => {
        if (cancelled) return;
        setMetrics(hist.slice(-MAX_POINTS));
        const last = hist.length ? hist[hist.length - 1].id : 0;
        es = new EventSource(`/api/runs/${encodeURIComponent(runId)}/stream?after_id=${last}`);
        es.onopen = () => setLive(true);
        es.onerror = () => setLive(false);
        es.onmessage = (e) => setMetrics((m) => [...m, JSON.parse(e.data) as Metric].slice(-MAX_POINTS));
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      es?.close();
      setLive(false);
    };
  }, [runId]);

  const last = metrics[metrics.length - 1];
  const xs = metrics.map((m) => m.tick);
  const col = (k: keyof Metric) => metrics.map((m) => m[k] as number);
  const run = runs.find((r) => r.run_id === runId);

  return (
    <main>
      <header>
        <h1>Game of Life – avanzamenti</h1>
        <select value={runId ?? ""} onChange={(e) => setRunId(e.target.value)}>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>{r.run_id} ({r.learning})</option>
          ))}
        </select>
        <span className={live ? "badge on" : "badge"}>{live ? "live" : "offline"}</span>
      </header>

      {!runs.length && <p>Nessuna run ancora: avvia il worker.</p>}

      {last && (
        <section className="stats">
          <Stat label="Tick" value={last.tick} />
          <Stat label="Generazione" value={last.generation} />
          <Stat label="Popolazione" value={last.population} />
          <Stat label="Fitness media" value={last.avg_fitness.toFixed(2)} />
          <Stat label="Fitness max" value={last.max_fitness.toFixed(2)} />
          <Stat label="Apprendimento" value={run?.learning ?? "-"} />
        </section>
      )}

      {last && (
        <section className="charts">
          <LineChart title="Popolazione" xs={xs} series={[{ label: "creature", color: "#2563eb", values: col("population") }]} />
          <LineChart title="Fitness" xs={xs} series={[
            { label: "media", color: "#16a34a", values: col("avg_fitness") },
            { label: "max", color: "#d97706", values: col("max_fitness") },
          ]} />
          <LineChart title="Energia media" xs={xs} series={[{ label: "energia", color: "#9333ea", values: col("avg_energy") }]} />
        </section>
      )}
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
