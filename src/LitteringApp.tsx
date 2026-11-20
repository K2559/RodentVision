import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { Film, Upload } from "lucide-react";
import type { GeminiEvent } from "./types";

type ReviewEvent = Omit<GeminiEvent, "score"> & { score: number | null; decision: "pending" | "confirmed" | "rejected" };
type Clip = { path: string; events?: ReviewEvent[]; reviewed?: boolean; analyzed?: boolean };
const names = {
  pedestrian_littering: "Pedestrian littering",
  vehicle_littering: "Vehicle littering",
  bird_feeding: "Bird feeding",
  dog_presence: "Dog occurrence",
};
const categories = Object.keys(names) as GeminiEvent["category"][];
const clipName = (path: string) => path.split(/[\\/]/).at(-1) ?? path;
const displayName = (path: string) => clipName(path).replace(/\.[^.]+$/, "");

function timeCode(seconds: number) {
  const [hours, minutes, remainder] = [
    Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, Math.floor(seconds) % 60,
  ].map(value => String(value).padStart(2, "0"));
  const fraction = (seconds % 1).toFixed(3).slice(1).replace(/\.?0+$/, "");
  return (hours === "00" ? minutes + ":" + remainder : hours + ":" + minutes + ":" + remainder) + fraction;
}

function reportTimes(clip: Clip) {
  const confirmed = (clip.events ?? []).filter(event => event.decision === "confirmed")
    .sort((a, b) => a.timestampSeconds - b.timestampSeconds)
    .map(event => timeCode(event.timestampSeconds) + (event.category === "dog_presence" ? " (dog)" : ""));
  if (!clip.reviewed) confirmed.push("REVIEW PENDING");
  return confirmed.join("\n") || "NA";
}

function csvCell(value: string) {
  const safe = /^[\s]*[=+\-@]|^[\t\r\n]/.test(value) ? "'" + value : value;
  return '"' + safe.replace(/"/g, '""') + '"';
}

export default function LitteringApp() {
  const bridge = window.rodentVision;
  const [clips, setClips] = useState<Clip[]>([]);
  const [selected, setSelected] = useState("");
  const [key, setKey] = useState("");
  const [configured, setConfigured] = useState(false);
  const [duration, setDuration] = useState(0);
  const [running, setRunning] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [report, setReport] = useState(false);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [manualCategory, setManualCategory] = useState<GeminiEvent["category"]>("pedestrian_littering");
  const videoRef = useRef<HTMLVideoElement>(null);
  const clip = clips.find(item => item.path === selected);

  useEffect(() => {
    if (!bridge) return;
    bridge.geminiConfigured().then(setConfigured, () => setError("Could not read Gemini configuration."));
    return bridge.onGeminiPhase(setStatus);
  }, [bridge]);

  function select(path: string) {
    if (path === selected) return;
    setSelected(path);
    setDuration(0);
  }

  function addPaths(paths: string[]) {
    const accepted = [...new Set(paths.filter(path => /\.(mp4|mov|mkv|avi|webm|m4v)$/i.test(path)))];
    setError(accepted.length ? "" : "Choose an MP4, MOV, MKV, AVI, WebM or M4V video.");
    if (!accepted.length) return;
    setClips(previous => [
      ...previous,
      ...accepted.filter(path => !previous.some(item => item.path === path)).map(path => ({ path })),
    ]);
    select(accepted[0]);
    setReport(false);
    setStatus("");
  }

  function drop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    if (!bridge || running) return;
    addPaths(Array.from(event.dataTransfer.files, file => bridge.getPathForFile(file)));
  }

  async function browse() {
    try {
      const paths = await bridge.chooseVideos();
      if (paths?.length) addPaths(paths);
    } catch (cause) {
      setError(String(cause));
    }
  }

  function updateClip(path: string, update: (current: Clip) => Clip) {
    setClips(items => items.map(item => item.path === path ? update(item) : item));
  }

  async function analyze() {
    if (!clip || !duration || running) return;
    setRunning(true);
    setError("");
    setStatus("Starting Gemini analysis");
    try {
      const result = await bridge.analyzeWithGemini({ video: clip.path, duration, key });
      if (result) {
        updateClip(clip.path, current => ({
          ...current,
          reviewed: false,
          events: [...(current.events ?? []), ...result.events.map(event => ({ ...event, decision: "pending" as const }))],
          analyzed: true,
        }));
      }
      setStatus(result ? "Analysis complete · " + result.events.length + " candidate events" : "Analysis cancelled.");
    } catch (cause) {
      setStatus("");
      setError(String(cause).replace(/^Error invoking remote method '[^']+': (?:Error: )?/, ""));
    } finally {
      setRunning(false);
    }
  }

  function decide(index: number, decision: ReviewEvent["decision"]) {
    if (!clip) return;
    updateClip(clip.path, current => ({
      ...current,
      reviewed: false,
      events: current.events?.map((event, i) => i === index ? { ...event, decision } : event),
    }));
  }

  function addEvent() {
    if (!clip || !videoRef.current || !duration) return;
    const timestampSeconds = Math.floor(videoRef.current.currentTime * 1000) / 1000;
    updateClip(clip.path, current => ({
      ...current,
      reviewed: false,
      events: [...(current.events ?? []), { timestampSeconds, category: manualCategory, score: null, decision: "confirmed" }],
    }));
  }

  const reportClips = [...clips].sort((a, b) => displayName(a.path).localeCompare(displayName(b.path), undefined, { numeric: true }));

  async function save(format: "csv" | "pdf") {
    setSaving(true);
    setError("");
    try {
      const rows = [
        ["Name of Video Clip", "Time of the littering act / dog occurrence"],
        ...reportClips.map(item => [displayName(item.path), reportTimes(item)]),
      ];
      const csv = rows.map(row => row.map(csvCell).join(",")).join("\r\n");
      const saved = await bridge.saveLitteringReport(format, format === "csv" ? csv : undefined);
      if (saved) setStatus("Saved " + clipName(saved));
    } catch (cause) {
      setError(String(cause));
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="app-shell littering-app"
      onDragOver={event => {
        if (!event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = running ? "none" : "copy";
        if (!running) setDragging(true);
      }}
      onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }}
      onDrop={drop}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Film size={17} /></div>
          <div><strong>RodentVision</strong><span>Littering event review</span></div>
        </div>
        <div className="topbar-actions">
          <button className="quiet-button" type="button" aria-pressed={!report} disabled={saving} onClick={() => setReport(false)}>Review</button>
          <button className="quiet-button" type="button" aria-pressed={report} disabled={saving || running} onClick={() => setReport(true)}>Report</button>
        </div>
      </header>

      <div className="workspace littering-workspace" hidden={report}>
        <aside className="config-panel">
          <section className="panel-section">
            <div className="section-heading"><h2>Source videos</h2></div>
            <button className="quiet-button" onClick={browse} disabled={!bridge || running}><Upload size={14} /> Choose videos</button>
            <div className="littering-clips">
              {clips.map(item => (
                <button key={item.path} type="button" aria-pressed={item.path === selected}
                  disabled={running} className="littering-clip" onClick={() => select(item.path)} title={item.path}>
                  <strong>{clipName(item.path)}</strong>
                  <small>{item.reviewed ? "Review complete" : item.events ? item.events.length + " candidates" : "Awaiting analysis"}</small>
                </button>
              ))}
            </div>
          </section>
          <section className="panel-section littering-config">
            <div className="section-heading"><h2>Gemini Decision 1</h2></div>
            {configured ? <p>API key configured in the environment.</p> : (
              <label>Gemini API key
                <input type="password" value={key} onChange={event => setKey(event.target.value)} disabled={running}
                  autoComplete="off" placeholder="Enter API key" />
              </label>
            )}
            <p>Analyze sends the selected video to Google Gemini.</p>
            {running ? (
              <button className="quiet-button" onClick={() => void bridge.cancelGemini()}>Cancel analysis</button>
            ) : (
              <button className="littering-primary" onClick={analyze}
                disabled={!bridge || !clip || !duration || duration > 450 || clip.analyzed || (!configured && !key.trim())}>
                Analyze with Gemini
              </button>
            )}
            {duration > 450 && <p>Use a clip of 7 minutes 30 seconds or less for Gemini analysis.</p>}
          </section>
        </aside>

        <section className="viewer-panel littering-viewer">
          <div className="viewer-toolbar"><strong>{clip ? clipName(clip.path) : "Source video"}</strong></div>
          <div className="media-stage">
            {clip && bridge ? (
              <video key={clip.path} ref={videoRef} className="video" controls playsInline preload="metadata"
                src={bridge.mediaUrl(clip.path)}
                onLoadedMetadata={event => {
                  const length = event.currentTarget.duration;
                  setDuration(Number.isFinite(length) ? length : 0);
                }}
                onError={() => { setDuration(0); setError("Could not play this clip. Try an MP4 encoded with H.264."); }} />
            ) : (
              <div className="video-drop">
                <div className="drop-icon"><Upload size={22} /></div>
                <strong>Drop videos here</strong>
                <span>or choose files from your computer</span>
              </div>
            )}
          </div>
        </section>

        <aside className="review-panel">
          <div className="review-header"><h2>Event candidates</h2></div>
          {clip ? <>
            {!clip.events?.length ? <p>{clip.analyzed ? "Gemini returned no candidates. Review the full clip before signing off." : "Candidate timestamps appear here after Gemini analysis."}</p> : (
              <ol className="littering-events">
                {clip.events.map((event, index) => (
                  <li key={index}>
                    <button className="quiet-button" onClick={() => {
                      if (videoRef.current) videoRef.current.currentTime = event.timestampSeconds;
                    }} title={event.score === null ? "Added during review" : "Gemini score " + event.score.toFixed(2)}>{timeCode(event.timestampSeconds)} · {names[event.category]}</button>
                    <label><span className="littering-sr">Decision for event {index + 1}</span>
                      <select value={event.decision} disabled={running}
                        onChange={change => decide(index, change.target.value as ReviewEvent["decision"])}>
                        <option value="pending">Pending</option>
                        <option value="confirmed">Confirm</option>
                        <option value="rejected">Reject</option>
                      </select>
                    </label>
                  </li>
                ))}
              </ol>
            )}
            <div className="littering-manual">
              <select value={manualCategory} aria-label="New event category" onChange={event => setManualCategory(event.target.value as GeminiEvent["category"])}>
                {categories.map(category => <option key={category} value={category}>{names[category]}</option>)}
              </select>
              <button className="quiet-button" type="button" disabled={running || !duration} onClick={addEvent}>Add at current time</button>
            </div>
            <label className="littering-signoff">
              <input type="checkbox" checked={Boolean(clip.reviewed)} disabled={running || !duration || clip.events?.some(event => event.decision === "pending")}
                onChange={event => {
                  const reviewed = event.target.checked;
                  updateClip(clip.path, current => ({ ...current, reviewed }));
                }} />
              Full video reviewed
            </label>
          </> : <p>Choose a source video to review its events.</p>}
        </aside>
      </div>

      {report && <section className="littering-report">
        <div className="littering-report-toolbar">
          <h1>Littering Act Report</h1>
          <div className="topbar-actions">
            <button className="quiet-button" disabled={!reportClips.length || !bridge || saving} onClick={() => void save("csv")}>Save CSV</button>
            <button className="littering-primary" disabled={!reportClips.length || !bridge || saving} onClick={() => void save("pdf")}>Save PDF</button>
          </div>
        </div>
        <table aria-label="Littering Act Report">
          <thead><tr><th scope="col">Name of Video Clip</th><th scope="col">Time of the littering act / dog occurrence</th></tr></thead>
          <tbody>{reportClips.length ? reportClips.map(item => (
            <tr key={item.path}><td>{displayName(item.path)}</td><td>{reportTimes(item)}</td></tr>
          )) : <tr><td colSpan={2}>Add video clips in Review.</td></tr>}</tbody>
        </table>
      </section>}

      <footer className="statusbar littering-status" role="status">
        {error ? <span className="footer-error">{error}</span> : <span>{status || (bridge ? "Ready" : "Launch the RodentVision desktop app for video analysis.")}</span>}
      </footer>
      {dragging && <div className="littering-drop-overlay" aria-hidden="true">Drop videos to add them</div>}
    </main>
  );
}
