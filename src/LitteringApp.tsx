import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";
import { Film, Settings, Upload } from "lucide-react";
import type { GeminiEvent, GeminiSettings } from "./types";
import { clipName, displayName, eventNames as names, readHumanReview, reviewCsvTemplate } from "./human-review";
import type { ReviewedTime } from "./human-review";
import LitteringReport from "./LitteringReport";

type ReviewEvent = Omit<GeminiEvent, "score"> & { score: number | null; decision: "pending" | "confirmed" | "rejected" };
type Clip = {
  path: string;
  duration?: number;
  events?: ReviewEvent[];
  reviewed?: boolean;
  analyzed?: boolean;
  imported?: ReviewedTime[];
};
const defaultModel = "gemini-decision-1";
const modelIdPattern = /^gemini-decision-[a-z0-9]+(?:-[a-z0-9]+)*$/;
const settingsKey = "rodentvision.littering.model.v1";
const categories = Object.keys(names) as GeminiEvent["category"][];

function loadSettings(): GeminiSettings {
  try {
    const saved = JSON.parse(localStorage.getItem(settingsKey) ?? "{}");
    return {
      model: typeof saved.model === "string" && modelIdPattern.test(saved.model) ? saved.model : defaultModel,
      resolution: saved.resolution === "low" ? "low" : "high",
    };
  } catch {
    return { model: defaultModel, resolution: "high" };
  }
}

function timeCode(seconds: number) {
  const milliseconds = Math.round(seconds * 1000);
  const wholeSeconds = Math.floor(milliseconds / 1000);
  const [hours, minutes, remainder] = [
    Math.floor(wholeSeconds / 3600), Math.floor(wholeSeconds / 60) % 60, wholeSeconds % 60,
  ].map(value => String(value).padStart(2, "0"));
  const fraction = String(milliseconds % 1000).padStart(3, "0").replace(/0+$/, "");
  const time = hours === "00" ? minutes + ":" + remainder : hours + ":" + minutes + ":" + remainder;
  return fraction ? time + "." + fraction : time;
}

function reportTimes(clip: Clip) {
  const events = clip.imported ?? (clip.events ?? []).filter(event => event.decision === "confirmed");
  const confirmed = [...events]
    .sort((a, b) => a.timestampSeconds - b.timestampSeconds)
    .map(event => timeCode(event.timestampSeconds) + (event.category === "dog_presence" ? " (dog)" : ""));
  if (!clip.reviewed && !clip.imported) confirmed.push("REVIEW PENDING");
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
  const [draftKey, setDraftKey] = useState("");
  const [configured, setConfigured] = useState(false);
  const [settings, setSettings] = useState(loadSettings);
  const [draft, setDraft] = useState(settings);
  const [running, setRunning] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [report, setReport] = useState(false);
  const [fitPage, setFitPage] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [saving, setSaving] = useState(false);
  const [importing, setImporting] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [manualCategory, setManualCategory] = useState<GeminiEvent["category"]>("pedestrian_littering");
  const videoRef = useRef<HTMLVideoElement>(null);
  const csvRef = useRef<HTMLInputElement>(null);
  const clip = clips.find(item => item.path === selected);
  const duration = clip?.duration ?? 0;
  const busy = running || importing || saving;

  useEffect(() => {
    if (!bridge) return;
    bridge.geminiConfigured().then(setConfigured, () => setError("Could not read Gemini configuration."));
    return bridge.onGeminiPhase(setStatus);
  }, [bridge]);

  function select(path: string) {
    setSelected(path);
  }

  function openSettings() {
    videoRef.current?.pause();
    setDraft(settings);
    setDraftKey(key);
    setShowSettings(true);
  }

  function applySettings() {
    const model = draft.model.trim();
    if (!modelIdPattern.test(model) || model.length > 80) {
      setError("Enter a Gemini Decision model ID, for example gemini-decision-1.");
      return;
    }
    const updated = { model, resolution: draft.resolution };
    setSettings(updated);
    setKey(draftKey.trim());
    setError("");
    setStatus("Settings saved.");
    setShowSettings(false);
    try {
      localStorage.setItem(settingsKey, JSON.stringify(updated));
    } catch {
      setStatus("Settings applied for this session.");
    }
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
    if (!bridge || busy || showSettings) return;
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
    if (!clip || !duration || busy || clip.imported) return;
    setRunning(true);
    setError("");
    setStatus("Starting Gemini analysis");
    try {
      const result = await bridge.analyzeWithGemini({ video: clip.path, duration, key, ...settings });
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

  const reportRows = useMemo(() => [...clips]
    .sort((a, b) => displayName(a.path).localeCompare(displayName(b.path), undefined, { numeric: true }))
    .map(item => ({ id: item.path, clip: displayName(item.path), times: reportTimes(item) })), [clips]);

  async function importReview(file: File) {
    setImporting(true);
    setError("");
    setStatus("");
    try {
      const updates = readHumanReview(await file.text(), clips);
      setClips(items => items.map(item => {
        const reviewed = updates.get(item.path);
        return reviewed ? { ...item, imported: reviewed } : item;
      }));
      setStatus("Applied human review to " + updates.size + (updates.size === 1 ? " clip." : " clips."));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setImporting(false);
    }
  }

  async function save(format: "csv" | "pdf") {
    setSaving(true);
    setError("");
    try {
      const rows = [
        ["Name of Video Clip", "Time of the littering act / dog occurrence"],
        ...reportRows.map(row => [row.clip, row.times]),
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
        event.dataTransfer.dropEffect = busy || showSettings ? "none" : "copy";
        if (!busy && !showSettings) setDragging(true);
      }}
      onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false); }}
      onDrop={drop}>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Film size={17} /></div>
          <div><strong>RodentVision</strong><span>Littering event review</span></div>
        </div>
        <div className="topbar-actions">
          <button className="quiet-button" type="button" aria-pressed={!report && !showSettings} disabled={busy} onClick={() => { setReport(false); setShowSettings(false); }}>Review</button>
          <button className="quiet-button" type="button" aria-pressed={report && !showSettings} disabled={busy} onClick={() => {
            videoRef.current?.pause();
            setReport(true);
            setShowSettings(false);
          }}>Report</button>
          <button className="quiet-button" type="button" aria-pressed={showSettings} disabled={busy || showSettings} onClick={openSettings}><Settings size={15} /> Settings</button>
        </div>
      </header>

      {showSettings && <section className="littering-settings">
        <button className="quiet-button" onClick={() => setShowSettings(false)}>Back to {report ? "Report" : "Review"}</button>
        <h1>Settings</h1>
        <form onSubmit={event => { event.preventDefault(); applySettings(); }}>
          <fieldset>
            <legend>Gemini API</legend>
            {configured ? <p>Using the GEMINI_API_KEY environment variable.</p> : (
              <label>Gemini API key
                <input type="password" value={draftKey} onChange={event => setDraftKey(event.target.value)} autoComplete="off" />
                <small>The key entered here is used for this session.</small>
              </label>
            )}
          </fieldset>
          <fieldset>
            <legend>Analysis model</legend>
            <label>Gemini Decision model ID
              <input value={draft.model} maxLength={80} onChange={event => setDraft({ ...draft, model: event.target.value })} required />
            </label>
            <label>Video resolution
              <select value={draft.resolution} onChange={event => setDraft({ ...draft, resolution: event.target.value === "low" ? "low" : "high" })}>
                <option value="high">High</option>
                <option value="low">Low</option>
              </select>
            </label>
          </fieldset>
          <button className="littering-primary" type="submit">Save settings</button>
        </form>
      </section>}

      <div className="workspace littering-workspace" hidden={report || showSettings}>
        <aside className="config-panel">
          <section className="panel-section">
            <div className="section-heading"><h2>Source videos</h2></div>
            <button className="quiet-button" onClick={browse} disabled={!bridge || running}><Upload size={14} /> Choose videos</button>
            <div className="littering-clips">
              {clips.map(item => (
                <button key={item.path} type="button" aria-pressed={item.path === selected}
                  disabled={running} className="littering-clip" onClick={() => select(item.path)} title={item.path}>
                  <strong>{clipName(item.path)}</strong>
                  <small>{item.imported ? "Human review imported" : item.reviewed ? "Review complete" : item.events ? item.events.length + " candidates" : "Awaiting analysis"}</small>
                </button>
              ))}
            </div>
          </section>
          <section className="panel-section littering-config">
            <div className="section-heading"><h2>Video analysis</h2></div>
            <p>{settings.model} · {settings.resolution === "high" ? "High" : "Low"} resolution</p>
            {!configured && !key.trim() && <button className="quiet-button" type="button" onClick={openSettings}>Set up Gemini</button>}
            <p>Analyze sends the selected video to Google Gemini.</p>
            {running ? (
              <button className="quiet-button" onClick={() => void bridge.cancelGemini()}>Cancel analysis</button>
            ) : (
              <button className="littering-primary" onClick={analyze}
                disabled={!bridge || !clip || !duration || duration > 450 || clip.analyzed || Boolean(clip.imported) || (!configured && !key.trim())}>
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
                  if (Number.isFinite(length) && length > 0) updateClip(clip.path, current => ({ ...current, duration: length }));
                }}
                onError={() => {
                  updateClip(clip.path, current => ({ ...current, duration: undefined }));
                  setError("Could not play this clip. Try an MP4 encoded with H.264.");
                }} />
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
          <div className="review-header"><h2>{clip?.imported ? "Imported human review" : "Event candidates"}</h2></div>
          {clip?.imported ? <>
            {clip.imported.length ? <ol className="littering-events">
              {clip.imported.map(event => <li key={event.category + ":" + event.timestampSeconds}>
                <button className="quiet-button" onClick={() => {
                  if (videoRef.current) videoRef.current.currentTime = event.timestampSeconds;
                }}>{timeCode(event.timestampSeconds)} · {names[event.category]}</button>
              </li>)}
            </ol> : <p>NA · No events in the imported review.</p>}
            <button className="quiet-button" type="button" onClick={() => {
              updateClip(clip.path, current => ({ ...current, imported: undefined }));
              setStatus("Restored the prior review for " + clipName(clip.path));
            }}>Restore prior review</button>
          </> : clip ? <>
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

      {report && !showSettings && <section className="littering-report">
        <div className="littering-report-toolbar">
          <strong>A4 preview</strong>
          <div className="topbar-actions">
            <label className="report-zoom">Zoom
              <select value={fitPage ? "fit" : "100"} onChange={event => setFitPage(event.target.value === "fit")}>
                <option value="fit">Fit page</option>
                <option value="100">100%</option>
              </select>
            </label>
            <a className="quiet-button" download="human-review-template.csv" href={"data:text/csv;charset=utf-8," + encodeURIComponent(reviewCsvTemplate)}>CSV template</a>
            <button className="quiet-button" disabled={!clips.length || busy} onClick={() => csvRef.current?.click()}>Import reviewed CSV</button>
            <input hidden ref={csvRef} type="file" accept=".csv,text/csv" onChange={event => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void importReview(file);
            }} />
            <button className="quiet-button" disabled={!reportRows.length || !bridge || busy} onClick={() => void save("csv")}>Save CSV</button>
            <button className="littering-primary" disabled={!reportRows.length || !bridge || busy} onClick={() => void save("pdf")}>Save PDF</button>
          </div>
        </div>
        <LitteringReport rows={reportRows} fitPage={fitPage} />
      </section>}

      <footer className="statusbar littering-status" role="status">
        {error ? <span className="footer-error">{error}</span> : <span>{status || (bridge ? "Ready" : "Launch the RodentVision desktop app for video analysis.")}</span>}
      </footer>
      {dragging && <div className="littering-drop-overlay" aria-hidden="true">Drop videos to add them</div>}
    </main>
  );
}
