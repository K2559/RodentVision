import type { LitteringCategory } from "./types";

export const eventNames = {
  pedestrian_littering: "Pedestrian littering",
  vehicle_littering: "Vehicle littering",
  bird_feeding: "Bird feeding",
  dog_presence: "Dog occurrence",
};

export type ReviewedTime = { timestampSeconds: number; category: LitteringCategory };

export const reviewCsvTemplate = [
  "clip_name,event_time,event_type",
  "1B,02:05,pedestrian_littering",
  "2B,NA,",
  "23B,04:38,dog_presence",
  "",
].join("\r\n");

export const clipName = (path: string) => path.split(/[\\/]/).at(-1) ?? path;
export const displayName = (path: string) => clipName(path).replace(/\.(mp4|mov|mkv|avi|webm|m4v)$/i, "");

function parseCsv(text: string) {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  let closed = false;
  text = text.replace(/^\uFEFF/, "");
  for (let i = 0; i <= text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === undefined) throw new Error("The CSV contains an unclosed quote.");
      if (char !== '"') field += char;
      else if (text[i + 1] === '"') {
        field += '"';
        i++;
      } else {
        quoted = false;
        closed = true;
      }
    } else if (char === "," || char === "\n" || char === "\r" || char === undefined) {
      row.push(field.trim());
      field = "";
      closed = false;
      if (char !== ",") {
        rows.push(row);
        row = [];
        if (char === "\r" && text[i + 1] === "\n") i++;
      }
    } else if (char === '"' && !field && !closed) {
      quoted = true;
    } else {
      if (closed || char === '"') throw new Error("The CSV contains an invalid quote.");
      field += char;
    }
  }
  return rows;
}

function parseTime(value: string) {
  if (/^\d+(?:\.\d{1,3})?$/.test(value)) return Number(value);
  if (!/^(?:\d{1,4}:)?\d{1,2}:[0-5]\d(?:\.\d{1,3})?$/.test(value)) return NaN;
  const parts = value.split(":").map(Number);
  if (parts.length === 3 && parts[1] >= 60) return NaN;
  return parts.reduce((total, part) => total * 60 + part, 0);
}

export function readHumanReview(text: string, clips: { path: string; duration?: number }[]) {
  const [header, ...rows] = parseCsv(text);
  if (header.join(",").toLowerCase() !== "clip_name,event_time,event_type") {
    throw new Error("Use the CSV columns clip_name, event_time, event_type in that order.");
  }
  const byName = new Map<string, typeof clips>();
  for (const clip of clips) {
    const name = displayName(clip.path).toLowerCase();
    const matches = byName.get(name) ?? [];
    matches.push(clip);
    byName.set(name, matches);
  }
  const updates = new Map<string, ReviewedTime[]>();
  const noEvents = new Set<string>();
  for (const [index, row] of rows.entries()) {
    if (row.every(value => !value)) continue;
    const line = "CSV row " + (index + 2);
    const [name, time, category] = row;
    if (row.length !== 3 || !name || !time || /[\\/]/.test(name)) {
      throw new Error(line + " needs a clip filename and time or NA.");
    }
    const matches = byName.get(displayName(name).toLowerCase());
    if (!matches?.length) throw new Error(line + " does not match a loaded clip: " + name);
    if (matches.length > 1) throw new Error(line + " matches multiple clips: " + name);
    const clip = matches[0];
    const events = updates.get(clip.path) ?? [];
    if (time.toUpperCase() === "NA") {
      if (category || updates.has(clip.path)) {
        throw new Error(line + " must use a single NA row with a blank event_type.");
      }
      noEvents.add(clip.path);
    } else {
      const timestampSeconds = parseTime(time);
      if (!Number.isFinite(timestampSeconds) || timestampSeconds > Number.MAX_SAFE_INTEGER / 1000
        || (clip.duration !== undefined && timestampSeconds >= clip.duration)) {
        throw new Error(line + " has an invalid time or a time outside the clip.");
      }
      if (!Object.hasOwn(eventNames, category)) throw new Error(line + " has an unknown event_type: " + category);
      if (noEvents.has(clip.path)) throw new Error(line + " mixes event times and NA for the same clip.");
      if (events.some(event => event.timestampSeconds === timestampSeconds && event.category === category)) {
        throw new Error(line + " repeats an event for the same clip.");
      }
      events.push({ timestampSeconds, category: category as LitteringCategory });
    }
    updates.set(clip.path, events);
  }
  if (!updates.size) throw new Error("The CSV does not contain any reviewed clips.");
  for (const events of updates.values()) events.sort((a, b) => a.timestampSeconds - b.timestampSeconds);
  return updates;
}
