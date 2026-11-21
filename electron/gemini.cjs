const fs = require("node:fs");
const { open } = require("node:fs/promises");
const { setTimeout: pause } = require("node:timers/promises");

const BASE = "https://generativelanguage.googleapis.com";
const MODEL = "gemini-decision-1";
const MODEL_ID = /^gemini-decision-[a-z0-9]+(?:-[a-z0-9]+)*$/;
const RESOLUTIONS = new Set(["high", "low"]);
const MIN_EVIDENCE = .5;
const STRONG_NORMAL_ACTION = .75;
const MIME = { ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/mov", ".avi": "video/avi", ".webm": "video/webm", ".mkv": "video/x-matroska" };

const rubric = [
  "Inspect the full chronological CCTV clip for new littering, deliberate bird feeding, and recognizable dog occurrence.",
  "Require a causal link between a person or vehicle and a new release or abandoned deposit. Exclude ordinary movement, collection, proper bin disposal, old debris, and shadows.",
  "Use before, action, and after together. Look for material transfer from the source to newly occupied ground. An obscured drop may qualify if the causal sequence is visible. A gesture, nearby vehicle, animal, or an unidentifiable object alone does not establish deposition.",
  "Timestamp littering at first ground contact and dog occurrence at the first moment the dog is within view and recognizable, even if partly occluded. Use elapsed seconds from the clip start. Assess every question independently and report each distinct event once.",
].join(" ");

function buildQuestions(duration) {
  const count = Math.ceil(duration);
  const questions = {};
  const windows = [];
  for (let second = 0; second < count; second++) {
    questions["release_" + second] = {
      type: "noul",
      instructions: "Does newly released or abandoned material first reach the ground nearest elapsed second " + second + "? Require before/action/after evidence of source-to-ground transfer. Exclude retrieval, bin use, old debris, continued presence and departure.",
    };
    questions["dog_" + second] = {
      type: "noul",
      instructions: "Does a recognizable dog first come fully into the camera field of view nearest elapsed second " + second + "? Do not count continued presence.",
    };
  }
  for (let start = 0; start < count; start += 24) {
    const end = Math.min(start + 30, count);
    const prefix = "window_" + start;
    const scope = "In elapsed seconds [" + start + ", " + end + "), using full-clip context, ";
    windows.push({ start, end, prefix });
    questions[prefix + "_physical"] = {
      type: "choice", instructions: scope + "which physical interpretation has evidence?",
      criteria: {
        new_deposit: "A person or vehicle causally released or abandoned new material on the ground, including deliberate bird feeding.",
        handling: "Collection, temporary handling, rehandling, or proper bin disposal.",
        unresolved: "Ordinary activity, previously present debris, or insufficient evidence.",
      },
    };
    questions[prefix + "_confirm"] = { type: "noul", instructions: scope + "is a new deposit supported by a linked source, release or placement, and a new deposit or release path? Use what was present before and what changed after." };
    questions[prefix + "_category"] = {
      type: "choice", instructions: scope + "classify the strongest supported deposition.",
      criteria: {
        pedestrian_littering: "A person newly discards material onto the ground.",
        vehicle_littering: "Material discarded directly from a vehicle or by its occupant.",
        bird_feeding: "A person deliberately scatters or places food for birds.",
        unresolved: "No supported category.",
      },
    };
    questions[prefix + "_normal"] = { type: "noul", instructions: scope + "is the apparent disposal positively explained by ordinary motion, collection, temporary handling, bin use, old debris revealed after occlusion, or a visual artifact, with no new deposition? Insufficient evidence by itself is not a positive explanation." };
    questions[prefix + "_dog"] = { type: "noul", instructions: scope + "does a recognizable dog first come fully into the field of view, allowing partial occlusion? Exclude continued presence, humans, other animals, and shadows." };
    if (end === count) break;
  }
  return { questions, windows };
}

function decode(response, questions, windows) {
  const answers = response?.answers;
  const probability = value => typeof value === "number" && value >= 0 && value <= 1;
  if (!answers || Object.keys(answers).length !== Object.keys(questions).length) {
    throw new Error("Gemini returned an incomplete set of answers.");
  }
  for (const [id, question] of Object.entries(questions)) {
    const answer = answers[id];
    const choices = Object.keys(question.criteria ?? {});
    const valid = question.type === "noul"
      ? probability(answer?.noul)
      : choices.includes(answer?.choice)
        && Object.keys(answer.probabilities ?? {}).length === choices.length
        && choices.every(choice => probability(answer.probabilities?.[choice]))
        && Math.abs(choices.reduce((sum, choice) => sum + answer.probabilities[choice], 0) - 1) <= .03
        && choices.every(choice => answer.probabilities[choice] <= answer.probabilities[answer.choice] + .000001);
    if (answer?.type !== question.type || !valid) {
      throw new Error("Gemini returned an invalid answer for " + id + ".");
    }
  }

  const candidates = [];
  function collect(window, prefix, category, confirmation) {
    if (confirmation < MIN_EVIDENCE) return;
    let peak = null;
    for (let second = window.start; second <= window.end; second++) {
      const score = second < window.end ? answers[prefix + second].noul : 0;
      if (score >= MIN_EVIDENCE) {
        if (!peak || score > peak.score) peak = { timestampSeconds: second, score };
      } else if (peak) {
        candidates.push({
          timestampSeconds: peak.timestampSeconds,
          score: Math.min(peak.score, confirmation),
          category,
          start: window.start,
          end: window.end,
        });
        peak = null;
      }
    }
  }

  for (const window of windows) {
    const physical = answers[window.prefix + "_physical"];
    const category = answers[window.prefix + "_category"].choice;
    const confirmed = answers[window.prefix + "_confirm"].noul;
    const normal = answers[window.prefix + "_normal"].noul;
    if (physical.choice === "new_deposit" && category !== "unresolved" && normal < STRONG_NORMAL_ACTION) {
      const confirm = Math.min(physical.probabilities.new_deposit, confirmed);
      collect(window, "release_", category, confirm);
    }
    collect(window, "dog_", "dog_presence", answers[window.prefix + "_dog"].noul);
  }

  const retained = [];
  for (const candidate of candidates.sort((a, b) => b.score - a.score || a.timestampSeconds - b.timestampSeconds)) {
    const duplicate = retained.some(kept =>
      (kept.category === "dog_presence") === (candidate.category === "dog_presence")
      && candidate.start < kept.end && kept.start < candidate.end
      && Math.abs(kept.timestampSeconds - candidate.timestampSeconds) <= 6);
    if (!duplicate) retained.push(candidate);
  }
  return retained.sort((a, b) => a.timestampSeconds - b.timestampSeconds)
    .map(({ timestampSeconds, score, category }) => ({ timestampSeconds, score, category }));
}

function uploadEndpoint(value) {
  const url = new URL(value);
  if (url.origin !== BASE || url.username || url.password) {
    throw new Error("Google returned an unexpected upload URL.");
  }
  return url;
}

async function analyze({ video, extension, duration, key, model = MODEL, resolution = "high", onPhase, signal, fetcher = fetch }) {
  const mime = MIME[extension];
  if (typeof model !== "string" || model.length > 80 || !MODEL_ID.test(model)) {
    throw new Error("Enter a Gemini Decision model ID, for example gemini-decision-1.");
  }
  if (!RESOLUTIONS.has(resolution)) throw new Error("Choose High or Low video resolution.");
  if (!Object.hasOwn(MIME, extension)) throw new Error("Choose a supported video file.");
  if (!Number.isFinite(duration) || duration <= 0 || duration > 450) {
    throw new Error("Gemini analysis supports clips up to 7 minutes 30 seconds.");
  }
  const { questions, windows } = buildQuestions(duration);
  const stat = await fs.promises.stat(video);
  if (stat.size <= 0 || stat.size > 2_000_000_000) throw new Error("Choose a video file between 1 byte and 2 GB.");

  async function request(stage, url, options = {}, requestSignal = signal) {
    const result = await fetcher(url, {
      ...options,
      redirect: "error",
      headers: { "x-goog-api-key": key, ...options.headers },
      signal: AbortSignal.any([AbortSignal.timeout(600_000), ...(requestSignal ? [requestSignal] : [])]),
    });
    if (!result.ok) {
      const detail = {
        400: " Check the API key and video format.",
        401: " Check your Gemini API key.",
        403: " Check your API key and access to the selected Gemini Decision model.",
        404: " Check whether the selected Gemini Decision model is available to this Google project.",
        429: " Check the project’s Gemini quota and billing.",
      }[result.status] ?? "";
      throw new Error(stage + " failed (HTTP " + result.status + ")." + detail);
    }
    return result;
  }

  onPhase("Uploading video to Gemini");
  const started = await request("Gemini upload setup", BASE + "/upload/v1beta/files", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Goog-Upload-Protocol": "resumable",
      "X-Goog-Upload-Command": "start",
      "X-Goog-Upload-Header-Content-Length": String(stat.size),
      "X-Goog-Upload-Header-Content-Type": mime,
    },
    body: JSON.stringify({ file: { display_name: "littering-review-video" } }),
  });
  const endpoint = uploadEndpoint(started.headers.get("x-goog-upload-url"));
  let remote;
  try {
    const input = await open(video, "r");
    const body = input.createReadStream();
    try {
      const uploaded = await request("Gemini upload", endpoint, {
        method: "POST",
        duplex: "half",
        headers: {
          "Content-Type": mime,
          "Content-Length": String(stat.size),
          "X-Goog-Upload-Offset": "0",
          "X-Goog-Upload-Command": "upload, finalize",
        },
        body,
      });
      remote = (await uploaded.json()).file;
    } finally {
      body.destroy();
      await input.close();
    }
    if (!/^files\/[a-zA-Z0-9_-]+$/.test(remote?.name ?? "")) {
      throw new Error("Gemini returned an invalid file reference.");
    }
    const fileUrl = BASE + "/v1beta/" + remote.name;
    const deadline = Date.now() + 600_000;
    onPhase("Preparing the video in Gemini");
    while (remote.state === "PROCESSING") {
      if (Date.now() >= deadline) throw new Error("Gemini video preparation timed out.");
      await pause(3000, undefined, { signal });
      remote = await (await request("Gemini video preparation", fileUrl)).json();
    }
    if (remote.state !== "ACTIVE" || remote.uri !== fileUrl) {
      throw new Error("Gemini could not prepare this video.");
    }
    onPhase("Checking events with " + model);
    const result = await request("Gemini analysis", BASE + "/v1beta/decisions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        input: [{ type: "text", text: rubric }, { type: "video", uri: remote.uri, mime_type: mime, resolution }],
        questions,
      }),
    });
    const response = await result.json();
    return { model, events: decode(response, questions, windows) };
  } finally {
    if (/^files\/[a-zA-Z0-9_-]+$/.test(remote?.name ?? "")) {
      // Clean up the uploaded file even when the analysis is cancelled.
      try {
        await fetcher(BASE + "/v1beta/" + remote.name, {
          method: "DELETE",
          headers: { "x-goog-api-key": key },
          redirect: "error",
          signal: AbortSignal.timeout(5000),
        });
      } catch { /* Google also expires Files API uploads. */ }
    }
  }
}

module.exports = { analyze, buildQuestions, decode };
