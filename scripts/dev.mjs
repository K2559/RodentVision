import { spawn } from "node:child_process";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const viteEntry = path.join(root, "node_modules", "vite", "bin", "vite.js");
const electronEntry = path.join(root, "node_modules", "electron", "cli.js");
const port = Number(process.env.RODENTVISION_PORT || 5178);
const url = `http://127.0.0.1:${port}`;

const vite = spawn(process.execPath, [viteEntry, "--host", "127.0.0.1", "--port", String(port)], {
  cwd: root,
  stdio: "inherit",
  windowsHide: true,
});

async function waitForServer() {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // Vite is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error(`Vite did not become ready at ${url}`);
}

let electron;
try {
  await waitForServer();
  electron = spawn(process.execPath, [electronEntry, "."], {
    cwd: root,
    stdio: "inherit",
    windowsHide: false,
    env: { ...process.env, VITE_DEV_SERVER_URL: url },
  });
  electron.on("exit", (code) => {
    vite.kill();
    process.exit(code ?? 0);
  });
} catch (error) {
  vite.kill();
  console.error(error);
  process.exit(1);
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    electron?.kill();
    vite.kill();
  });
}
