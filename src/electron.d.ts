import type { RodentVisionBridge } from "./types";

declare global {
  interface Window {
    rodentVision: RodentVisionBridge;
  }
}

export {};
