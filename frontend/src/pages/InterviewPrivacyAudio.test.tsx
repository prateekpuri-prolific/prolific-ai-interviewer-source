import { afterEach, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import InterviewPage from "./InterviewPage";

vi.mock("../lib/api", () => ({ widget: { config: async () => ({
  widget_key: "synthetic", participant_id_mode: "random", audio_available: true,
  audio_default_on: true, voice_response_available: false, response_mode_default: "text",
}) } }));

class TestSocket {
  static OPEN = 1;
  static instance: TestSocket;
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  send = vi.fn(); close = vi.fn();
  constructor() { TestSocket.instance = this; }
}

class TestAudio {
  static instances: TestAudio[] = [];
  onended: (() => void) | null = null;
  onerror: (() => void) | null = null;
  play = vi.fn(async () => undefined); pause = vi.fn(); currentTime = 0;
  constructor() { TestAudio.instances.push(this); }
}

let root: Root | undefined;
afterEach(async () => {
  if (root) await act(async () => root?.unmount());
  document.body.innerHTML = "";
  vi.unstubAllGlobals();
});

it("the interview UI waits for a warning's ended event before requesting the next spoken question", async () => {
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("WebSocket", TestSocket);
  vi.stubGlobal("Audio", TestAudio);
  TestAudio.instances = [];
  Element.prototype.scrollIntoView = vi.fn();
  URL.createObjectURL = vi.fn(() => "blob:synthetic");
  URL.revokeObjectURL = vi.fn();
  const spoken: string[] = [];
  vi.stubGlobal("fetch", vi.fn(async (_url: string, init: RequestInit) => {
    spoken.push(JSON.parse(String(init.body)).text);
    return { ok: true, blob: async () => new Blob(["synthetic"]) };
  }));
  const host = document.createElement("div"); document.body.append(host);
  root = createRoot(host);
  await act(async () => {
    root!.render(<MemoryRouter initialEntries={["/interview/synthetic"]}><Routes><Route path="/interview/:widgetKey" element={<InterviewPage />} /></Routes></MemoryRouter>);
  });
  const start = [...host.querySelectorAll("button")].find(b => b.textContent?.includes("Start Interview"));
  expect(start).toBeTruthy();
  await act(async () => { start!.click(); });
  await act(async () => { TestSocket.instance.onopen?.(); });
  await act(async () => {
    TestSocket.instance.onmessage?.({ data: JSON.stringify({ type: "privacy_warning", text: "Synthetic privacy warning" }) });
    TestSocket.instance.onmessage?.({ data: JSON.stringify({ type: "message", text: "Synthetic next question", role: "agent" }) });
  });
  expect(spoken).toEqual(["Synthetic privacy warning"]);
  expect(host.querySelector('[role="alert"]')?.textContent).toContain("Privacy notice");
  expect(host.querySelector('[role="alert"]')?.textContent).toContain("Synthetic privacy warning");
  expect(host.querySelector('#chat-scroll')?.textContent).not.toContain("Synthetic privacy warning");
  expect(TestAudio.instances[0].pause).not.toHaveBeenCalled();
  await act(async () => { TestAudio.instances[0].onended?.(); });
  expect(spoken).toEqual(["Synthetic privacy warning", "Synthetic next question"]);
});
