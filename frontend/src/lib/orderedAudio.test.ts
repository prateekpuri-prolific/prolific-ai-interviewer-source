import { describe, expect, it } from "vitest";
import { OrderedAudio } from "./orderedAudio";

describe("spoken privacy warning barrier", () => {
  it("waits for playback completion, preserving two warnings before the question", async () => {
    const queue = new OrderedAudio();
    const events: string[] = [];
    let finish!: () => void;
    const a = queue.enqueue(async () => { events.push("warning1"); await new Promise<void>(r => { finish = r; }); });
    const b = queue.enqueue(async () => { events.push("warning2"); });
    const c = queue.enqueue(async () => { events.push("question"); }, () => events.push("microphone"));
    await Promise.resolve();
    expect(events).toEqual(["warning1"]);
    finish();
    await Promise.all([a, b, c]);
    expect(events).toEqual(["warning1", "warning2", "question", "microphone"]);
  });
  it("blocks following speech after failure until explicit reset", async () => {
    const queue = new OrderedAudio();
    const events: string[] = [];
    await expect(queue.enqueue(async () => { throw Error("speaker failed"); })).rejects.toThrow();
    await queue.enqueue(async () => { events.push("question"); });
    expect(events).toEqual([]);
    queue.reset();
    await queue.enqueue(async () => { events.push("retry"); });
    expect(events).toEqual(["retry"]);
  });
  it("cancels queued speech and prevents microphone activation after stop", async () => {
    const queue = new OrderedAudio();
    const events: string[] = [];
    const old = queue.enqueue(async () => { events.push("old"); }, () => events.push("microphone"));
    queue.reset();
    await old;
    expect(events).toEqual([]);
  });
});
