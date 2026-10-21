/** FIFO playback barrier. A failed warning must not be overtaken by a question. */
export class OrderedAudio {
  private tail: Promise<void> = Promise.resolve();
  private generation = 0;
  private failed = false;
  private pending = 0;

  reset() {
    this.generation++;
    this.failed = false;
    this.pending = 0;
    this.tail = Promise.resolve();
  }

  enqueue(play: () => Promise<void>, onDrained?: () => void): Promise<void> {
    const generation = this.generation;
    this.pending++;
    const result = this.tail.then(async () => {
      if (generation !== this.generation || this.failed) return;
      try {
        await play();
      } catch (error) {
        if (generation === this.generation) this.failed = true;
        throw error;
      } finally {
        if (generation === this.generation) this.pending--;
      }
      if (generation === this.generation && !this.failed && this.pending === 0) onDrained?.();
    });
    this.tail = result.catch(() => undefined);
    return result;
  }
}
