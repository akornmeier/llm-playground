import { describe, it, expect, vi } from "vitest";
import { retrieve } from "../retriever.js";

describe("retrieve", () => {
  it("returns ranked results from the store", async () => {
    const mockStore = {
      add: vi.fn(),
      query: vi.fn().mockResolvedValue([
        { content: "relevant chunk", metadata: { caseId: "1" }, score: 0.9 },
        { content: "less relevant", metadata: { caseId: "2" }, score: 0.5 },
      ]),
    };
    const results = await retrieve("search query", mockStore as any, {
      topK: 5,
    });
    expect(results).toHaveLength(2);
    expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
  });

  it("filters results below score threshold", async () => {
    const mockStore = {
      add: vi.fn(),
      query: vi.fn().mockResolvedValue([
        { content: "good", metadata: {}, score: 0.9 },
        { content: "bad", metadata: {}, score: 0.1 },
      ]),
    };
    const results = await retrieve("query", mockStore as any, {
      topK: 5,
      minScore: 0.5,
    });
    expect(results).toHaveLength(1);
    expect(results[0].content).toBe("good");
  });

  it("returns empty array when no results match", async () => {
    const mockStore = {
      add: vi.fn(),
      query: vi.fn().mockResolvedValue([]),
    };
    const results = await retrieve("query", mockStore as any, { topK: 5 });
    expect(results).toEqual([]);
  });
});
