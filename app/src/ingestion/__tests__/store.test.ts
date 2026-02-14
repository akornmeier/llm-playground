import { describe, it, expect, beforeEach } from "vitest";
import { VectorStore } from "../store.js";

describe("VectorStore", () => {
  let store: VectorStore;

  beforeEach(() => {
    store = new VectorStore();
  });

  it("stores and retrieves documents by similarity", async () => {
    await store.add([
      {
        content: "The Fourth Amendment protects against unreasonable searches",
        metadata: { id: "1" },
      },
      {
        content: "Contract law requires offer and acceptance",
        metadata: { id: "2" },
      },
    ]);
    const results = await store.query("search and seizure", { topK: 1 });
    expect(results).toHaveLength(1);
    expect(results[0].content).toBeDefined();
  });

  it("respects topK parameter", async () => {
    await store.add([
      { content: "First document about law", metadata: { id: "1" } },
      { content: "Second document about courts", metadata: { id: "2" } },
      { content: "Third document about judges", metadata: { id: "3" } },
    ]);
    const results = await store.query("legal documents", { topK: 2 });
    expect(results).toHaveLength(2);
  });

  it("returns empty array when store is empty", async () => {
    const results = await store.query("anything", { topK: 5 });
    expect(results).toEqual([]);
  });
});
