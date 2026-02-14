import { describe, it, expect } from "vitest";
import { chunkDocument } from "../chunker.js";

describe("chunkDocument", () => {
  it("splits by section headers in legal text", () => {
    const text =
      "I. BACKGROUND\nSome facts here.\nII. ANALYSIS\nLegal analysis.";
    const chunks = chunkDocument(text, { strategy: "section-aware" });
    expect(chunks).toHaveLength(2);
    expect(chunks[0].content).toContain("BACKGROUND");
    expect(chunks[1].content).toContain("ANALYSIS");
  });

  it("falls back to naive splitting for unstructured text", () => {
    const text = "A ".repeat(500);
    const chunks = chunkDocument(text, { strategy: "naive", maxChars: 200 });
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((c) => expect(c.content.length).toBeLessThanOrEqual(200));
  });

  it("preserves metadata through chunking", () => {
    const text = "I. BACKGROUND\nFacts.";
    const chunks = chunkDocument(text, {
      strategy: "section-aware",
      metadata: { caseId: "123", court: "SCOTUS" },
    });
    expect(chunks[0].metadata).toEqual(
      expect.objectContaining({ caseId: "123" }),
    );
  });

  it("handles text with no section headers using section-aware strategy", () => {
    const text = "Just a plain paragraph of legal text.";
    const chunks = chunkDocument(text, { strategy: "section-aware" });
    expect(chunks).toHaveLength(1);
    expect(chunks[0].content).toContain("plain paragraph");
  });

  it("naive splitting accumulates sentences then splits at boundary", () => {
    const text =
      "First sentence here. Second sentence here. Third sentence is also here. Fourth one too.";
    const chunks = chunkDocument(text, { strategy: "naive", maxChars: 50 });
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((c) => expect(c.content.length).toBeLessThanOrEqual(50));
  });

  it("naive splitting uses default maxChars when not specified", () => {
    const text = "Short text.";
    const chunks = chunkDocument(text, { strategy: "naive" });
    expect(chunks).toHaveLength(1);
    expect(chunks[0].content).toBe("Short text.");
  });

  it("assigns sequential index to each chunk", () => {
    const text = "I. FIRST\nContent.\nII. SECOND\nMore.\nIII. THIRD\nEnd.";
    const chunks = chunkDocument(text, { strategy: "section-aware" });
    expect(chunks).toHaveLength(3);
    expect(chunks[0].index).toBe(0);
    expect(chunks[1].index).toBe(1);
    expect(chunks[2].index).toBe(2);
  });

  it("handles empty text", () => {
    const chunks = chunkDocument("", { strategy: "section-aware" });
    expect(chunks).toHaveLength(0);
  });
});
