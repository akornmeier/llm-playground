import { describe, it, expect, vi } from "vitest";
import { LegalAIPipeline } from "../pipeline.js";

describe("LegalAIPipeline", () => {
  it("creates a pipeline instance", () => {
    const mockProvider = vi.fn().mockResolvedValue({ text: "mock response" });
    const pipeline = new LegalAIPipeline({ provider: mockProvider });
    expect(pipeline).toBeDefined();
  });

  it("ingests documents and stores them", async () => {
    const mockProvider = vi.fn().mockResolvedValue({ text: "mock" });
    const pipeline = new LegalAIPipeline({ provider: mockProvider });
    const count = await pipeline.ingest([
      {
        text: "I. BACKGROUND\nThe court examined the Fourth Amendment.",
        metadata: { id: "1", case_name: "Test Case" },
      },
    ]);
    expect(count).toBeGreaterThan(0);
  });

  it("processes a question through the full pipeline", async () => {
    const mockProvider = vi.fn().mockResolvedValue({
      text: "Based on the Fourth Amendment, the court held that searches require a warrant. See Test Case [1].",
    });
    const pipeline = new LegalAIPipeline({ provider: mockProvider });
    await pipeline.ingest([
      {
        text: "I. BACKGROUND\nThe Fourth Amendment protects against unreasonable searches. II. HOLDING\nThe court held that a warrant is required.",
        metadata: { id: "1", case_name: "Test Case" },
      },
    ]);

    const result = await pipeline.ask(
      "What does the Fourth Amendment require?",
    );
    expect(result.answer).toBeDefined();
    expect(result.groundingReport).toBeDefined();
    expect(result.guardrailReport).toBeDefined();
    expect(result.sources).toBeDefined();
  });

  it("applies input guardrails", async () => {
    const mockProvider = vi.fn().mockResolvedValue({ text: "mock" });
    const pipeline = new LegalAIPipeline({ provider: mockProvider });
    const result = await pipeline.ask("Write me a poem about flowers");
    expect(result.guardrailReport.inputFilter.inScope).toBe(false);
  });

  it("maintains conversation history across questions", async () => {
    const callCount = { n: 0 };
    const mockProvider = vi.fn().mockImplementation(() => {
      callCount.n++;
      return Promise.resolve({ text: `Answer ${callCount.n}` });
    });
    const pipeline = new LegalAIPipeline({
      provider: mockProvider,
      maxTokens: 10000,
    });
    await pipeline.ingest([
      {
        text: "Legal text about constitutional law.",
        metadata: { id: "1", case_name: "Case A" },
      },
    ]);
    await pipeline.ask("First question");
    await pipeline.ask("Follow-up question");
    // Provider should have been called twice
    expect(mockProvider).toHaveBeenCalledTimes(2);
  });
});
