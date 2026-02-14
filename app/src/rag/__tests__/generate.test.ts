import { describe, it, expect, vi } from "vitest";
import { generateAnswer } from "../generate.js";

describe("generateAnswer", () => {
  it("calls the provider with the built prompt", async () => {
    const mockProvider = vi.fn().mockResolvedValue({
      text: "Based on Smith v Jones [1], the court held X.",
    });
    const chunks = [
      {
        content: "The court held X",
        metadata: { case_name: "Smith v Jones" },
        score: 0.9,
      },
    ];
    const result = await generateAnswer("What happened?", chunks, mockProvider);
    expect(result.text).toContain("Smith v Jones");
    expect(mockProvider).toHaveBeenCalledOnce();
  });

  it("returns the generated text and sources used", async () => {
    const mockProvider = vi.fn().mockResolvedValue({
      text: "The answer is X [1].",
    });
    const chunks = [
      {
        content: "source text",
        metadata: { case_name: "Case A" },
        score: 0.9,
      },
    ];
    const result = await generateAnswer("question", chunks, mockProvider);
    expect(result.text).toBeDefined();
    expect(result.sources).toHaveLength(1);
  });

  it("handles empty chunks gracefully", async () => {
    const mockProvider = vi.fn().mockResolvedValue({
      text: "I don't have enough information to answer.",
    });
    const result = await generateAnswer("question", [], mockProvider);
    expect(result.text).toBeDefined();
    expect(result.sources).toHaveLength(0);
  });
});
