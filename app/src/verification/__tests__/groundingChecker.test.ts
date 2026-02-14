import { describe, it, expect, vi } from "vitest";
import { checkGrounding } from "../groundingChecker.js";

describe("checkGrounding", () => {
  it("marks citations found in corpus as grounded", async () => {
    const mockStore = {
      add: vi.fn(),
      query: vi.fn().mockResolvedValue([
        {
          content: "Smith v. Jones, 550 U.S. 544 established that...",
          metadata: {},
          score: 0.95,
        },
      ]),
    };
    const result = await checkGrounding(
      "As held in Smith v. Jones, 550 U.S. 544 (2007), the standard...",
      mockStore as any,
    );
    expect(result.citations).toHaveLength(1);
    expect(result.citations[0].grounded).toBe(true);
  });

  it("flags citations not found in corpus as unverified", async () => {
    const mockStore = {
      add: vi.fn(),
      query: vi.fn().mockResolvedValue([]),
    };
    const result = await checkGrounding(
      "As held in Fake v. Case, 999 U.S. 1 (2099)...",
      mockStore as any,
    );
    expect(result.citations[0].grounded).toBe(false);
    expect(result.hasUnverifiedCitations).toBe(true);
  });

  it("handles text with no citations", async () => {
    const mockStore = {
      add: vi.fn(),
      query: vi.fn().mockResolvedValue([]),
    };
    const result = await checkGrounding("No citations here.", mockStore as any);
    expect(result.citations).toHaveLength(0);
    expect(result.hasUnverifiedCitations).toBe(false);
  });

  it("uses score threshold to determine grounding", async () => {
    const mockStore = {
      add: vi.fn(),
      query: vi
        .fn()
        .mockResolvedValue([
          { content: "Some vaguely related text", metadata: {}, score: 0.3 },
        ]),
    };
    const result = await checkGrounding(
      "In Smith v. Jones, 550 U.S. 544 (2007)...",
      mockStore as any,
      { scoreThreshold: 0.5 },
    );
    expect(result.citations[0].grounded).toBe(false);
  });
});
