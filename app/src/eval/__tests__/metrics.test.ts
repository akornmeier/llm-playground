import { describe, it, expect } from "vitest";
import { citationAccuracy, hallucinationRate } from "../metrics.js";

describe("citationAccuracy", () => {
  it("computes ratio of grounded citations", () => {
    const groundingResult = {
      citations: [
        {
          citation: { type: "case" as const, text: "Smith v Jones" },
          grounded: true,
        },
        {
          citation: { type: "case" as const, text: "Fake v Case" },
          grounded: false,
        },
      ],
      hasUnverifiedCitations: true,
    };
    expect(citationAccuracy(groundingResult)).toBe(0.5);
  });

  it("returns 1.0 when all citations are grounded", () => {
    const groundingResult = {
      citations: [
        {
          citation: { type: "case" as const, text: "Real v Case" },
          grounded: true,
        },
      ],
      hasUnverifiedCitations: false,
    };
    expect(citationAccuracy(groundingResult)).toBe(1.0);
  });

  it("returns 1.0 when there are no citations", () => {
    const groundingResult = {
      citations: [],
      hasUnverifiedCitations: false,
    };
    expect(citationAccuracy(groundingResult)).toBe(1.0);
  });
});

describe("hallucinationRate", () => {
  it("computes ratio of ungrounded citations", () => {
    const groundingResult = {
      citations: [
        {
          citation: { type: "case" as const, text: "A" },
          grounded: true,
        },
        {
          citation: { type: "case" as const, text: "B" },
          grounded: false,
        },
        {
          citation: { type: "case" as const, text: "C" },
          grounded: false,
        },
      ],
      hasUnverifiedCitations: true,
    };
    expect(hallucinationRate(groundingResult)).toBeCloseTo(0.667, 2);
  });

  it("returns 0 when all citations are grounded", () => {
    const groundingResult = {
      citations: [
        {
          citation: { type: "case" as const, text: "A" },
          grounded: true,
        },
      ],
      hasUnverifiedCitations: false,
    };
    expect(hallucinationRate(groundingResult)).toBe(0);
  });

  it("returns 0 when there are no citations", () => {
    const groundingResult = { citations: [], hasUnverifiedCitations: false };
    expect(hallucinationRate(groundingResult)).toBe(0);
  });
});
