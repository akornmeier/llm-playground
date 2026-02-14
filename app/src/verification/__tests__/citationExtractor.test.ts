import { describe, it, expect } from "vitest";
import { extractCitations } from "../citationExtractor.js";

describe("extractCitations", () => {
  it("extracts case citations with reporter format", () => {
    const text =
      "As held in Smith v. Jones, 550 U.S. 544 (2007), the standard requires...";
    const citations = extractCitations(text);
    expect(citations).toHaveLength(1);
    expect(citations[0]).toEqual(
      expect.objectContaining({
        type: "case",
        text: expect.stringContaining("550 U.S. 544"),
      }),
    );
  });

  it("extracts statute citations", () => {
    const text = "Under 42 U.S.C. § 1983, a plaintiff may...";
    const citations = extractCitations(text);
    expect(citations).toHaveLength(1);
    expect(citations[0]).toEqual(
      expect.objectContaining({
        type: "statute",
        text: expect.stringContaining("42 U.S.C."),
      }),
    );
  });

  it("extracts multiple citations from same text", () => {
    const text =
      "In Brown v. Board, 347 U.S. 483 (1954), and under 42 U.S.C. § 1983...";
    const citations = extractCitations(text);
    expect(citations).toHaveLength(2);
  });

  it("handles Federal Reporter citations", () => {
    const text = "See Johnson v. State, 234 F.3d 890 (5th Cir. 2000)";
    const citations = extractCitations(text);
    expect(citations).toHaveLength(1);
    expect(citations[0].type).toBe("case");
  });

  it("returns empty array for text without citations", () => {
    expect(extractCitations("No legal citations here.")).toEqual([]);
  });
});
