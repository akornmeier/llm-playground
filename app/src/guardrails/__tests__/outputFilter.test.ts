import { describe, it, expect } from "vitest";
import { filterOutput } from "../outputFilter.js";

describe("filterOutput", () => {
  it("flags responses with no citations on factual claims", () => {
    const result = filterOutput(
      "The Supreme Court ruled that this is unconstitutional.",
      { requireCitations: true },
    );
    expect(result.warnings).toContain("factual_claim_without_citation");
  });

  it("flags responses that give direct legal advice", () => {
    const result = filterOutput(
      "You should file a motion to dismiss immediately.",
    );
    expect(result.warnings).toContain("direct_legal_advice");
  });

  it("passes well-cited informational responses", () => {
    const result = filterOutput(
      "In Smith v. Jones, 550 U.S. 544 (2007), the Court held that the standard requires a showing of actual malice.",
    );
    expect(result.warnings).toHaveLength(0);
  });

  it("does not flag citations check when requireCitations is false", () => {
    const result = filterOutput("The court ruled this is unconstitutional.", {
      requireCitations: false,
    });
    expect(result.warnings).not.toContain("factual_claim_without_citation");
  });
});
