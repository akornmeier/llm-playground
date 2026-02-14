import { describe, it, expect } from "vitest";
import { buildPrompt } from "../promptBuilder.js";

describe("buildPrompt", () => {
  it("includes retrieved context with source numbers", () => {
    const chunks = [
      {
        content: "The court held X",
        metadata: { case_name: "Smith v Jones", court: "SCOTUS" },
        score: 0.9,
      },
    ];
    const prompt = buildPrompt("What did the court decide?", chunks);
    expect(prompt.system).toContain("[1]");
    expect(prompt.system).toContain("Smith v Jones");
    expect(prompt.system).toContain("The court held X");
    expect(prompt.user).toContain("What did the court decide?");
  });

  it("instructs the model to cite sources", () => {
    const prompt = buildPrompt("question", []);
    expect(prompt.system.toLowerCase()).toContain("cite");
  });

  it("instructs the model to refuse when evidence is insufficient", () => {
    const prompt = buildPrompt("question", []);
    expect(prompt.system.toLowerCase()).toMatch(
      /insufficient|cannot|unable|don't have/,
    );
  });

  it("numbers multiple sources sequentially", () => {
    const chunks = [
      {
        content: "First source",
        metadata: { case_name: "Case A" },
        score: 0.9,
      },
      {
        content: "Second source",
        metadata: { case_name: "Case B" },
        score: 0.8,
      },
    ];
    const prompt = buildPrompt("question", chunks);
    expect(prompt.system).toContain("[1]");
    expect(prompt.system).toContain("[2]");
  });
});
