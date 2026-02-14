import { describe, it, expect } from "vitest";
import { filterInput } from "../inputFilter.js";

describe("filterInput", () => {
  it("detects SSN in user input", () => {
    const result = filterInput("Find cases about John Smith, SSN 123-45-6789");
    expect(result.containsPII).toBe(true);
    expect(result.piiTypes).toContain("ssn");
  });

  it("detects email in user input", () => {
    const result = filterInput("Contact john@example.com for details");
    expect(result.containsPII).toBe(true);
    expect(result.piiTypes).toContain("email");
  });

  it("detects out-of-scope requests", () => {
    const result = filterInput("Write me a poem about flowers");
    expect(result.inScope).toBe(false);
  });

  it("passes clean legal queries", () => {
    const result = filterInput("What is the standard for summary judgment?");
    expect(result.containsPII).toBe(false);
    expect(result.inScope).toBe(true);
  });

  it("allows legal questions with case names", () => {
    const result = filterInput(
      "Explain the holding in Brown v. Board of Education",
    );
    expect(result.inScope).toBe(true);
    expect(result.containsPII).toBe(false);
  });
});
