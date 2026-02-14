/**
 * Input filter — detects PII and out-of-scope requests before they reach
 * the LLM pipeline.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface InputFilterResult {
  containsPII: boolean;
  piiTypes: string[];
  inScope: boolean;
}

// ---------------------------------------------------------------------------
// PII patterns
// ---------------------------------------------------------------------------

const PII_PATTERNS: Array<{ type: string; pattern: RegExp }> = [
  { type: "ssn", pattern: /\d{3}-\d{2}-\d{4}/ },
  { type: "email", pattern: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/ },
  { type: "phone", pattern: /\d{3}[-.)]\d{3}[-.)]\d{4}/ },
];

// ---------------------------------------------------------------------------
// Scope keywords — if none are present the query is considered out-of-scope
// ---------------------------------------------------------------------------

const LEGAL_KEYWORDS = [
  "law",
  "court",
  "legal",
  "case",
  "statute",
  "regulation",
  "contract",
  "liability",
  "plaintiff",
  "defendant",
  "judge",
  "attorney",
  "motion",
  "ruling",
  "verdict",
  "trial",
  "appeal",
  "jurisdiction",
  "precedent",
  "constitution",
  "amendment",
  "tort",
  "negligence",
  "damages",
  "injunction",
  "habeas",
  "prosecution",
  "counsel",
  "litigation",
  "summary judgment",
  "holding",
  " v. ",
  " v ",
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Analyse user input for PII and scope relevance.
 */
export function filterInput(text: string): InputFilterResult {
  const piiTypes: string[] = [];

  for (const { type, pattern } of PII_PATTERNS) {
    if (pattern.test(text)) {
      piiTypes.push(type);
    }
  }

  const lower = text.toLowerCase();
  const inScope = LEGAL_KEYWORDS.some((kw) => lower.includes(kw));

  return {
    containsPII: piiTypes.length > 0,
    piiTypes,
    inScope,
  };
}
