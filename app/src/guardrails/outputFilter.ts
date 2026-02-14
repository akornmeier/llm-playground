/**
 * Output filter — checks generated responses for direct legal advice and
 * unsupported factual claims before returning them to the user.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface OutputFilterOptions {
  requireCitations?: boolean;
}

export interface OutputFilterResult {
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Detection patterns
// ---------------------------------------------------------------------------

/** Phrases that indicate direct legal advice. */
const LEGAL_ADVICE_PATTERNS = [
  /\byou should\b/i,
  /\byou must\b/i,
  /\bi recommend you\b/i,
  /\bfile a motion\b/i,
  /\byou need to file\b/i,
];

/** Phrases that indicate a factual legal claim. */
const FACTUAL_CLAIM_PATTERNS = [
  /\bcourt ruled\b/i,
  /\bcourt held\b/i,
  /\bis unconstitutional\b/i,
  /\bestablished that\b/i,
];

/**
 * Citation pattern — matches common legal citations such as
 * "550 U.S. 544" or "42 U.S.C. S 1983".
 */
const CITATION_PATTERN =
  /\d+\s+(?:U\.S\.|S\.\s*Ct\.|F\.\d+[a-z]*|F\.\s*Supp\.(?:\s*\d+[a-z]*)?)\s+\d+|U\.S\.C\.\s*§\s*\d+/;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Analyse generated text for potential issues.
 */
export function filterOutput(
  text: string,
  options?: OutputFilterOptions,
): OutputFilterResult {
  const warnings: string[] = [];

  // Check for direct legal advice
  const givesAdvice = LEGAL_ADVICE_PATTERNS.some((p) => p.test(text));
  if (givesAdvice) {
    warnings.push("direct_legal_advice");
  }

  // Check for factual claims without citations (when enabled)
  const requireCitations = options?.requireCitations ?? false;
  if (requireCitations) {
    const hasFactualClaim = FACTUAL_CLAIM_PATTERNS.some((p) => p.test(text));
    const hasCitation = CITATION_PATTERN.test(text);

    if (hasFactualClaim && !hasCitation) {
      warnings.push("factual_claim_without_citation");
    }
  }

  return { warnings };
}
