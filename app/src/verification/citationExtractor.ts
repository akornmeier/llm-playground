/**
 * Extracts legal citations from text using regex pattern matching.
 *
 * Recognises two broad categories:
 *   - **Case citations** — volume + reporter + page, e.g. "550 U.S. 544"
 *   - **Statute citations** — title + code + section, e.g. "42 U.S.C. § 1983"
 */

export type Citation = {
  type: "case" | "statute";
  text: string;
};

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

/**
 * Case citation: volume reporter page, with optional parenthetical.
 *
 * Reporters matched:
 *   U.S.  |  S. Ct.  |  F.2d / F.3d / F.4d  |  F. Supp. / F. Supp. 2d / 3d
 */
const CASE_PATTERN =
  /\d+\s+(?:U\.S\.|S\.\s*Ct\.|F\.\d+[a-z]*|F\.\s*Supp\.(?:\s*\d+[a-z]*)?)\s+\d+(?:\s*\([^)]*\))?/g;

/**
 * Statute citation: title U.S.C. section.
 */
const STATUTE_PATTERN = /\d+\s+U\.S\.C\.\s*§\s*\d+/g;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Extract all legal citations from the given text.
 *
 * Returns a deduplicated array of {@link Citation} objects in the order they
 * first appear.
 */
export function extractCitations(text: string): Citation[] {
  const seen = new Set<string>();
  const results: Citation[] = [];

  // Case citations
  for (const match of text.matchAll(CASE_PATTERN)) {
    const matched = match[0];
    if (!seen.has(matched)) {
      seen.add(matched);
      results.push({ type: "case", text: matched });
    }
  }

  // Statute citations
  for (const match of text.matchAll(STATUTE_PATTERN)) {
    const matched = match[0];
    if (!seen.has(matched)) {
      seen.add(matched);
      results.push({ type: "statute", text: matched });
    }
  }

  return results;
}
