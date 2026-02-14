/**
 * Grounding checker — verifies that legal citations in generated text
 * actually exist in the source corpus by querying the vector store.
 */

import type { VectorStore } from "../ingestion/store.js";
import { extractCitations, type Citation } from "./citationExtractor.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface GroundingResult {
  citations: Array<{
    citation: Citation;
    grounded: boolean;
  }>;
  hasUnverifiedCitations: boolean;
}

export interface GroundingOptions {
  /** Minimum similarity score to consider a citation grounded (default 0.5). */
  scoreThreshold?: number;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Check whether citations in `text` are grounded in the documents held by
 * `store`.
 *
 * For every citation found in the text the store is queried. If the top
 * result's similarity score meets or exceeds the threshold the citation is
 * considered grounded; otherwise it is flagged as unverified.
 */
export async function checkGrounding(
  text: string,
  store: VectorStore,
  options?: GroundingOptions,
): Promise<GroundingResult> {
  const threshold = options?.scoreThreshold ?? 0.5;
  const extracted = extractCitations(text);

  if (extracted.length === 0) {
    return { citations: [], hasUnverifiedCitations: false };
  }

  const citations: GroundingResult["citations"] = [];

  for (const citation of extracted) {
    const results = await store.query(citation.text, { topK: 1 });
    const topScore = results.length > 0 ? results[0].score : 0;
    citations.push({
      citation,
      grounded: topScore >= threshold,
    });
  }

  const hasUnverifiedCitations = citations.some((c) => !c.grounded);

  return { citations, hasUnverifiedCitations };
}
