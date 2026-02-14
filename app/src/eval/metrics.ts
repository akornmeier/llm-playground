/**
 * Evaluation metrics — compute quality scores from grounding results.
 */

import type { GroundingResult } from "../verification/groundingChecker.js";

/**
 * Compute the ratio of grounded (verified) citations.
 *
 * Returns 1.0 when there are no citations (nothing to verify).
 */
export function citationAccuracy(result: GroundingResult): number {
  const { citations } = result;
  if (citations.length === 0) return 1.0;

  const grounded = citations.filter((c) => c.grounded).length;
  return grounded / citations.length;
}

/**
 * Compute the ratio of ungrounded (hallucinated) citations.
 *
 * Returns 0 when there are no citations.
 */
export function hallucinationRate(result: GroundingResult): number {
  const { citations } = result;
  if (citations.length === 0) return 0;

  const ungrounded = citations.filter((c) => !c.grounded).length;
  return ungrounded / citations.length;
}
