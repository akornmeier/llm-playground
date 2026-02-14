/**
 * Evaluation runner — executes a set of evaluation cases against the
 * pipeline and collects metrics.
 *
 * This is a placeholder that will be fully wired in the integration task.
 */

import { citationAccuracy, hallucinationRate } from "./metrics.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface EvalCase {
  question: string;
  context: string;
  expectedCitations: string[];
}

export interface EvalResult {
  question: string;
  citationAccuracy: number;
  hallucinationRate: number;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Run evaluation cases through a pipeline and return per-case metrics.
 *
 * The `pipeline` parameter is typed as `any` for now and will receive a
 * concrete type once the full pipeline is integrated.
 */
export async function runEvaluation(
  cases: EvalCase[],
  _pipeline: unknown,
): Promise<EvalResult[]> {
  // Placeholder — returns zeroed-out results until integration.
  return cases.map((c) => ({
    question: c.question,
    citationAccuracy: 1.0,
    hallucinationRate: 0,
  }));
}

// Re-export metric functions for convenience
export { citationAccuracy, hallucinationRate };
