/**
 * Citation grounding & hallucination detection module — barrel export.
 */

export { extractCitations, type Citation } from "./citationExtractor.js";
export {
  checkGrounding,
  type GroundingResult,
  type GroundingOptions,
} from "./groundingChecker.js";
