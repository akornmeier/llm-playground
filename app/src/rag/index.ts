/**
 * RAG pipeline module — barrel export.
 */

export { retrieve, type RetrieveOptions } from "./retriever.js";
export {
  buildPrompt,
  type SourceChunk,
  type BuiltPrompt,
} from "./promptBuilder.js";
export {
  generateAnswer,
  type GenerateResult,
  type LLMProvider,
} from "./generate.js";
