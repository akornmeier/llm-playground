/**
 * Generate — orchestrates prompt building and LLM invocation.
 *
 * The LLM provider is injected as a function argument so that tests
 * can supply a mock without needing real API keys.
 */

import { buildPrompt, type SourceChunk } from "./promptBuilder.js";

export interface GenerateResult {
  text: string;
  sources: Array<{ content: string; metadata: Record<string, string> }>;
}

/**
 * An LLM provider function that accepts a prompt and returns generated text.
 */
export type LLMProvider = (prompt: {
  system: string;
  user: string;
}) => Promise<{ text: string }>;

/**
 * Generate an answer to `question` using the provided context chunks and
 * LLM provider.
 *
 * 1. Builds a citation-aware prompt from the question and chunks.
 * 2. Passes the prompt to the provider.
 * 3. Returns the generated text along with the source chunks used.
 */
export async function generateAnswer(
  question: string,
  chunks: SourceChunk[],
  provider: LLMProvider,
): Promise<GenerateResult> {
  const prompt = buildPrompt(question, chunks);
  const response = await provider(prompt);

  return {
    text: response.text,
    sources: chunks.map((chunk) => ({
      content: chunk.content,
      metadata: chunk.metadata,
    })),
  };
}
