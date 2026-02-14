/**
 * Prompt builder — constructs system and user prompts for the RAG pipeline.
 *
 * Formats retrieved chunks as numbered sources and instructs the model
 * to cite them using [N] notation.
 */

export interface SourceChunk {
  content: string;
  metadata: Record<string, string>;
  score: number;
}

export interface BuiltPrompt {
  system: string;
  user: string;
}

/**
 * Build a system + user prompt pair for the LLM.
 *
 * The system prompt includes:
 * - Role description (legal research assistant)
 * - Numbered source excerpts with metadata
 * - Instructions to cite sources using [N] notation
 * - Instructions to refuse when evidence is insufficient
 */
export function buildPrompt(
  question: string,
  chunks: SourceChunk[],
): BuiltPrompt {
  const sourcesSection =
    chunks.length > 0
      ? chunks
          .map((chunk, i) => {
            const metaEntries = Object.entries(chunk.metadata)
              .map(([key, value]) => `${key}: ${value}`)
              .join(", ");
            const metaLine = metaEntries ? ` (${metaEntries})` : "";
            return `[${i + 1}]${metaLine}\n${chunk.content}`;
          })
          .join("\n\n")
      : "No sources available.";

  const system = [
    "You are a legal research assistant. Your role is to answer questions about legal cases and legal concepts based on the provided source materials.",
    "",
    "## Sources",
    "",
    sourcesSection,
    "",
    "## Instructions",
    "",
    "- Cite your sources using [N] notation (e.g. [1], [2]) when referencing information from the provided sources.",
    "- If the provided sources contain insufficient evidence to answer the question, clearly state that you don't have enough information and cannot provide a reliable answer.",
    "- Base your answers strictly on the provided sources. Do not fabricate information.",
  ].join("\n");

  return {
    system,
    user: question,
  };
}
