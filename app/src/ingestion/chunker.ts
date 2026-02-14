/**
 * Document chunking for legal text.
 *
 * Supports section-aware splitting (Roman numeral headers, "Section", "Article",
 * "PART", numbered headings) and naive character-limit splitting.
 */

export interface Chunk {
  content: string;
  metadata: Record<string, string>;
  index: number;
}

export interface ChunkOptions {
  strategy: "section-aware" | "naive";
  maxChars?: number;
  metadata?: Record<string, string>;
}

/**
 * Pattern that matches common legal section headers at the start of a line:
 *   - Roman numerals followed by a dot and space (I. II. III. IV. etc.)
 *   - "Section", "Article", or "PART" followed by a space
 *   - A digit followed by a dot and an uppercase letter (e.g. "1. DEFINITIONS")
 */
const SECTION_HEADER_RE =
  /^(?:(?:I{1,3}V?|VI{0,3}|IX|X{1,3})\.\s+|(?:Section|Article|PART)\s+|\d+\.\s+[A-Z])/m;

/**
 * Split text into chunks using the chosen strategy.
 */
export function chunkDocument(text: string, options: ChunkOptions): Chunk[] {
  const meta = options.metadata ?? {};

  if (options.strategy === "section-aware") {
    return chunkSectionAware(text, meta);
  }

  return chunkNaive(text, options.maxChars ?? 1000, meta);
}

// ---------------------------------------------------------------------------
// Section-aware chunking
// ---------------------------------------------------------------------------

function chunkSectionAware(
  text: string,
  metadata: Record<string, string>,
): Chunk[] {
  const lines = text.split("\n");
  const sections: string[] = [];
  let current = "";

  for (const line of lines) {
    if (SECTION_HEADER_RE.test(line) && current.length > 0) {
      sections.push(current.trim());
      current = line + "\n";
    } else {
      current += line + "\n";
    }
  }

  if (current.trim().length > 0) {
    sections.push(current.trim());
  }

  if (sections.length === 0) {
    return [];
  }

  return sections.map((content, index) => ({
    content,
    metadata: { ...metadata },
    index,
  }));
}

// ---------------------------------------------------------------------------
// Naive chunking (character-limit based)
// ---------------------------------------------------------------------------

function chunkNaive(
  text: string,
  maxChars: number,
  metadata: Record<string, string>,
): Chunk[] {
  // Split on sentence boundaries (period / question mark / exclamation followed
  // by whitespace) while keeping the delimiter attached to the preceding sentence.
  const sentences = text.match(/[^.!?]+[.!?]*\s*/g) ?? [text];
  const chunks: Chunk[] = [];
  let buffer = "";

  for (const sentence of sentences) {
    if (buffer.length + sentence.length > maxChars && buffer.length > 0) {
      chunks.push({
        content: buffer.trim(),
        metadata: { ...metadata },
        index: chunks.length,
      });
      buffer = "";
    }

    // If a single sentence exceeds maxChars, hard-split it.
    if (sentence.length > maxChars) {
      let remaining = sentence;
      while (remaining.length > 0) {
        const slice = remaining.slice(0, maxChars);
        chunks.push({
          content: slice.trim(),
          metadata: { ...metadata },
          index: chunks.length,
        });
        remaining = remaining.slice(maxChars);
      }
    } else {
      buffer += sentence;
    }
  }

  if (buffer.trim().length > 0) {
    chunks.push({
      content: buffer.trim(),
      metadata: { ...metadata },
      index: chunks.length,
    });
  }

  return chunks;
}
