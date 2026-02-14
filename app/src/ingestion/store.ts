/**
 * In-memory vector store using cosine similarity on word-frequency vectors.
 *
 * No external dependencies — suitable for testing and lightweight usage.
 * The embedding strategy can be swapped out later by replacing the internal
 * `toVector` function with a real embedding model.
 */

export interface StoreDocument {
  content: string;
  metadata: Record<string, string>;
}

export interface QueryResult {
  content: string;
  metadata: Record<string, string>;
  score: number;
}

interface StoredEntry {
  content: string;
  metadata: Record<string, string>;
  vector: Map<string, number>;
}

// ---------------------------------------------------------------------------
// Tokenisation & vectorisation helpers
// ---------------------------------------------------------------------------

/**
 * Normalise and tokenise text into lowercase words, stripping punctuation.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);
}

/**
 * Build a term-frequency vector (word -> count) from text.
 */
function toVector(text: string): Map<string, number> {
  const tokens = tokenize(text);
  const freq = new Map<string, number>();
  for (const token of tokens) {
    freq.set(token, (freq.get(token) ?? 0) + 1);
  }
  return freq;
}

/**
 * Cosine similarity between two term-frequency maps.
 * Returns a value in [0, 1] (negative components are impossible with counts).
 */
function cosineSimilarity(
  a: Map<string, number>,
  b: Map<string, number>,
): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (const [word, countA] of a) {
    normA += countA * countA;
    const countB = b.get(word);
    if (countB !== undefined) {
      dot += countA * countB;
    }
  }

  for (const countB of b.values()) {
    normB += countB * countB;
  }

  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ---------------------------------------------------------------------------
// VectorStore
// ---------------------------------------------------------------------------

export class VectorStore {
  private entries: StoredEntry[] = [];

  /**
   * Add documents to the store. Each document is vectorised and stored
   * in memory for later similarity search.
   */
  async add(docs: StoreDocument[]): Promise<void> {
    for (const doc of docs) {
      this.entries.push({
        content: doc.content,
        metadata: { ...doc.metadata },
        vector: toVector(doc.content),
      });
    }
  }

  /**
   * Query the store for the `topK` most similar documents to `text`.
   */
  async query(text: string, options: { topK: number }): Promise<QueryResult[]> {
    if (this.entries.length === 0) return [];

    const queryVec = toVector(text);

    const scored = this.entries.map((entry) => ({
      content: entry.content,
      metadata: entry.metadata,
      score: cosineSimilarity(queryVec, entry.vector),
    }));

    scored.sort((a, b) => b.score - a.score);

    return scored.slice(0, options.topK);
  }
}
