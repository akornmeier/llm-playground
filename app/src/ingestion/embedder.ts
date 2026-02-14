/**
 * Simple text-to-vector embedding using term frequency.
 *
 * This is a lightweight, dependency-free implementation suitable for testing
 * and prototyping. In production, swap this out for a real embedding model
 * (e.g. OpenAI text-embedding-3-small or a local sentence-transformers model).
 */

/**
 * Tokenise text into lowercase alphanumeric words.
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);
}

/**
 * Convert text to a sparse term-frequency vector represented as a flat
 * number array. The vector uses a simple hashing trick to map words into
 * a fixed-size bucket space, making it possible to compare vectors of
 * different documents directly.
 */
export function embed(text: string, dimensions: number = 384): number[] {
  const tokens = tokenize(text);
  const vector = new Array<number>(dimensions).fill(0);

  for (const token of tokens) {
    const hash = simpleHash(token) % dimensions;
    vector[hash] += 1;
  }

  // L2 normalise so cosine similarity reduces to dot product
  const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
  if (norm > 0) {
    for (let i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
  }

  return vector;
}

/**
 * Simple deterministic hash for a string. Not cryptographic — just needs
 * to spread words across the vector dimensions reasonably well.
 */
function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = Math.abs(hash | 0);
  }
  return hash;
}
