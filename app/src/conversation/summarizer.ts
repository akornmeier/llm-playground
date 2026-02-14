import type { Message } from "./history.js";

/**
 * Simple summarization by extracting the first sentence from each message.
 * This is a placeholder for LLM-based summarization.
 */
export function summarizeMessages(messages: Message[]): string {
  return messages
    .map((msg) => {
      const firstSentence = msg.content.split(/[.!?]/)[0]?.trim();
      return firstSentence || msg.content.trim();
    })
    .filter((s) => s.length > 0)
    .join(". ");
}
