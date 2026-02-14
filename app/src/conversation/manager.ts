import type { Message } from "./history.js";
import { ConversationHistory } from "./history.js";

export type ManagerOptions = {
  maxTokens: number;
  systemMessage?: string;
};

export type ConversationContext = {
  messages: Message[];
  estimatedTokens: number;
};

function estimateTokens(content: string): number {
  const words = content.split(/\s+/).filter((w) => w.length > 0);
  return Math.round(words.length * 1.3);
}

export class ConversationManager {
  private history: ConversationHistory;
  private options: ManagerOptions;

  constructor(options: ManagerOptions) {
    this.options = options;
    this.history = new ConversationHistory();
  }

  addMessage(message: Message): void {
    this.history.add(message);
  }

  getContext(): ConversationContext {
    const allMessages = this.history.messages;
    const budget = this.options.maxTokens;

    // Reserve tokens for system message if present
    let systemMessage: Message | null = null;
    let systemTokens = 0;
    if (this.options.systemMessage) {
      systemMessage = {
        role: "system",
        content: this.options.systemMessage,
      };
      systemTokens = estimateTokens(systemMessage.content);
    }

    const remainingBudget = budget - systemTokens;

    // Add messages from most recent backwards until budget exceeded
    const selected: Message[] = [];
    let usedTokens = 0;

    for (let i = allMessages.length - 1; i >= 0; i--) {
      const msgTokens = estimateTokens(allMessages[i].content);
      if (usedTokens + msgTokens > remainingBudget) {
        break;
      }
      selected.unshift(allMessages[i]);
      usedTokens += msgTokens;
    }

    // Build final message list
    const result: Message[] = [];
    if (systemMessage) {
      result.push(systemMessage);
    }
    result.push(...selected);

    return {
      messages: result,
      estimatedTokens: systemTokens + usedTokens,
    };
  }
}
