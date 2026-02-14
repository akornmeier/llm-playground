export type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};

export class ConversationHistory {
  private _messages: Message[] = [];

  get messages(): Message[] {
    return [...this._messages];
  }

  get estimatedTokens(): number {
    const totalWords = this._messages.reduce((sum, msg) => {
      const words = msg.content.split(/\s+/).filter((w) => w.length > 0);
      return sum + words.length;
    }, 0);
    return Math.round(totalWords * 1.3);
  }

  add(message: Message): void {
    this._messages.push(message);
  }

  clear(): void {
    this._messages = [];
  }
}
