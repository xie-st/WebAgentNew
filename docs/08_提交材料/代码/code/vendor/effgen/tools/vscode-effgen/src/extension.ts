/**
 * effGen VSCode Extension
 *
 * Provides:
 *  1. Prompt-template completion — triggers on `effgen.prompts.<TAB>` in Python files.
 *  2. Run code lens — inline "▶ Run with effGen" button on LibraryPrompt / effgen_chat calls.
 *  3. Hover docs — show prompt template description on hover over known template names.
 */
import * as vscode from "vscode";

// ---------------------------------------------------------------------------
// Built-in prompt registry (mirrors effGen's Python prompt library).
// In a full language-server setup this would be fetched dynamically from the
// running effGen server via GET /dashboard/data.json.  For this minimal
// extension the registry is embedded so the extension works offline.
// ---------------------------------------------------------------------------

interface PromptTemplate {
  name: string;
  description: string;
  template: string;
  category: string;
}

const BUILTIN_TEMPLATES: PromptTemplate[] = [
  {
    name: "general",
    description: "General-purpose assistant prompt with configurable persona",
    template: 'LibraryPrompt("general", topic="{topic}")',
    category: "general",
  },
  {
    name: "coding",
    description: "Code generation and review prompt with language-aware instructions",
    template: 'LibraryPrompt("coding", language="{language}", task="{task}")',
    category: "coding",
  },
  {
    name: "reasoning",
    description: "Step-by-step reasoning prompt using chain-of-thought",
    template: 'LibraryPrompt("reasoning", problem="{problem}")',
    category: "reasoning",
  },
  {
    name: "analysis",
    description: "Data/text analysis prompt with structured output",
    template: 'LibraryPrompt("analysis", data="{data}", goal="{goal}")',
    category: "analysis",
  },
  {
    name: "summarize",
    description: "Concise summarisation prompt with configurable length",
    template: 'LibraryPrompt("summarize", text="{text}", max_words={max_words})',
    category: "general",
  },
  {
    name: "translate",
    description: "Translation prompt with source/target language specification",
    template: 'LibraryPrompt("translate", text="{text}", target_language="{lang}")',
    category: "general",
  },
  {
    name: "eval_rubric",
    description: "LLM-as-judge evaluation rubric prompt",
    template: 'LibraryPrompt("eval_rubric", response="{response}", rubric="{rubric}")',
    category: "eval",
  },
  {
    name: "agent_system",
    description: "System prompt for tool-using agents",
    template: 'LibraryPrompt("agent_system", tools="{tools}", persona="{persona}")',
    category: "agent",
  },
];

// ---------------------------------------------------------------------------
// Helper: fetch templates from the running server (best-effort)
// ---------------------------------------------------------------------------

async function fetchServerTemplates(serverUrl: string): Promise<PromptTemplate[]> {
  try {
    const url = `${serverUrl.replace(/\/$/, "")}/dashboard/data.json`;
    const response = await fetch(url, { signal: AbortSignal.timeout(2000) });
    if (!response.ok) {
      return [];
    }
    const data = (await response.json()) as {
      prompt_templates?: PromptTemplate[];
    };
    return data.prompt_templates ?? [];
  } catch {
    return [];
  }
}

// ---------------------------------------------------------------------------
// Completion provider
// ---------------------------------------------------------------------------

class EffgenCompletionProvider implements vscode.CompletionItemProvider {
  private templates: PromptTemplate[] = [...BUILTIN_TEMPLATES];

  constructor(private serverUrl: string) {}

  async refresh(): Promise<void> {
    const serverTemplates = await fetchServerTemplates(this.serverUrl);
    if (serverTemplates.length > 0) {
      this.templates = serverTemplates;
    }
  }

  getTemplates(): PromptTemplate[] {
    return [...this.templates];
  }

  provideCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position
  ): vscode.CompletionItem[] {
    const linePrefix = document.lineAt(position).text.substring(0, position.character);

    // Trigger on LibraryPrompt( or effgen.prompts. or %effgen_
    const triggered =
      linePrefix.includes("LibraryPrompt(") ||
      linePrefix.includes("effgen.prompts.") ||
      linePrefix.includes("%effgen_");

    if (!triggered) {
      return [];
    }

    return this.templates.map((t) => {
      const item = new vscode.CompletionItem(
        t.name,
        vscode.CompletionItemKind.Snippet
      );
      item.detail = `effGen / ${t.category}`;
      item.documentation = new vscode.MarkdownString(
        `**${t.name}** (${t.category})\n\n${t.description}\n\n\`\`\`python\n${t.template}\n\`\`\``
      );
      item.insertText = new vscode.SnippetString(
        t.template.replace(/{(\w+)}/g, "${1:$1}")
      );
      item.sortText = `0_${t.name}`;
      return item;
    });
  }
}

// ---------------------------------------------------------------------------
// Hover provider
// ---------------------------------------------------------------------------

class EffgenHoverProvider implements vscode.HoverProvider {
  private templates: Map<string, PromptTemplate> = new Map(
    BUILTIN_TEMPLATES.map((t) => [t.name, t])
  );

  updateTemplates(templates: PromptTemplate[]): void {
    this.templates = new Map(templates.map((t) => [t.name, t]));
  }

  provideHover(
    document: vscode.TextDocument,
    position: vscode.Position
  ): vscode.Hover | undefined {
    const range = document.getWordRangeAtPosition(position, /["'][\w_-]+["']/);
    if (!range) {
      return undefined;
    }
    const word = document.getText(range).replace(/["']/g, "");
    const template = this.templates.get(word);
    if (!template) {
      return undefined;
    }

    const md = new vscode.MarkdownString();
    md.appendMarkdown(`### effGen Prompt: \`${template.name}\`\n\n`);
    md.appendMarkdown(`**Category:** ${template.category}  \n`);
    md.appendMarkdown(`${template.description}\n\n`);
    md.appendCodeblock(template.template, "python");
    return new vscode.Hover(md, range);
  }
}

// ---------------------------------------------------------------------------
// Code lens provider
// ---------------------------------------------------------------------------

class EffgenCodeLensProvider implements vscode.CodeLensProvider {
  // Matches:  LibraryPrompt("...) or effgen_chat("... or %%effgen_agent
  private static readonly PATTERN =
    /^\s*(LibraryPrompt\s*\(|effgen_chat\s*\(|%%effgen_agent\b|%effgen_chat\b)/;

  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    const lenses: vscode.CodeLens[] = [];
    for (let i = 0; i < document.lineCount; i++) {
      const line = document.lineAt(i);
      if (EffgenCodeLensProvider.PATTERN.test(line.text)) {
        const range = new vscode.Range(i, 0, i, 0);
        lenses.push(
          new vscode.CodeLens(range, {
            title: "▶ Run with effGen",
            command: "effgen.runPrompt",
            arguments: [document.uri, i],
            tooltip: "Run this prompt via the effGen API server",
          })
        );
      }
    }
    return lenses;
  }
}

// ---------------------------------------------------------------------------
// Run command handler
// ---------------------------------------------------------------------------

async function runPromptAtLine(
  uri: vscode.Uri,
  line: number,
  serverUrl: string,
  defaultModel: string
): Promise<void> {
  const document = await vscode.workspace.openTextDocument(uri);
  // Extract up to 10 lines starting from the code lens line
  const end = Math.min(line + 10, document.lineCount);
  const snippet = document.getText(
    new vscode.Range(line, 0, end, 0)
  );

  // Simple extraction: get the first string argument as the prompt
  const promptMatch =
    snippet.match(/effgen_chat\s*\(\s*["'](.+?)["']/) ??
    snippet.match(/LibraryPrompt\s*\(\s*["'](\w+)["']/) ??
    snippet.match(/%effgen_chat\s+(.+)/);

  const promptText = promptMatch ? promptMatch[1] : snippet.split("\n")[0];

  // Show output channel
  const channel = vscode.window.createOutputChannel("effGen");
  channel.show(true);
  channel.appendLine(`[effGen] Running: ${promptText.substring(0, 80)}…`);
  channel.appendLine(`[effGen] Server: ${serverUrl}  Model: ${defaultModel}`);

  try {
    const res = await fetch(`${serverUrl.replace(/\/$/, "")}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: defaultModel,
        messages: [{ role: "user", content: promptText }],
      }),
      signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) {
      channel.appendLine(`[effGen] Error ${res.status}: ${await res.text()}`);
      return;
    }
    const data = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const content = data.choices?.[0]?.message?.content ?? "(no content)";
    channel.appendLine("\n[effGen] Response:\n");
    channel.appendLine(content);
  } catch (err) {
    channel.appendLine(`[effGen] Request failed: ${String(err)}`);
    channel.appendLine("[effGen] Tip: start the server with `effgen serve` or set effgen.serverUrl");
  }
}

// ---------------------------------------------------------------------------
// Extension activation
// ---------------------------------------------------------------------------

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const config = vscode.workspace.getConfiguration("effgen");
  const serverUrl: string = config.get("serverUrl") ?? "http://localhost:8080";
  const defaultModel: string = config.get("defaultModel") ?? "gpt-5-nano";

  const completionProvider = new EffgenCompletionProvider(serverUrl);
  const hoverProvider = new EffgenHoverProvider();
  const codeLensProvider = new EffgenCodeLensProvider();

  const refreshTemplates = async (): Promise<void> => {
    await completionProvider.refresh();
    hoverProvider.updateTemplates(completionProvider.getTemplates());
  };

  // Register providers for Python files (and Jupyter notebooks)
  const selector: vscode.DocumentSelector = [
    { language: "python" },
    { language: "jupyter" },
  ];

  context.subscriptions.push(
    vscode.languages.registerCompletionItemProvider(selector, completionProvider, '"', "'", "."),
    vscode.languages.registerHoverProvider(selector, hoverProvider),
    vscode.languages.registerCodeLensProvider(selector, codeLensProvider)
  );

  // Register run command
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "effgen.runPrompt",
      async (uri: vscode.Uri, line: number) => {
        await runPromptAtLine(uri, line, serverUrl, defaultModel);
      }
    )
  );

  // Register show registry command
  context.subscriptions.push(
    vscode.commands.registerCommand("effgen.showRegistry", async () => {
      await refreshTemplates();
      const panel = vscode.window.createWebviewPanel(
        "effgenRegistry",
        "effGen Prompt Registry",
        vscode.ViewColumn.One,
        { enableScripts: false }
      );
      panel.webview.html = buildRegistryHtml(completionProvider.getTemplates());
    })
  );

  // Refresh from server on startup (best-effort)
  refreshTemplates().catch(() => {});

  console.log("effGen extension activated");
}

export function deactivate(): void {}

// ---------------------------------------------------------------------------
// Registry HTML panel
// ---------------------------------------------------------------------------

function buildRegistryHtml(templates: PromptTemplate[]): string {
  const rows = templates
    .map(
      (t) =>
        `<tr><td><code>${escapeHtml(t.name)}</code></td><td>${escapeHtml(t.category)}</td><td>${escapeHtml(t.description)}</td></tr>`
    )
    .join("\n");
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>effGen Prompt Registry</title>
  <style>
    body { font-family: var(--vscode-font-family); padding: 1em; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid var(--vscode-panel-border); padding: 0.4em 0.8em; text-align: left; }
    th { background: var(--vscode-editor-selectionBackground); }
  </style>
</head>
<body>
  <h2>effGen Prompt Registry</h2>
  <table>
    <thead><tr><th>Name</th><th>Category</th><th>Description</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
</body>
</html>`;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
