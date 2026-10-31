// The one syntax highlighter on this site.
//
// The landing site and the documentation site both render code, and until now
// they did it two different ways: the landing pages ran this tokenizer, the
// documentation ran Prism. Two highlighters mean two vocabularies, two themes
// to keep in step and one more package in the bundle, so there is now one — and
// it is this one, because it needs no dependency, runs offline and colours the
// six languages the site actually shows.
//
// It emits *tokens*, not markup. `highlightCode` joins them into an HTML string
// for the landing components that render with `dangerouslySetInnerHTML`;
// `tokenize` hands the array to the documentation's `CodeBlock`, which splits it
// on newlines so a block can number and highlight individual lines.
//
// **Class names are additive, never replaced.** Where a token deserves a finer
// label than the landing stylesheet knows about — a shell builtin, a boolean, a
// mapping key — the finer class is *appended* to the one the landing site has
// always used. A stylesheet that only knows the coarse class keeps rendering
// exactly what it rendered before; one that defines the finer class can say
// more. That is what lets the two sites share a tokenizer and keep their own
// palettes.

/** One run of source text and the class list that colours it. */
export interface SynToken {
  /** Space-separated class list, or `null` for text that is not coloured. */
  cls: string | null;
  text: string;
}

const PY_KEYWORDS = new Set([
  "and", "as", "assert", "async", "await", "break",
  "class", "continue", "def", "del", "elif", "else", "except", "finally", "for",
  "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
  "or", "pass", "raise", "return", "try", "while", "with", "yield",
]);

// Coloured as keywords by the landing site since it shipped; the documentation
// gives them their own hue, which is why they carry the extra class.
const PY_CONSTANTS = new Set(["False", "None", "True"]);

const PY_BUILTINS = new Set([
  "print", "len", "range", "list", "dict", "set", "tuple", "str", "int",
  "float", "bool", "bytes", "open", "isinstance", "type", "super",
]);

const BASH_KEYWORDS = new Set([
  "if", "then", "else", "elif", "fi", "for", "in", "do", "done", "while",
  "case", "esac", "function", "return", "break", "continue", "export", "set",
  "unset", "alias", "local", "readonly", "echo",
]);

// A subset of the keywords above, used only to *append* the finer class.
// Deliberately short: a word belongs here when it is a shell builtin rather
// than shell syntax, and `set` stays out of it because `effgen config set`
// appears on more pages of this site than `set -e` does.
const BASH_SHELL_BUILTINS = new Set([
  "export", "unset", "alias", "local", "readonly", "echo",
]);

const TS_KEYWORDS = new Set([
  "as", "async", "await", "break", "case", "catch", "class", "const",
  "continue", "default", "delete", "do", "else", "enum", "export", "extends",
  "finally", "for", "from", "function", "if", "implements", "import", "in",
  "instanceof", "interface", "let", "new", "of", "private", "protected",
  "public", "readonly", "return", "satisfies", "static", "switch", "this",
  "throw", "try", "type", "typeof", "var", "void", "while", "yield",
]);

const TS_CONSTANTS = new Set(["true", "false", "null", "undefined"]);

const TS_BUILTINS = new Set([
  "Array", "Boolean", "JSON", "Math", "Number", "Object", "Promise", "String",
  "boolean", "console", "number", "string", "unknown", "any", "never",
]);

/** Grow the token list, merging adjacent uncoloured runs so it stays compact. */
function push(out: SynToken[], cls: string | null, text: string): void {
  if (text === "") return;
  const last = out[out.length - 1];
  if (last && last.cls === null && cls === null) {
    last.text += text;
    return;
  }
  out.push({ cls, text });
}

function tokenizePython(src: string): SynToken[] {
  // Order matters: comments and strings first, so a `#` inside a string and a
  // quote inside a comment are both read as part of what encloses them.
  const re =
    /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?''')|(\b[rbfu]{0,2}"(?:[^"\\\n]|\\.)*"|\b[rbfu]{0,2}'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|(\b\d+(?:\.\d+)?\b)|(@[\w.]+)|(\b[A-Z][A-Za-z0-9_]*\b)|(\b[A-Za-z_][A-Za-z0-9_]*\b)|([^\w\s])/g;

  const out: SynToken[] = [];
  let lastIndex = 0;
  // The word before this one, so `def name` and `class Name` — the two places
  // Python actually *introduces* a name — can be told from every other use of
  // one. Only the extra class carries the distinction.
  let prevWord = "";
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    if (m.index > lastIndex) push(out, null, src.slice(lastIndex, m.index));
    const [token, comment, tripleStr, str, num, decorator, capWord, word, sym] = m;
    if (comment) push(out, "syn-comment", token);
    else if (tripleStr) push(out, "syn-string", token);
    else if (str) push(out, "syn-string", token);
    else if (num) push(out, "syn-number", token);
    else if (decorator) push(out, "syn-decorator", token);
    else if (capWord) {
      // `True`, `None` and a class name are all capitalised words to the
      // regex above; only the extra class tells them apart, and only a
      // stylesheet that wants the distinction has to care.
      if (PY_CONSTANTS.has(capWord)) push(out, "syn-class syn-boolean", token);
      else if (prevWord === "class") push(out, "syn-class syn-def", token);
      else push(out, "syn-class", token);
      prevWord = capWord;
    } else if (word) {
      if (PY_KEYWORDS.has(word)) push(out, "syn-keyword", token);
      else if (PY_BUILTINS.has(word)) {
        // `print` is the one builtin Python's own highlighting has always
        // treated as a keyword; the second class lets a stylesheet agree.
        push(out, word === "print" ? "syn-builtin syn-keyword" : "syn-builtin", token);
      } else if (prevWord === "def") push(out, "syn-function syn-def", token);
      else if (src[m.index + token.length] === "(") push(out, "syn-function", token);
      else push(out, null, token);
      prevWord = word;
    } else if (sym) {
      if ("()[]{}".includes(sym)) push(out, "syn-punct", token);
      else if (",.;:".includes(sym)) push(out, "syn-punct", token);
      else if ("=+-*/%<>!&|^~".includes(sym)) push(out, "syn-operator", token);
      else push(out, null, token);
    } else {
      push(out, null, token);
    }
    lastIndex = m.index + token.length;
  }
  if (lastIndex < src.length) push(out, null, src.slice(lastIndex));
  return out;
}

function tokenizeBash(src: string): SynToken[] {
  const out: SynToken[] = [];
  const lines = src.split("\n");
  lines.forEach((line, lineIndex) => {
    if (lineIndex > 0) push(out, null, "\n");
    if (line.trimStart().startsWith("#")) {
      push(out, "syn-comment", line);
      return;
    }
    // A comment can also start part-way along a line, as long as the `#` is
    // not inside a quoted string. Everything from there to the end of the
    // line is the comment.
    let body = line;
    let trailing = "";
    for (let i = 0, quote = ""; i < line.length; i++) {
      const ch = line[i];
      if (quote) {
        if (ch === "\\") i++;
        else if (ch === quote) quote = "";
      } else if (ch === '"' || ch === "'") quote = ch;
      else if (ch === "#" && (i === 0 || /\s/.test(line[i - 1]))) {
        body = line.slice(0, i);
        trailing = line.slice(i);
        break;
      }
    }
    const re =
      /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\$\w+|\$\{[^}]+\})|(\b\d+(?:\.\d+)?\b)|(--?[A-Za-z][\w-]*)|(\b[A-Za-z_][\w-]*\b)|([|&;><=])|([^\w\s])/g;
    let last = 0;
    let firstToken = true;
    let m: RegExpExecArray | null;
    while ((m = re.exec(body)) !== null) {
      if (m.index > last) push(out, null, body.slice(last, m.index));
      const [tok, str, vari, num, flag, word, redir, sym] = m;
      if (str) push(out, "syn-string", tok);
      else if (vari) push(out, "syn-variable", tok);
      else if (num) push(out, "syn-number", tok);
      else if (flag) push(out, "syn-flag", tok);
      else if (word) {
        // `NAME=value` — the left-hand side is being assigned to, not run.
        const assigned = body[m.index + tok.length] === "=";
        if (BASH_KEYWORDS.has(word)) {
          push(out, BASH_SHELL_BUILTINS.has(word) ? "syn-keyword syn-shell-builtin" : "syn-keyword", tok);
        } else if (firstToken) {
          push(out, assigned ? "syn-function syn-command syn-assign" : "syn-function syn-command", tok);
        } else {
          push(out, assigned ? "syn-assign" : null, tok);
        }
        firstToken = false;
      } else if (redir) push(out, "syn-operator", tok);
      else if (sym) push(out, null, tok);
      else push(out, null, tok);
      if (!/^\s+$/.test(tok)) firstToken = false;
      last = m.index + tok.length;
    }
    if (last < body.length) push(out, null, body.slice(last));
    if (trailing) push(out, "syn-comment", trailing);
  });
  return out;
}

function tokenizeJson(src: string): SynToken[] {
  const re =
    /("(?:[^"\\]|\\.)*")(\s*:)?|(\b(?:true|false|null)\b)|(-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|([{}[\],:])/g;
  const out: SynToken[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    if (m.index > last) push(out, null, src.slice(last, m.index));
    const [tok, str, colon, lit, num, punct] = m;
    if (str) {
      push(out, colon ? "syn-property" : "syn-string", str);
      if (colon) {
        const ws = colon.slice(0, -1);
        push(out, null, ws);
        push(out, "syn-punct", ":");
      }
    } else if (lit) push(out, "syn-keyword syn-boolean", tok);
    else if (num) push(out, "syn-number", tok);
    else if (punct) push(out, "syn-punct", tok);
    else push(out, null, tok);
    last = m.index + tok.length;
  }
  if (last < src.length) push(out, null, src.slice(last));
  return out;
}

function tokenizeYaml(src: string): SynToken[] {
  const out: SynToken[] = [];
  const lines = src.split("\n");
  lines.forEach((line, lineIndex) => {
    if (lineIndex > 0) push(out, null, "\n");
    const commentAt = line.search(/(^|\s)#/);
    const body = commentAt >= 0 ? line.slice(0, commentAt) : line;
    const comment = commentAt >= 0 ? line.slice(commentAt) : "";

    // `  - key: value`, `key:`, `- item`, or a bare scalar continuation.
    const m = /^(\s*)(- )?([^\s:#][^:#]*?)(\s*:)(\s|$)/.exec(body);
    if (m) {
      const [, indent, dash, key, colon, tail] = m;
      push(out, null, indent);
      if (dash) push(out, "syn-punct", dash);
      push(out, "syn-property", key);
      push(out, "syn-punct", colon.trimStart() === ":" ? colon : colon);
      push(out, null, tail);
      pushYamlValue(out, body.slice(m[0].length));
    } else {
      const dashOnly = /^(\s*)(- )/.exec(body);
      if (dashOnly) {
        push(out, null, dashOnly[1]);
        push(out, "syn-punct", dashOnly[2]);
        pushYamlValue(out, body.slice(dashOnly[0].length));
      } else {
        pushYamlValue(out, body);
      }
    }
    if (comment) push(out, "syn-comment", comment);
  });
  return out;
}

function pushYamlValue(out: SynToken[], value: string): void {
  if (value === "") return;
  const re =
    /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b(?:true|false|null|yes|no|on|off)\b)|(-?\b\d+(?:\.\d+)?\b)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(value)) !== null) {
    if (m.index > last) push(out, null, value.slice(last, m.index));
    const [tok, str, lit, num] = m;
    if (str) push(out, "syn-string", tok);
    else if (lit) push(out, "syn-keyword syn-boolean", tok);
    else if (num) push(out, "syn-number", tok);
    else push(out, null, tok);
    last = m.index + tok.length;
  }
  if (last < value.length) push(out, null, value.slice(last));
}

function tokenizeToml(src: string): SynToken[] {
  const out: SynToken[] = [];
  const lines = src.split("\n");
  lines.forEach((line, lineIndex) => {
    if (lineIndex > 0) push(out, null, "\n");
    if (line.trimStart().startsWith("#")) {
      push(out, "syn-comment", line);
      return;
    }
    const table = /^(\s*)(\[+)([^\]]*)(\]+)(.*)$/.exec(line);
    if (table) {
      push(out, null, table[1]);
      push(out, "syn-punct", table[2]);
      push(out, "syn-class", table[3]);
      push(out, "syn-punct", table[4]);
      push(out, null, table[5]);
      return;
    }
    const pair = /^(\s*)([A-Za-z0-9_.-]+|"[^"]*")(\s*)(=)(.*)$/.exec(line);
    if (pair) {
      push(out, null, pair[1]);
      push(out, "syn-property", pair[2]);
      push(out, null, pair[3]);
      push(out, "syn-punct", pair[4]);
      pushYamlValue(out, pair[5]);
      return;
    }
    push(out, null, line);
  });
  return out;
}

function tokenizeTypeScript(src: string): SynToken[] {
  const re =
    /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*'|`(?:[^`\\]|\\.)*`)|(\b\d+(?:\.\d+)?\b)|(\b[A-Z][A-Za-z0-9_]*\b)|(\b[A-Za-z_$][A-Za-z0-9_$]*\b)|([^\w\s])/g;
  const out: SynToken[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src)) !== null) {
    if (m.index > last) push(out, null, src.slice(last, m.index));
    const [tok, comment, str, num, capWord, word, sym] = m;
    if (comment) push(out, "syn-comment", tok);
    else if (str) push(out, "syn-string", tok);
    else if (num) push(out, "syn-number", tok);
    else if (capWord) {
      push(out, TS_BUILTINS.has(capWord) ? "syn-builtin" : "syn-class", tok);
    } else if (word) {
      if (TS_CONSTANTS.has(word)) push(out, "syn-keyword syn-boolean", tok);
      else if (TS_KEYWORDS.has(word)) push(out, "syn-keyword", tok);
      else if (TS_BUILTINS.has(word)) push(out, "syn-builtin", tok);
      else if (src[m.index + tok.length] === "(") push(out, "syn-function", tok);
      else push(out, null, tok);
    } else if (sym) {
      if ("()[]{}".includes(sym) || ",.;:".includes(sym)) push(out, "syn-punct", tok);
      else if ("=+-*/%<>!&|^~?".includes(sym)) push(out, "syn-operator", tok);
      else push(out, null, tok);
    } else push(out, null, tok);
    last = m.index + tok.length;
  }
  if (last < src.length) push(out, null, src.slice(last));
  return out;
}

/** What each accepted `language` string is tokenized as. */
const LANGUAGES: Record<string, (src: string) => SynToken[]> = {
  python: tokenizePython,
  py: tokenizePython,
  bash: tokenizeBash,
  sh: tokenizeBash,
  shell: tokenizeBash,
  console: tokenizeBash,
  json: tokenizeJson,
  yaml: tokenizeYaml,
  yml: tokenizeYaml,
  toml: tokenizeToml,
  typescript: tokenizeTypeScript,
  ts: tokenizeTypeScript,
  tsx: tokenizeTypeScript,
  javascript: tokenizeTypeScript,
  js: tokenizeTypeScript,
  jsx: tokenizeTypeScript,
};

/** True when this site can colour the language, so a caller can say so. */
export function isHighlighted(language: string): boolean {
  return language.toLowerCase() in LANGUAGES;
}

/**
 * Split source into coloured runs.
 *
 * The concatenation of every `text` is the input, character for character —
 * nothing is dropped and nothing is added, which is what lets a caller split
 * the result on newlines and still render the source exactly as it was written.
 * An unknown language comes back as a single uncoloured token.
 */
export function tokenize(code: string, language: string): SynToken[] {
  const fn = LANGUAGES[language.toLowerCase()];
  return fn ? fn(code) : [{ cls: null, text: code }];
}

const escape = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/** The same tokens, as an HTML string, for a caller that renders markup. */
export function highlightCode(code: string, language: string): string {
  return tokenize(code, language)
    .map((t) => (t.cls ? `<span class="${t.cls}">${escape(t.text)}</span>` : escape(t.text)))
    .join("");
}
