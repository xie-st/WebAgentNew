// The landing site's code blocks are highlighted by the module the whole site
// shares. This file is the address they have always imported from, kept so the
// call sites do not have to move; the tokenizer itself lives one directory up,
// beside the documentation site that now uses the same one.
//
// `shared/syntaxHighlight.ts` also explains why a token can carry more than one
// class, which is what lets the two sites keep their own palettes.
export { highlightCode, tokenize, isHighlighted } from "@/shared/syntaxHighlight";
export type { SynToken } from "@/shared/syntaxHighlight";
