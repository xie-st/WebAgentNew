// Everything a documentation page is built out of, in one import.
//
//   import { DocPage, CodeBlock, Callout, ParamTable, SeeAlso } from '../components/docs'
//
// A page should not need anything that is not here. If it does, the thing it
// needs is probably a primitive the rest of the documentation will want too, and
// belongs in `components/` beside these rather than inside one page.
//
// What each one is for:
//
//   DocPage      the page frame — title, lede, breadcrumbs, outline, previous/next
//   Callout      an aside: note, tip, warning, danger
//   ParamTable   options and parameters, one row each, matching `--help` exactly
//   ApiTable     a free-form table, when the rows are not options
//   CodeBlock    one sample; `CodeTabs` when it is one task done several ways
//   Terminal     a captured terminal session, as text
//   Figure       a captured screenshot, with what produced it
//   MermaidDiagram  a diagram, in both colour modes, inside its own scroll box
//   Steps        something that has to happen in order
//   FeatureList  a set of related capabilities
//   QuickLinks   a grid of onward links, for an index page
//   SeeAlso      the three pages a reader most often wants next

export { default as DocPage } from './DocPage'
export {
  InfoBox,
  Callout,
  ApiTable,
  QuickLinks,
  FeatureList,
  Steps,
  Step,
  SeeAlso,
  Lede,
} from './DocPage'
export { slugify } from './slugify'
export type { CalloutType, Outline } from './DocPage'

export { default as CodeBlock, CodeTabs } from './CodeBlock'
export type { CodeBlockProps, CodeTab } from './CodeBlock'

export { default as MermaidDiagram } from './MermaidDiagram'

export { default as ParamTable } from './ui/ParamTable'
export type { Param } from './ui/ParamTable'
export { default as Terminal } from './ui/Terminal'
export { default as Figure } from './ui/Figure'
