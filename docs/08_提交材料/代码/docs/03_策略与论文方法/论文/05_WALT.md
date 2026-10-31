# WALT: Web Agents that Learn Tools

- arXiv: 2510.01524
- URL: https://arxiv.org/html/2510.01524
- 来源：arXiv HTML（LaTeXML）自动转换，公式以 $...$ 内联保留，图表编号可能与原文有出入。

---

WALT: Web Agents that Learn Tools

##### Report GitHub Issue

Title:

Content selection saved. Describe the issue below:

Description: arXiv is now an independent nonprofit! Learn more Back to arXiv License: CC BY 4.0 arXiv:2510.01524v1 [cs.CV] 01 Oct 2025

# WALT: Web Agents that Learn Tools

Viraj Prabhu Yutong Dai Matthew Fernandez Jing Gu Krithika Ramakrishnan Affiliation: Yanqi Luo Silvio Savarese Caiming Xiong Junnan Li Zeyuan Chen Ran Xu Affiliation: Salesforce AI Research

###### Abstract

Web agents promise to automate complex browser tasks, but current methods remain brittle—relying on step-by-step UI interactions and heavy LLM reasoning that break under dynamic layouts and long horizons. Humans, by contrast, exploit website-provided functionality through high-level operations like search, filter, and sort. We introduce WALT (Web Agents that Learn Tools), a framework that reverse-engineers latent website functionality into reusable invocable tools. Rather than hypothesizing ad-hoc skills, WALT exposes robust implementations of automations already designed into websites—spanning discovery (search, filter, sort), communication (post, comment, upvote), and content management (create, edit, delete). Tools abstract away low-level execution: instead of reasoning about how to click and type, agents simply call search(query) or create(listing). This shifts the computational burden from fragile step-by-step reasoning to reliable tool invocation. On VisualWebArena and WebArena, WALT achieves higher success with fewer steps and less LLM-dependent reasoning, establishing a robust and generalizable paradigm for browser automation.

## 1 Introduction

Figure 1: WALT transforms browser agent automation from brittle step-by-step reasoning to efficient tool-based abstraction. Given the task “find the cheapest blue kayak,” traditional web agents execute a lengthy sequence of primitive UI actions focusing on search boxes, hovering over dropdowns, clicking categories, and sorting and scanning results. In contrast, our method WALT (Web Agents that Learn Tools), designs a deterministic tool that exposes this website-provided functionality to the agent: search(query=‘blue kayak’, category=‘Boats’, sort_by=‘price’), reducing execution from 8+ fragile UI steps to 1 robust operation.

Consider searching for the cheapest blue kayak on a classifieds page (Fig. 1): existing web agents reason through each step—how to interact with the search box, locate filter controls, determine the correct sort option—while simultaneously handling implementation details like element selection and timing. In contrast, humans naturally think about this task in terms of website functionality: “search for kayaks, filter by price, identify the first blue one.” They abstract away the implementation details and focus on what they want to accomplish, not how the interface mechanics work.

This human capability stems from recognizing reusable patterns across websites — not only in search and filtering, but also in content creation and management (e.g., creating, editing, deleting listings) and social interactions (e.g., commenting, messaging, upvoting). Humans leverage this prior knowledge to quickly adapt their interaction strategies to new websites. In the web agent context, this intuition has inspired work on discovering “skills” (Wang et al., 2025b) for web agents—reusable action sequences that encapsulate common interaction patterns and can be applied across similar website elements or tasks.

However, existing skill discovery approaches suffer from two key limitations in which skills are discovered and how they are implemented. First, they either mine skills only from successful trajectories (Wang et al., 2025b; Sarch et al., 2024; Wang et al., 2025a)—codifying existing behaviors—or require agents to hypothesize useful automations (Zheng et al., 2025), often yielding unintuitive, overly specific, or irrelevant skills. Second, both approaches implement skills as brittle UI action sequences, highly sensitive to dynamic elements and design changes.

We propose WALT (Web Agents that Learn Tools). Unlike prior “skills” or “workflows,” which are agent-induced action sequences, our tools correspond to website-provided functionality—search bars, filters, sorting mechanisms, commenting systems, and navigation controls—that site designers have already engineered as robust automations. Each tool is exposed to the agent as a high-level deterministic call, with an underlying implementation discovered and validated through reverse-engineering. This reframing shifts the agent’s capability frontier: instead of learning brittle approximations of interaction patterns, WALT surfaces the functionality already embedded in websites as reliable, reusable tools.

On each website, WALT follows a demonstrate-generate-validate loop for each identified tool: (1) a web agent comprehensively demonstrates the functionality (e.g., all filters and sort options for search); (2) a tool generation agent maps execution traces to structured tools with validated input schemas, prioritizing deterministic actions but allowing agentic steps for dynamic elements, and attempting to replace UI sequences with more robust URL manipulation through API reverse-engineering; (3) a test agent verifies functionality against pre-vetted test inputs.

This abstraction transforms the agent’s computational burden: instead of reasoning about “how do I search for X, then filter by Y, then sort by Z” through complex UI sequences, the agent simply calls search(X), filter(Y), sort(Z) and focuses on higher-level planning. Tool discovery and optimization happen offline during website exploration, ensuring both efficiency and reliability.

We benchmark our method on VisualWebArena (Koh et al., 2024a) and WebArena (Zhou et al., 2024), discovering over 50 reusable tools spanning search and filtering, content creation and management (e.g., create, edit, delete listings), and communication or social interactions (e.g., commenting, messaging, upvoting). WALT achieves state-of-art success rates of 52.9% on VisualWebArena and 50.1% on WebArena, significantly outperforming prior work. Ablation studies further reveal that our proposed contributions — discovered tools, multimodal DOM parsing, and external verification – yield gains in both success rates (10%-30% across splits) and efficiency (1.3-1.4x fewer steps on average). Overall, WALT transforms browser agent automation from brittle step-by-step reasoning to efficient tool-based abstraction.

## 2 Related Work

Web Agents. Agents capable of directly operating a browser to perform tasks hold promise for automating online tasks. Prior work advances web agents along four axes: Perception concerns what the agent sees and how it grounds elements: some methods parse raw HTML (Gur et al., 2022; Deng et al., 2023), others process full-page screenshots with vision-language models (Furuta et al., 2023; He et al., 2024), often augmented by Set-of-Mark (SoM) visual prompts (Yang et al., 2023); recent work improves page understanding and grounding via prompting (Zheng et al., 2024; Yao et al., 2023) and task-specific training (Furuta et al., 2023; Zhang et al., 2025a; Pahuja et al., 2025; Qi et al., 2024). Planning scales test-time exploration with search (e.g., MCTS and related variants) to choose better action sequences (Koh et al., 2024b; Putta et al., 2024; Yu et al., 2024; Gu et al., 2024). Reasoning enhances step selection through chain-of-thought and ReAct-style prompting (Wei et al., 2022; Yao et al., 2023). Action execution determines how decisions touch the page: agents that use HTML or SoM predict DOM targets, whereas screenshot-only agents act via pixel-space coordinates (Xu et al., 2024), which can be more brittle to layout change. Our approach targets the action-execution and planning axes by mining reusable, efficient tools offline—encapsulating site functionality with validated schemas, URL-level operations, and targeted agentic fallbacks—so agents solve tasks faster and more reliably than step-by-step UI policies.

Benchmarks for Web Agents. Benchmarks for web agents are expanding rapidly and span simulated and real environments. Early simulated testbeds (Shi et al., 2017; Liu et al., 2018) emphasize basic navigation such as clicking and form filling, whereas Yao et al. (2022) focused on e-commerce tasks. Zhou et al. (2024) introduced WebArena, a realistic web simulation environment with replicas of various website types (e.g. shopping, public forum, maps, etc.), with rich functionality and realistic underlying databases, and a set of complex natural language tasks paired with robust rule-based evaluators. VisualWebArena (Koh et al., 2024a) further extended this environment to include visually-grounded tasks that require rich multimodal understanding, along with supporting website and evaluator additions. A complementary direction evaluates agents on real websites or production sandboxes (He et al., 2024; Zhang et al., 2025b; Drouin et al., 2024; Boisvert et al., 2024), covering e-commerce, enterprise software, and everyday workflows. We focus on WebArena and VisualWebArena, and propose a method to autonomously discover and construct reusable, website-specific tools that significantly improve agent performance and efficiency.

API-using Web Agents. While UI-level actions are the default interface to the Web, they can be inefficient and brittle. Accordingly, some works exploit API documentation to design high-level actions from APIs and thus augment or bypass UI interactions (Song et al., 2024; Ni et al., 2025). In contrast, we do not assume any API documentation – which is often undocumented or proprietary – and instead attempt to reverse-engineer website-provided functionality into callable tools with validated input schemas, URL-parameter promotion, and agentic recovery, all learned autonomously via systematic exploration.

Skill Discovery for Web Agents. Some recent works focus on discovering skills for web agents by mining successful agent trajectories: SkillWeaver (Zheng et al., 2025) produces unit-tested Python functions from successful attempts, whereas AWM Wang et al. (2025b) and ASI (Wang et al., 2025a) induce skills online (represented as text and programs, respectively) by prompting an agent to induce skills from action subsequences in successful trajectories. Both lines of work typically mine only from successful executions and implement skills by composing primitive actions, which can be brittle and effectively codify current behavior without expanding capability. By contrast, we systematically explore website-specific functionality and exploit observable regularities and site infrastructure; our learned tools are stress-tested and iteratively optimized for reliability and modularity. Unlike prior work that composes longer UI sequences, we discover and implement new, website-grounded tools with schema validation, selector stabilization, URL reverse-engineering, and targeted agentic fallbacks.

## 3 Approach

We frame browser automation as the discovery and use of tools—high-level, callable operations that abstract away fragile low-level interactions. Unlike prior work that induces ad-hoc skills or scripted action sequences, WALT treats websites as sources of structured functionality (e.g., search, filter, post). Each tool is backed by a validated action script – primarily deterministic URL/DOM operations with targeted agentic steps - Figure 2 summarizes the two-stage pipeline: strategic discovery of tool candidates followed by their construction and validation.

Figure 2: Overview of WALT. Left—Discovery: the browser agent explores key site sections to propose tool candidates and record stabilized interaction traces (robust selectors with fallbacks). Right—Construction & validation: the tool constructor turns traces into an action script (navigation, extraction, interaction, agentic steps), promotes eligible UI chains to URL operations, induces a validated input schema, then registers and tests the tool end to end. Feedback refines selectors, schema, and script until a robust single-call tool is produced.

### 3.1 Problem Formulation

Let $\mathcal{W}=\{w_{1},w_{2},\ldots,w_{n}\}$ denote a set of websites, and $\mathcal{T}=\{t_{1},t_{2},\ldots,t_{m}\}$ denote a set of tasks. A browser agent $\mathcal{B}_{\text{browser}}$ typically solves these tasks using primitive actions $\mathcal{A}_{\text{prim}}=\{a_{\texttt{click}},a_{\texttt{type}},a_{\texttt{navigate}},\ldots\}$ . Our goal is to discover and implement tools that can be registered as high-level actions $\mathcal{A}_{\text{tools}}$ , enabling $\mathcal{B}_{\text{browser}}$ to operate with an expanded action space $\mathcal{A}_{\text{prim}}\cup\mathcal{A}_{\text{tools}}$ for more efficient and reliable task execution.

We define a tool $u$ as a callable high-level action $u:\mathcal{S}\rightarrow\text{Goal}$ where $\mathcal{S}$ specifies structured input parameters and Goal is the target outcome. Once validated, tools are exposed to the policy as atomic actions that augment the agent’s vocabulary. Our approach involves two stages:

Stage 1: Tool Discovery systematically explores user-facing website sections to identify reusable functionality patterns. The process begins by prompting a web agent to navigate to key sections (content, discovery, communication areas) and discovering interactive elements through targeted interactions such as hovering over dropdowns and clicking menus. Finally, it is tasked with strategically proposing a list of reusable tool candidates with clear user intent that maximize coverage and minimize redundancy.

Stage 2: Tool Construction & Validation transforms discovered tool candidates $\tilde{u}$ into validated, executable tools through a two-agent system. The process introduces a specialized tool construction agent $\mathcal{B}_{\text{tool}}$ that collaborates with the browser agent $\mathcal{B}_{\text{browser}}$ : the browser agent executes candidates to produce traces, while the tool construction agent analyzes these traces and induces structured action scripts. Candidate tools are then validated against test inputs before being exposed as callable high-level actions.

### 3.2 Algorithm Design

#### Objective.

We formulate tool construction as a multi-objective optimization problem. First, the tool discovery phase generates a set of $K$ tool candidates $\tilde{\mathcal{U}}=\{\,(s_{i},E_{i},G_{i})\mid i=1,\ldots,K\,\}$ where $s_{i}$ is the start URL, $E_{i}$ are the relevant interactive elements, and $G_{i}$ is the specific goal to be accomplished. For each candidate, the browser agent $\mathcal{B}_{\text{browser}}$ first demonstrates the functionality to produce a grounded execution trace $\mathcal{X}$ , which a separate tool construction agent $\mathcal{B}_{\text{tool}}$ then analyzes to synthesize an executable tool and associated test inputs. Formally,

given $\displaystyle\tilde{u}=(s_{i},E_{i},G_{i})\quad\ $ (1)

execute & generate $\displaystyle\tilde{u}\xrightarrow{\mathcal{B}_{\text{browser}}}\mathcal{X}\xrightarrow{\mathcal{B}_{\text{tool}}}(u,\mathcal{I}_{\text{test}})\quad\ $ (2)

minimize $\displaystyle\text{FailRate}(u,\mathcal{I}_{\text{test}})+\text{StepCount}(u)+\text{AgenticRatio}(u).$ (3)

Here, FailRate is the fraction of failing test cases (measuring correctness), StepCount is the number of primitive operations the implementation executes (measuring efficiency), and AgenticRatio is the fraction of steps that require LLM-dependent reasoning (measuring determinism). The process iterates—updating the tool and test set with feedback—until a validated $u^{*}$ is obtained or the attempt budget is exhausted.

#### Tool Construction.

Each tool is realized as an action script: a finite sequence of navigation, extraction, UI interaction, and agentic steps that the browser agent executes automatically. We seek to expose tools as atomic actions: the agent calls them by contract and relies on their internal execution, without having to reason about intermediate steps. We deliberately bias tool execution towards deterministic operations (navigation and interaction) to improve robustness and efficiency, but permit agentic steps when interfaces are dynamic or ambiguous. We demonstrate the generality of this pipeline across constructing a diverse range of tools for discovery (e.g., search, filter, sort), content management (create, edit, delete), and communication (comment, message, upvote).

The tool construction pipeline unfolds in three tightly coupled phases:

Tool demonstration begins with $\mathcal{B}_{\text{browser}}$ executing the candidate to collect traces $\mathcal{X}$ of primitive interactions under realistic timing, focus, and layout conditions. Rather than replaying brittle action sequences as in prior workflow/skill approaches (Wang et al., 2025b; Sarch et al., 2024; Zheng et al., 2025), we explicitly prioritize discovering stable DOM selectors and store element hashes to allow for deterministic replay. This ensures subsequent execution targets elements that remain valid under minor UI changes. During exploration we also record viable alternative selectors, yielding vetted backups without overfitting to a single locator.

Tool generation maps execution traces $\mathcal{X}$ into a stepwise action script in two passes. In the first pass, four types of steps are supported: UI interaction uses cached element hashes from the execution trace for precise element targeting, with alternate selectors as backup. Navigation steps allow for changing URL routes or query parameters, which are then used to extract DOM contents for subsequent steps. Extraction steps allow for extracting DOM state for subsequent steps. Agentic steps are used for dynamic interactions that may be difficult to deterministically model, such as lazy-loaded content or file uploads. The multiple safeguards built into the action script ensure higher robustness and recoverability than prior work.

In the second pass, $\mathcal{B}_{\text{tool}}$ attempts to further optimize the existing action script by reverse-engineering parameterizable URL routes when the site exposes them (e.g. search query parameters). If successful, this replaces multi-step UI subsequences with a single URL manipulation, shortening execution and typically reducing both $\text{StepCount}(u)$ and $\text{AgenticRatio}(u)$ – an optimization missing in earlier work. Finally, $\mathcal{B}_{\text{tool}}$ induces an input schema $\mathcal{S}_{\text{inp}}$ with rich validation and support for diverse datatypes (e.g. enums for dropdown menus), optional fields, and usage examples, as well as a detailed tool description that specifies the tool’s intended usage, preconditions, and expected outcomes.

Tool registration and validation closes the loop. We register $(u,\mathcal{S}_{\text{inp}},\mathcal{I}_{\text{test}})$ as a callable action and execute it end-to-end with a new $\mathcal{B}_{\text{browser}}$ over pre-vetted $\mathcal{I}_{\text{test}}$ . Failures yield structured feedback $\mathcal{F}$ (selector drift, uncovered enum values, timing issues, or semantic mismatches after URL promotion) which $\mathcal{B}_{\text{tool}}$ uses to refine selectors (preferring the most stable hashes observed), amend $\mathcal{S}_{\text{inp}}$ (e.g., adding missing options or relaxing optionals), or edit the action script (e.g., relocating a selector or backing off an over-aggressive URL substitution). This closed validation loop systematically improves correctness and robustness, in contrast to one-shot script extraction.

Only tools that pass validation within a fixed number of attempts are exposed to the agent at runtime. As an additional failsafe for the deployed agent against unanticipated failures (e.g. significant UI changes), we equip the tool-enriched agent with agentic fallback - a last resort that spawns a fresh agent to handle a failing action script on the fly. Finally, we equip the agent with two more generic tools – a multimodal DOM parser that converts an HTML page to an interleaved input that simplifies cross-modal reasoning, and an external verification tool that we use to corroborate the agent’s self-reported task outcome, following the verifier design proposed in Xue et al. (2025).

In this manner, WALT turns complex website functionality into simple tool calls. By pairing grounded interaction ( $\mathcal{B}_{\text{browser}}$ ) with schema-checked, URL-optimized executors ( $\mathcal{B}_{\text{tool}}$ ), it delivers robust tools across discovery, content, and communication that run faster and with fewer LLM calls.

## 4 Experiments

### 4.1 Overview

We evaluate WALT on two established web agent benchmarks: VisualWebArena (Koh et al., 2024a) and WebArena (Zhou et al., 2024). Our experiments demonstrate that WALT achieves significant improvements over prior state-of-the-art methods by leveraging website-provided tools rather than brittle UI interaction sequences, improving success rates while reducing action steps. We conduct comprehensive ablation studies to validate the contribution of each component and provide detailed analysis of when and why WALT succeeds.

### 4.2 Benchmarks

VisualWebArena contains 910 visually-grounded and human-annotated web-tasks instantiated in three highly-realistic and fully-featured websites – Classifieds (234), Shopping (466), and Reddit (210). WebArena includes 812 more general tasks spanning five websites (two of which overlap with VisualWebArena) – GitLab (180), Map (109), Shopping (187), CMS (also referred to as Shopping Admin - 182), Reddit (106), and Multi-site (48). Tasks are defined by a human-annotated intent (e.g. “find the cheapest blue kayak and return its URL”) and evaluator functions (e.g. “assert URL == <XYZ>”). Besides a robust set of (exact, inclusion, and fuzzy) string and URL matching, the benchmarks also support sophisticated evaluators based on parsing page HTML and image contents. Agents are evaluated by their binary success rate – a stringent metric that only considers task completion rather than partial success, and is measured objectively by the evaluator function rather than a subjective LLM judgement.

### 4.3 Implementation Details

For our base agent, we pair a VLM planner (GPT-5 OpenAI (2025)) with a browser action executor (GPT-5-mini) using standard web actions (click, type, navigate, etc.). Observations include a page screenshot with indexed Set-of-Mark (SoM) boxes and a list of interactive elements keyed by the same indices. State is maintained via a multimodal message queue. For retrieval, we store trajectory summaries in a vector database keyed by task intent; at run time we embed the current intent and append the nearest summary in the DB (with similarity threshold $0.3$ ) as additional context. Agents authenticate to each site before execution, run for at most 30 steps, and replan every 15. We use GPT-5-mini as the verification LLM following the design of WebJudge Xue et al. (2025). The multimodal DOM parser converts a markdown dump of the page into an interleaved representation. Implementations build on browser-use (Browser-Use Team, 2024a) and workflow-use libraries (Browser-Use Team, 2024b).

### 4.4 Baselines

We compare against a representative set of state-of-the-art methods :

- Skill-based web agents: Specifically, on WebArena we benchmark against SkillWeaver (Zheng et al., 2025), AWM (Wang et al., 2025b), and ASI (Wang et al., 2025a). On VisualWebArena, we benchmark against concurrent world in tool-oriented web agents Yu et al. (2025).

- Web agents with test-time scaling: We benchmark against methods that use MCTS (Koh et al., 2024b) and reflective-MCTS (Yu et al., 2024), as well as one that uses model-based planning (Gu et al., 2024).

- API-using web agents: We benchmark against Hybrid Agent (Song et al., 2024), which generates actions from API documentations curated for WebArena.

- Computer-Use Agents: Specifically, we benchmark the Claude Computer-Use Agent (Anthropic, 2024), implementation details in Appendix A.2.

Additionally, we benchmark against SGV (Andrade et al., 2025), which proposes using an external verification module to mitigate LLM agreement bias. Finally, we include strong baselines from the original benchmark papers as well as human performance as an upper bound.

### 4.5 Main Results

Method Classifieds Shopping Reddit Avg.

GPT-4V+SoM (Koh et al., 2024a) 9.8 17.1 19.3 16.4

TreeSearch (Koh et al., 2024b) 26.5 29.0 20.5 26.4

WebDreamer (Gu et al., 2024) 25.0 26.3 15.9 23.2

Computer-Use (Anthropic, 2024) 36.7 21.9 27.5 27.0

ExaCT (Yu et al., 2024) 41.0 32.3 28.7 33.7

AWorld Yu et al. (2025) - - - 36.5

SGV (Andrade et al., 2025) 52.0 57.0 33.0 50.2

WALT (Ours) 64.1 53.4 39.0 52.9

Human (Koh et al., 2024a) 91.7 88.4 87.1 88.7

Figure 3: Results on VisualWebArena. Left. We report success rate (%) on each split as well as a weighted average. Right. We compare WALT’s performance and efficiency with a baseline implementation as control.

Method Gitlab Map Shopping CMS Reddit Multi Avg.

GPT-4+CoT (Zhou et al., 2024) - - - - - - 14.4

SkillWeaver (Zheng et al., 2025) 22.2 33.9 27.2 25.8 50.0 - 29.8

AWM (Wang et al., 2025b) 28.9 39.4 34.8 39.0 51.9 18.8 35.5

ASI (Wang et al., 2025a) 32.2 43.1 40.1 44.0 54.7 20.8 40.4

Hybrid Agent (Song et al., 2024) 44.4 45.9 25.7 41.2 51.9 16.7 38.9

WALT (Ours) 57.0 58.7 41.2 56.2 48.5 20.8 50.1

Human (Zhou et al., 2024) - - - - - - 78.2

Table 1: Performance comparison on WebArena benchmarks showing success rates (%) across different domains. Bold values indicate best performance in each column.

Table 2: Ablations on VisualWebArena-Classifieds showing the impact of different components on success rate (SR) and average number of steps. Results shown for different LLM backbones.

browser LLM tools dom-parser verify avg #steps ( $\downarrow$ ) SR (%) $\uparrow$

gpt-4.1 none text self 7.6 34.9

gpt-4.1 discovered text self 6.6-13.1% 36.4-4.3%

gemini-2.5-flash none text self 10.5 52.6

gemini-2.5-flash discovered text self 8.3-26.5% 55.3+5.1%

gpt-5-mini none text self 8.9 57.5

gpt-5-mini discovered text self 6.5-27.0% 61.5+7.0%

gpt-5-mini human demo text self 7.4-16.9% 66.0+16.2%

gpt-5-mini none multimodal self 7.5-15.7% 59.0+2.6%

gpt-5-mini none text external 11.0+23.6% 59.4+3.3%

gpt-5-mini discovered multimodal external 7.0-21.3% 64.1+11.5%

(a) Task type and complexity breakdown

(b) Success rate analysis by task complexity

(c) Action type analysis

Figure 5: Detailed analysis of the composition, success rates, and runtime invocations of tools discovered on the VisualWebArena Classifieds split.

We report performance on both benchmarks in Figure 3 and Table 1. We find:

$\triangleright$ WALT achieves state-of-the-art success rates. WALT attains the best average score (52.9%), with large gains on Classifieds (64.1%, +12.1 absolute over SGV) and Reddit (39.0%, +6.0 absolute), while remaining competitive on Shopping (53.4% vs. 57.0% for SGV). Further, it nearly doubles the success rate of the Claude Computer Use baseline (which uses an image-based observation space), also outperforms strong baselines based test-time search and tool use by 15-20 points.

On WebArena, WALT again achieves the highest overall average success rate on 5 of 6 splits (tied on the sixth), outperforming prior work in all domains by a large margin, and outperforming the best-performing skill-induction based method (ASI) by 9 points.

$\triangleright$ Tools improve both success rates and efficiency. In Figure 3 (right), we demonstrate both the performance (measured by success rate) and efficiency (measured by average # steps) of WALT on each VisualWebArena split. As a control, we benchmark our baseline implementation which uses an identical architecture but does not use tools. As seen, tools are crucial, improving performance by as much as 30.7% (relative) and efficiency by 1.4x. The baseline agent’s significantly lower success rates also validate that gains are not due to a stronger underlying LLM (GPT-5) alone.

Performance ablations. We ablate WALT on VisualWebArena Classifieds (Table 2). We first vary the LLM execution agent, and find agents equipped with discovered tools are consistently more accurate and efficient (e.g. GPT-5-mini: 7% higher success rate, 27% fewer steps). Stronger backbones benefit more, indicating that better reasoning improves tool selection and composition rather than low-level manipulation. Finally, we also benchmark a human demo strategy as a performance upper bound, wherein the authors manually demonstrate a set of tools rather than having the agent discover them - tools generated thus yield the highest success rate (66.0%). Impressively, however, WALT is able to recover most of this performance fully autonomously (64.1%), with 5% fewer steps.

Next, we ablate the two ancillary method components: we find that both multimodal DOM parsing (+2.6%) and external verification (+3.3%) yield modest performance gains, with the latter coming at the cost of extra checks (more steps). Combining all components yields the highest success (64.1%), still with substantially fewer actions than baseline policies (21.3% fewer steps).

Analyzing WALT. In Figure 5, we perform a fine-grained analysis of our method on the Classifieds split. First, in Figure 5(a) we break down each discovered tool by the total step count of its action script and its distribution across step types and functionalities. We make the following observations: i) tools span a range of functionalities across communication, content management, and search, ii) tools with the shortest action scripts correspond to URL promotions (typically discover-oriented), whereas those with longer scripts skew heavily towards deterministic UI interactions (typically content-management e.g. form-filling). iii) Agentic steps are rare: In fact, only 3 out of the 9 tools have at least one agentic step.

In Figure 5(b), we analyze the success rates of each of these tools, measured by the ratio of successful tool invocations by the agent during the entire evaluation run. Tools are used frequently (e.g. search listings is invoked 262 times) and achieve nearly perfect success rates, attesting to high reliability. Finally, Figure 5(c) breaks down the action type distribution of each tool for successful and failed agent trajectories – as seen, the agent uses both primitive and tool actions extensively in both cases.

Figure 6 presents how WALT composes a small set of learned tools to solve heterogeneous tasks. In successful cases, discovery tools (search_listings, sort_results) jump directly to the right result set via URL parameters, and an extraction step filters by visual or textual cues (e.g., “animal-shaped” wall rack) before a content action (e.g., post_comment) finalizes the task. The traces show short programs (2–5 calls) with minimal UI clicking, explaining the step–count reductions reported in aggregate. The failure case exposes two common stressors: compound constraints that mix a global optimum (“most expensive”) with a fine-grained visual predicate (“image shows it on water”), and missing or site-gated side-effect tools (rating), which together exceed what the deterministic executor can guarantee without more agentic perception or broader tool coverage. Overall, the panel illustrates the intended division of labor—deterministic navigation and schema-checked operations for speed/robustness, with targeted agentic extraction when unavoidable.

## 5 Discussion

Figure 6: Qualitative rollouts of WALT. Each row shows a task with tiled screenshots (left to right) and the agent’s actions at each step (gray bars). Left, top: [PASS] “Recall exact item and return the most recent lister’s email.” The agent chains search_listings $\to$ sort_results $\to$ extract_content $\to$ click, then surfaces the email from the item page. Right, top: [FAIL] “Find the most expensive boat with an image showing it on water; rate 5 stars and comment.” The agent finds expensive listings and grounds the the visual predicate but is unable to execute the rating action reliably. Left, bottom: [PASS] “Cheapest wall rack between $30–$40 that matches the animal shape in the image.” URL-level search and sorting prune the space, and an extraction step picks the correct visual match before navigation. Right, bottom: [PASS] “Latest white Google Pixel; post a $10-under offer.” The agent locates the newest listing and uses post_comment to complete the interaction. Across successes, trajectories are short (2–5 calls) and dominated by URL/navigation and schema-checked operations.

In this work, we reframe browser automation around tools – callable abstractions reverse-engineered from website functionality – rather than agent-imagined skills implemented as a brittle sequence of UI actions. Our method WALT exposes existing website functionality as robust tools that accept a validated input schema and accomplish a specific goal via a sequence of UI interaction, extraction, agent, and navigation steps, each with strong failsafes built in. WALT achieves state-of-the-art performance on challenging web automation benchmarks while requiring fewer LLM interventions.

Our method has certain limitations. Offline tool discovery incurs an exploration and validation cost per-website, and the type and quality of the tools discovered is a function both of what our exploration uncovers and what the site exposes. Highly dynamic interfaces, A/B experiments, CAPTCHAs, and heavy anti-automation can reduce determinism or block URL promotion. Schemas may still miss rare parameter values; selector stabilization can drift after major redesigns; and some interactions (e.g., complex editors, file uploads) still require agentic steps. Our evaluation focuses on two research benchmarks, but broader external validity (e.g., enterprise apps) remains to be tested.

These limitations also present opportunities for future work. Online tool patching when selectors and schemas drift over time can improve robustness. Extracting canonical web patterns for common functionalities (e.g. search, filter, sort) can aid generalization. Hybrid integration with official APIs when available, external MCP servers (Luo et al., 2025), and more agent-accessible observation spaces (Lù et al., 2025) can help further expand capabilities. Overall, our tool abstraction paradigm suggests a practical path for safe, auditable automation: tools carry explicit contracts, examples, and validation traces, making web agents easier to monitor, share, and maintain as sites evolve.

Ethics Statement. All authors have read and agree to the ICLR Code of Ethics. The benchmarks used (VisualWebArena and WebArena) are publicly available testbeds that simulate interactions with websites, and no experiments were conducted with human subjects. Our method is designed for research purposes; however, as with any browser automation technique, misuse (e.g., for scraping or spam) is possible. We emphasize that WALT is intended to improve robustness and reproducibility of academic benchmarks, not to enable malicious automation. All data handling follows the licenses of the underlying benchmarks, and no private or user-sensitive data is involved.

Reproducibility Statement. We have made efforts to ensure reproducibility. The paper provides full details of the tool discovery and construction pipeline (Sec.3 and Sec. A), optimization objectives and algorithmic design (Sec.3), and benchmark setups (Sec.4). Implementation details, including model choices, observation formats, step limits, and verification procedures, are described in Sec.4. Appendix materials include pseudocode, algorithm tables, and ablation analyses. Our code will be made publicly available.

## References

- Andrade et al. (2025) Moises Andrade, Joonhyuk Cha, Brandon Ho, Vriksha Srihari, Karmesh Yadav, and Zsolt Kira. Let’s think in two steps: Mitigating agreement bias in mllms with self-grounded verification. arXiv preprint arXiv:2507.11662, 2025.

- Anthropic (2024) Anthropic. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku. https://www.anthropic.com/news/3-5-models-and-computer-use, 2024. Accessed: 2025-09-07.

- Boisvert et al. (2024) Léo Boisvert, Megh Thakkar, Maxime Gasse, Massimo Caccia, Thibault de Chezelles, Quentin Cappart, Nicolas Chapados, Alexandre Lacoste, and Alexandre Drouin. Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks. Advances in Neural Information Processing Systems, 37:5996–6051, 2024.

- Browser-Use Team (2024a) Browser-Use Team. Browser-use: Enable ai to control your browser. https://github.com/browser-use/browser-use, 2024a. Accessed: 2025-01-27.

- Browser-Use Team (2024b) Browser-Use Team. Workflow-use: Browser automation workflows for ai agents. https://github.com/browser-use/workflow-use, 2024b. Accessed: 2025-01-27.

- Deng et al. (2023) Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.

- Drouin et al. (2024) Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks? In International Conference on Machine Learning, pages 11642–11662. PMLR, 2024.

- Furuta et al. (2023) Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Matsuo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. Multimodal web navigation with instruction-finetuned foundation models. arXiv preprint arXiv:2305.11854, 2023.

- Gu et al. (2024) Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, et al. Is your llm secretly a world model of the internet? model-based planning for web agents. arXiv preprint arXiv:2411.06559, 2024.

- Gur et al. (2022) Izzeddin Gur, Ofir Nachum, Yingjie Miao, Mustafa Safdari, Austin Huang, Aakanksha Chowdhery, Sharan Narang, Noah Fiedel, and Aleksandra Faust. Understanding html with large language models. arXiv preprint arXiv:2210.03945, 2022.

- He et al. (2024) Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024.

- Koh et al. (2024a) Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Russ Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 881–905, 2024a.

- Koh et al. (2024b) Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. arXiv preprint arXiv:2407.01476, 2024b.

- Liu et al. (2018) Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. arXiv preprint arXiv:1802.08802, 2018.

- Lù et al. (2025) Xing Han Lù, Gaurav Kamath, Marius Mosbach, and Siva Reddy. Build the web for agents, not agents for the web. arXiv preprint arXiv:2506.10953, 2025.

- Luo et al. (2025) Ziyang Luo, Zhiqi Shen, Wenzhuo Yang, Zirui Zhao, Prathyusha Jwalapuram, Amrita Saha, Doyen Sahoo, Silvio Savarese, Caiming Xiong, and Junnan Li. Mcp-universe: Benchmarking large language models with real-world model context protocol servers. arXiv preprint arXiv:2508.14704, 2025.

- Ni et al. (2025) Xinyi Ni, Haonan Jian, Qiuyang Wang, Vedanshi Chetan Shah, and Pengyu Hong. Doc2agent: Scalable generation of tool-using agents from api documentation. arXiv preprint arXiv:2506.19998, 2025.

- OpenAI (2025) OpenAI. Gpt-5 system card. https://cdn.openai.com/gpt-5-system-card.pdf, 2025. Accessed: 2025-01-27.

- Pahuja et al. (2025) Vardaan Pahuja, Yadong Lu, Corby Rosset, Boyu Gou, Arindam Mitra, Spencer Whitehead, Yu Su, and Ahmed Awadallah. Explorer: Scaling exploration-driven web trajectory synthesis for multimodal web agents. arXiv preprint arXiv:2502.11357, 2025.

- Putta et al. (2024) Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. Agent q: Advanced reasoning and learning for autonomous ai agents. arXiv preprint arXiv:2408.07199, 2024.

- Qi et al. (2024) Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. arXiv preprint arXiv:2411.02337, 2024.

- Sarch et al. (2024) Gabriel Herbert Sarch, Lawrence Jang, Michael J Tarr, William W Cohen, Kenneth Marino, and Katerina Fragkiadaki. Vlm agents generate their own memories: Distilling experience into embodied programs of thought. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

- Shi et al. (2017) Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. In International Conference on Machine Learning, pages 3135–3144. PMLR, 2017.

- Song et al. (2024) Yueqi Song, Frank Xu, Shuyan Zhou, and Graham Neubig. Beyond browsing: Api-based web agents. arXiv preprint arXiv:2410.16464, 2024.

- Wang et al. (2025a) Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, and Daniel Fried. Inducing programmatic skills for agentic tasks. arXiv preprint arXiv:2504.06821, 2025a.

- Wang et al. (2025b) Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In Forty-second International Conference on Machine Learning, 2025b.

- Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

- Xie et al. (2024) Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

- Xu et al. (2024) Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. Aguvis: Unified pure vision agents for autonomous gui interaction. arXiv preprint arXiv:2412.04454, 2024.

- Xue et al. (2025) Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. An illusion of progress? assessing the current state of web agents. arXiv preprint arXiv:2504.01382, 2025.

- Yang et al. (2023) Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023.

- Yao et al. (2022) Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022.

- Yao et al. (2023) Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

- Yu et al. (2025) Chengyue Yu, Siyuan Lu, Chenyi Zhuang, Dong Wang, Qintong Wu, Zongyue Li, Runsheng Gan, Chunfeng Wang, Siqi Hou, Gaochi Huang, et al. Aworld: Orchestrating the training recipe for agentic ai. arXiv preprint arXiv:2508.20404, 2025.

- Yu et al. (2024) Xiao Yu, Baolin Peng, Vineeth Vajipey, Hao Cheng, Michel Galley, and Jianfeng Gao. Teaching ai agents to explore with reflective-mcts and exploratory learning, 2024b. arXiv preprint arXiv:2410.02052, 2024.

- Zhang et al. (2025a) Bofei Zhang, Zirui Shang, Zhi Gao, Wang Zhang, Rui Xie, Xiaojian Ma, Tao Yuan, Xinxiao Wu, Song-Chun Zhu, and Qing Li. Tongui: Building generalized gui agents by learning from multimodal web tutorials. arXiv preprint arXiv:2504.12679, 2025a.

- Zhang et al. (2025b) Xianren Zhang, Shreyas Prasad, Di Wang, Qiuhai Zeng, Suhang Wang, Wenbo Yan, and Mat Hans. A functionality-grounded benchmark for evaluating web agents in e-commerce domains. arXiv preprint arXiv:2508.15832, 2025b.

- Zheng et al. (2024) Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.

- Zheng et al. (2025) Boyuan Zheng, Michael Y Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, et al. Skillweaver: Web agents can self-improve by discovering and honing skills. arXiv preprint arXiv:2504.07079, 2025.

- Zhou et al. (2024) Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations, 2024.

## Appendix A Appendix

### A.1 Analysis

Tools. In Figure 7, we include a list of all tools discovered across the WebArena and VisualWebArena benchmarks, as well as the number of attempts required to obtain a validated implementation. As seen, most tools are discovered on the first attempt, but a few more nuanced functionalities (e.g. post a comment on a Gitlab issue, searching on OpenStreetMaps, and estimating shopping on Shopping) require as many as 4 attempts.

Performance. In Figure 9(a), we include additional fine-grained performance analysis of our method on the Classifieds benchmark. First, we analyze the frequency and average length of successful and failed trajectories, segmented by the agent’s own assessment of the task outcome – as found in concurrent work [Andrade et al., 2025], web agents suffer from an ”agreement bias” and frequently rationalize even failed trajectories as successful. Our approach mitigates this bias by using an external verifier to corroborate the agent’s assessment.

In Figure 9(b), we segment performance based on task difficulty (visual, reasoning, and overall), annotations for which are available in the benchmark. Unsurprisingly, failure rates increase with increasing difficulty of any type - impressively though, WALT’s failure rate does not cross 50% even on the most difficult tasks.

Qualitative Examples. In Figure 8, we show additional qualitative examples of trajectory rollouts via our method, and include both successful and failure cases. Notably, our method is able to perform challenging visual grounding (e.g. finding the coffee mug from a thumbnail image - top row, 2nd image) and reason across websites (top row, 3rd image). However, it still struggles with tasks requiring very complex multi-step reasoning (bottom row).

Figure 7: Number of tries until successful.

Figure 8: Qualitative examples showing WALT tool discovery and execution on representative tasks.

(a) Self-reporting analysis

(b) Difficulty analysis

Figure 9: Analysis of classifieds task performance across different dimensions.

Algorithm 1 WALT: Two-Agent Tool Construction (Appendix)

1: Candidate $\tilde{u}=(s_{i},E_{i},G_{i})$ , attempt budget $N_{\text{max}}$ 2: Validated tool $u^{*}\in\mathcal{A}_{\text{tools}}$ or FAIL 3: $\text{attempts}\leftarrow 0$ 4: while $\text{attempts}<N_{\text{max}}$ do 5: $\text{attempts}\leftarrow\text{attempts}+1$ 6: Phase I: Exploration & Stabilization (by $\mathcal{B}_{\text{browser}}$ ) 7: $\mathcal{X}\leftarrow\mathcal{B}_{\text{browser}}.\text{Execute}(s_{i},E_{i},G_{i})$ 8: if $\mathcal{X}=\text{FAIL}$ then 9: continue $\triangleright$ retry with alternate exploration strategy 10: end if 11: $\mathcal{X}\leftarrow\textsc{StabilizeSelectors}(\mathcal{X})$ $\triangleright$ resolve to stable DOM hashes/locators; drop unstable segments 12: if $\mathcal{X}=\text{UNSTABLE}$ then 13: continue 14: end if 15: Phase II: Synthesis & Optimization (by $\mathcal{B}_{\text{tool}}$ ) 16: $\text{plan}\leftarrow\emptyset$ 17: for each segment $\xi\in\mathcal{X}$ do 18: $\text{step}\leftarrow\mathcal{B}_{\text{tool}}.\text{ClassifyAndCreate}(\xi)$ $\triangleright$ navigation / interaction / agentic 19: $\text{plan}\leftarrow\text{plan}\cup\{\text{step}\}$ 20: end for 21: $\text{plan}\leftarrow\textsc{AddAgenticFallbacks}(\text{plan})$ $\triangleright$ re-query DOM, retry alt selector, etc. 22: $\text{plan}\leftarrow\textsc{ReplaceWithURLOps}(\text{plan})$ $\triangleright$ promote eligible UI subsequences to URL ops 23: $\mathcal{S}_{\text{inp}}\leftarrow\textsc{InferSchema}(\mathcal{X})$ $\triangleright$ enums, optionals, descriptions, examples 24: $\mathcal{I}_{\text{test}}\leftarrow\mathcal{B}_{\text{tool}}.\text{ExtractTestInputs}(\mathcal{X},\mathcal{S}_{\text{inp}})$ 25: $u\leftarrow(\text{plan},\mathcal{S}_{\text{inp}})$ 26: Phase III: Registration & Validation (by $\mathcal{B}_{\text{browser}}$ ) 27: $\text{RegisterTool}(u,\mathcal{S}_{\text{inp}},\mathcal{I}_{\text{test}})$ 28: $\text{result}\leftarrow\mathcal{B}_{\text{browser}}.\text{TestTool}(u,\mathcal{I}_{\text{test}})$ 29: if $\text{result}=\text{SUCCESS}$ then 30: return $u$ $\triangleright$ validated; added to $\mathcal{A}_{\text{tools}}$ as $u^{*}$ 31: else 32: $\mathcal{F}\leftarrow\text{GetValidationErrors}(\text{result})$ $\triangleright$ selector drift, missing enum, timeout, semantic mismatch 33: $(s_{i},E_{i},G_{i})\leftarrow\textsc{RefineCandidate}\big((s_{i},E_{i},G_{i}),\mathcal{F}\big)$ $\triangleright$ update selectors, schema, or plan hints 34: continue 35: end if 36: end while 37: return FAIL

### A.2 Implementation Details

Tool Creation Agent Algorithm and System Prompt. We include the system prompt for the tool discovery agent in Listing A.3 and algorithm and system prompt of the tool creation agent $\mathcal{B}_{\text{tool}}$ in Algorithm 1 and Listing A.3.

Baseline Implementation Details. We use the Claude Computer-Use Agent with a dedicated desktop environment setup similar to OS-World [Xie et al., 2024]. Each task initializes with a Chrome browser and task-specific web pages. The agent receives desktop screenshots as observations, predicts OS-level actions, and executes them via pyautogui commands. Task completion is determined by either reaching the maximum step limit or agent prediction, with evaluation based on the final active webpage and parsed response.

We use claude-4-sonnet-20250514 with thinking mode enabled (temperature=1). Due to Bedrock API limits, all screenshots and task images are resized to 1280 $\times$ 720, with a maximum of 30 steps per task. Note that active webpage detection relies on heuristic algorithms using Playwright and Chrome DevTools Protocol, which may incorrectly identify the current page in edge cases. Reported accuracies should be viewed as lower bounds rather than exact measurements.

### A.3 Use of Large Language Models

Large language models (LLMs) were used to polish (proofreading, revising, and compressing) the writing, specifically Claude-4-Sonnet and GPT-5.

Experimental support, please view the build logs for errors. Generated by L A T E xml .

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

- Click the "Report Issue" () button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

We gratefully acknowledge support from our major funders, member institutions, , and all contributors. Major funding support from
