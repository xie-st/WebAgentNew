# Read More, Think More: Revisiting Observation Reduction for Web Agents

- arXiv: 2604.01535
- URL: https://arxiv.org/html/2604.01535
- 来源：arXiv HTML（LaTeXML）自动转换，公式以 $...$ 内联保留，图表编号可能与原文有出入。

---

Read More, Think More: Revisiting Observation Reduction for Web Agents

##### Report GitHub Issue

Title:

Content selection saved. Describe the issue below:

Description: arXiv is now an independent nonprofit! Learn more Back to arXiv License: arXiv.org perpetual non-exclusive license arXiv:2604.01535v1 [cs.CL] 02 Apr 2026

# Read More, Think More:

Revisiting Observation Reduction for Web Agents

Masafumi Enomoto Ryoma Obara Haochen Zhang & Masafumi Oyamada Affiliation: NEC Corporation Email: {masafumi-enomoto,ryoma-obara,haochen-zhang,oyamada}@nec.com

###### Abstract

Web agents based on large language models (LLMs) rely on observations of web pages—commonly represented as HTML—as the basis for identifying available actions and planning subsequent steps. Prior work has treated the verbosity of HTML as an obstacle to performance and adopted observation reduction as a standard practice. We revisit this trend and demonstrate that the optimal observation representation depends on model capability and thinking token budget: (1) compact observations (accessibility trees) are preferable for lower-capability models, while detailed observations (HTML) are advantageous for higher-capability models; moreover, increasing thinking tokens further amplifies the benefit of HTML. (2) Our error analysis suggests that higher-capability models exploit layout information in HTML for better action grounding, while lower-capability models suffer from increased hallucination under longer inputs. We also find that incorporating observation history improves performance across most models and settings, and a diff-based representation offers a token-efficient alternative. Based on these findings, we suggest practical guidelines: adaptively select observation representations based on model capability and thinking token budget, and incorporate observation history using diff-based representations.

## 1 Introduction

LLM-based web agents have attracted growing research interest because they can automate repetitive web tasks such as form filling and ordering items (Ning and others, 2025). Web agents are generally implemented as a loop of action execution and observation (Yao and others, 2023): at each step, the agent executes an action (e.g., a click) and receives the current web page as an observation. This loop continues until the agent judges the task complete. The information contained in the web page observation is critical for identifying available next actions and tracking task progress.

Figure 1: Relationship between observation representation and task success rate (WorkArena L1). The x-axis shows the average number of input tokens per step; the y-axis shows the task success rate. “h” denotes HTML and “a” denotes the accessibility tree (a11y). Arrows indicate the change from HTML to a11y within the same model. Left: For lower-capability (open-source) models, reducing the observation (h $\rightarrow$ a) improves the success rate. Right: For higher-capability (proprietary) models, the success rate decreases instead.

A consistent trend in existing web agent research is to reduce the amount of observation information. The primary motivation is that HTML representations of web pages tend to be extremely long—for example, on WorkArena (a web agent benchmark), HTML pages range from a minimum of 40K to a maximum of 500K tokens (Drouin and others, 2024). Early LLMs could not directly take such long inputs due to context window limits (Deng and others, 2023). LLMs with million-token context windows, such as Gemini,11 1 https://gemini.google.com/ have emerged and can accept raw HTML directly. However, observation reduction remains dominant because task-irrelevant information impedes reasoning (Yang and others, 2025b). Various reduction strategies have been proposed, including extracting only task-relevant elements (Deng and others, 2023; Gur and others, 2024; Lù et al., 2024), converting pages to compact representations (Yang and others, 2025b; Lee and others, 2025), and adopting the accessibility tree (a11y) as a more compact alternative to HTML (Agashe and others, 2025; Wang and Liu, 2025). When historical observations from past steps are also used, reduction is similarly applied (Yang and others, 2025b; Lai and others, 2024).

In this work, we revisit the trend of observation reduction. The conventional rationale for reduction was rooted in the limited long-context capability of earlier LLMs. Recent LLMs, however, have demonstrated substantially improved ability to process long inputs (Wang, 2025), and extended chain-of-thought reasoning at inference time further improves their capability (Zhu and others, 2025). Motivated by these advances, we investigate the following research question using state-of-the-art LLMs: How does the effect of observation reduction vary with model capability and the amount of thinking tokens at inference time?

Our experiments on WorkArena L1 (Drouin and others, 2024) reveal that observation reduction is not universally beneficial—the optimal observation representation depends on model capability and thinking token budget: compact observations (a11y) are preferable for lower-capability models, while detailed observations (HTML) are advantageous for higher-capability models (Figure 1, §3.2) 22 2 In our experimental setup, proprietary models serve as a proxy for higher-capability models and open-source models for lower-capability models. ; moreover, increasing thinking tokens further amplifies the benefit of HTML. Action analysis suggests that higher-capability models exploit layout information in HTML for better action grounding, while lower-capability models suffer from increased hallucination under longer inputs (§3.3). Incorporating observation history improves performance across most models and settings, and a diff-based representation offers a token-efficient alternative (§3.4). Based on these findings, we suggest practical guidelines for web agent system design: we recommend adaptively selecting observation representations based on model capability and thinking token budget, and encourage incorporating observation history with diff-based representations as a token-efficient alternative.

## 2 Problem Setting

Following Yang and others (2025b), we formalize the web agent task as a partially observable Markov decision process (POMDP): $\langle S,A,O,P,R,\Omega\rangle$ , where $S$ is the state space, $A$ is the action space, $O$ is the observation space, $P:S\times A\rightarrow S$ is the state transition function, $R:S\times A\times S\rightarrow\mathbb{R}$ is the reward function, and $\Omega:S\rightarrow O$ is the observation function.

An LLM-based agent operates as follows. At each step $t$ , the agent selects an action $a_{t}\in A$ according to the policy $\pi_{\text{LLM}}(a_{t}\mid I,o_{t},\{(o_{i},a_{i})\}_{i=0}^{t-1})$ , where $I$ is the system prompt containing the task instruction, $o_{t}$ is the current observation, and $\{(o_{i},a_{i})\}_{i=0}^{t-1}$ is the history of past observation–action pairs. Upon executing action $a_{t}$ , the environment transitions to the next state $s_{t+1}=P(s_{t},a_{t})$ according to the transition function, and the agent receives a new observation $o_{t+1}=\Omega(s_{t+1})$ via the observation function. The observation $o_{t+1}$ is provided as a textual representation of the web page (HTML or accessibility tree) or as a rendered image (screenshot). This process repeats until the agent executes a termination action or the maximum number of steps $T_{\max}$ is reached. Task success is evaluated by the reward $r=R(s_{T},a_{T},s_{T+1})$ upon episode termination, where $T$ denotes the terminal step.

In this work, we vary the following two aspects of how observations are fed into the policy $\pi_{\text{LLM}}$ :

- 1.

Observation representation: The current observation $o_{t}$ is represented as HTML, an accessibility tree (a11y), a screenshot, or a combination thereof.

- 2.

Observation history length: The number of past observations $\{o_{i}\}_{i<t}$ included as history.

We evaluate these observation design choices in combination with different language models and varying amounts of thinking tokens.

## 3 Experiments

### 3.1 Experimental Setup

Benchmark. We evaluate on WorkArena (Drouin and others, 2024; Boisvert and others, 2024), a web agent benchmark built on the ServiceNow platform33 3 https://www.servicenow.com/ that assesses the ability to perform everyday workplace tasks such as ordering items and filling out forms. We use WorkArena L1, which consists of 330 tasks in total: 33 task types $\times$ 10 seeds.

Language Models. To assess the effect of observation design, we consider a diverse set of model configurations along the following axes: (1) Model capability: we compare higher-capability frontier proprietary models against lower-capability open-source models, and compare models of different sizes within the same family (e.g., gpt-oss-120b vs. gpt-oss-20b). (2) Amount of thinking tokens: we vary the parameter that controls the amount of thinking tokens for each reasoning model where applicable. For proprietary models, we use claude-sonnet-4-6 44 4 https://www.anthropic.com/news/claude-sonnet-4-6, gpt-5.1-2025-11-13 55 5 https://openai.com/index/gpt-5-1/, o3-mini-2025-01-31 66 6 https://openai.com/index/openai-o3-mini/, and gemini-2.5-flash (Team, 2025). For open-source models, we use gpt-oss-120b/20b (OpenAI, 2025), the Qwen3-VL family (235B-A22B-Thinking, 30B-A3B-Thinking, 30B-A3B-Instruct) (Yang and others, 2025a), and the Llama-3.1 family (70B-Instruct, 8B-Instruct) (Team, 2024). For gpt-5.1, o3-mini, and gpt-oss, thinking tokens are controlled via the reasoning_effort parameter (low/high); for gemini, via the thinking_budget parameter (128/16384). The Llama-3.1 family is included to evaluate performance of older-generation models that are not trained as reasoning models.

Action Grounding. Grounding refers to the process of mapping the action string produced by the agent to an actual browser operation. We adopt id-based grounding: the agent specifies an action $a_{t}$ by referencing the id of a DOM element present in the current observation $o_{t}$ . Each action is expressed as a string combining an action type, a target element id, and optional parameters (e.g., fill(’12’, ’hello’)). Actions are executed via Playwright,77 7 https://playwright.dev/ a browser automation framework. We use AgentLab (de Chezelles and others, 2025; Drouin and others, 2024), a framework for running and managing web agent experiments. The maximum number of steps is set to $T_{\max}=15$ following the framework default; tasks that exceed this limit are treated as failures.

Table 1: Task success rate (%) by observation representation and model, evaluated on Workarena L1. Numbers in parentheses indicate the difference from a11y. N/A indicates that the model does not support image input. Values in parentheses after model names indicate reasoning_effort (high/low) or thinking_budget. Input token counts are provided as reference values for gpt-5.1 (high), including image inputs.

a11y html a11y+scrs

#input tokens 6720 56653 7446

Higher-Capability Models (Proprietary)

Claude Sonnet 4.6 52.4 67.0 (+14.6) 53.6 (+1.2)

gpt-5.1-2025-11-13 (high) 55.8 73.3 (+17.5) 59.7 (+3.9)

gpt-5.1-2025-11-13 (low) 49.1 57.9 (+8.8) 55.8 (+6.7)

gemini-2.5-flash (budget=16384) 45.5 56.7 (+11.2) 48.5 (+3.0)

gemini-2.5-flash (budget=128) 28.5 34.5 (+6.0) 32.7 (+4.2)

o3-mini-2025-01-31 (high) 39.7 32.1 (-7.6) N/A

Lower-Capability Models (Open-Source)

gpt-oss-120b (high) 46.7 38.8 (-7.9) N/A

gpt-oss-120b (low) 33.6 24.8 (-8.8) N/A

gpt-oss-20b (high) 46.4 27.6 (-18.8) N/A

gpt-oss-20b (low) 20.0 18.2 (-1.8) N/A

Qwen3-VL-235B-A22B-Thinking 45.0 39.8 (-5.2) 44.8 (-0.2)

Qwen3-VL-30B-A3B-Thinking 33.2 26.1 (-7.1) 33.9 (+0.7)

Qwen3-VL-30B-A3B-Instruct 22.7 25.5 (+2.8) 24.8 (+2.1)

Llama-3.1-70B-Instruct 18.2 3.6 (-14.6) N/A

Llama-3.1-8B-Instruct 0.0 1.2 (+1.2) N/A

Table 2: Task success rate (%) by observation format (Workarena L1). Comparing xml format vs indent format for both a11y and html representations. For each format group, the a11y column shows the baseline success rate, and the html column shows the difference from a11y in parentheses.

xml indent

a11y html a11y html

Higher-Capability Models (Proprietary)

gpt-5.1-2025-11-13 (high) 60.0 73.3 (+13.3) 55.8 76.7 (+20.9)

gemini-2.5-flash (budget=16384) 51.5 56.7 (+5.2) 45.5 48.2 (+2.7)

Lower-Capability Models (Open-Source)

gpt-oss-120b (high) 50.0 38.8 (-11.2) 46.7 26.7 (-20.0)

gpt-oss-20b (high) 48.2 27.6 (-20.6) 46.4 14.8 (-31.5)

Table 3: Impact of switching from a11y to HTML on task success rate by category (WorkArena L1). Values show the contribution to overall benchmark score (%) when using HTML instead of a11y. Task categories: Form (record creation), Filter (list filtering), Sort (list sorting), Dashboard (chart reading), Navi (menu navigation), Knowledge (KB search), Catalog (service ordering).

Form Filter Sort Dashboard Navi Knowledge Catalog

#tasks 50 60 60 40 20 10 90

Higher-Capability Models (Proprietary)

Claude Sonnet 4.6 +1.2 +11.8 0.0 +1.5 0.0 0.0 0.0

gpt-5.1-2025-11-13 (high) -1.8 +8.8 +6.7 +3.0 0.0 +0.3 +0.6

gpt-5.1-2025-11-13 (low) -4.5 +7.0 +3.9 +2.7 0.0 0.0 -0.3

gemini-2.5-flash (budget=16384) -1.8 +6.1 +5.5 +3.3 0.0 -0.9 -0.9

gemini-2.5-flash (budget=128) -2.4 +4.2 +3.0 +1.2 0.0 -1.2 +1.2

o3-mini-2025-01-31 (high) -5.5 +0.9 +2.4 +1.2 0.0 -1.2 -5.5

Lower-Capability Models (Open-Source)

gpt-oss-120b (high) -3.9 +1.8 -2.4 +2.7 0.0 0.0 -6.1

gpt-oss-120b (low) -3.0 +0.6 +0.6 +0.9 0.0 -0.9 -7.0

gpt-oss-20b (high) -5.5 +0.9 -2.1 +0.3 0.0 -1.2 -11.2

gpt-oss-20b (low) -1.8 0.0 0.0 +1.5 0.0 -0.6 -0.9

Qwen3-VL-235B-A22B-Thinking -10.3 +0.6 +1.8 -5.8 0.0 -3.3 -33.6

Qwen3-VL-30B-A3B-Instruct -1.8 0.0 0.0 -0.3 0.0 -0.6 +5.5

Qwen3-VL-30B-A3B-Thinking -9.4 0.0 0.0 -3.9 0.0 -2.7 -24.2

Llama-3.1-70B-Instruct -2.4 0.0 -0.9 -2.7 0.0 -2.4 -6.1

Llama-3.1-8B-Instruct 0.0 0.0 +0.9 +0.3 0.0 0.0 0.0

### 3.2 Observation Representation and Task Success Rate

We examine the effect of observation representation on task success rate. The accessibility tree (a11y) is a compact representation that focuses primarily on interactive elements of the web page. HTML, by contrast, preserves a more detailed document structure of the web page; in our experimental setting, it additionally includes CSS-based layout information. Using a11y as the baseline, we evaluate two alternatives: (1) replacing a11y with HTML, and (2) augmenting a11y with screenshots (a11y+scrs). Results are shown in Table 1.

Model Capability and Observation Representation. The results indicate that higher-capability models (proprietary in our setup) are better able to exploit more information-rich observations. Higher-capability proprietary models (claude-sonnet-4-6, gpt-5.1, gemini-2.5-flash) perform better with HTML than with a11y, whereas lower-capability open-source models frequently degrade when HTML is used. A similar trend holds for screenshots: augmenting a11y with screenshots yields larger gains for higher-capability models. Within the same model family, smaller models tend to suffer greater performance degradation from HTML. For example, with reasoning_effort=high, gpt-oss-20b degrades more than gpt-oss-120b. Even among proprietary models, older and more compact reasoning models such as o3-mini suffer from HTML, likely due to their limited long-context capability.

Thinking Tokens and Observation Representation. More thinking tokens lead to greater ability to exploit detailed observations such as HTML. For instance, increasing reasoning_effort from low to high for gpt-5.1 yields a larger performance gain from HTML. A similar trend is observed for gemini-2.5-flash, where increasing thinking_budget from 128 to 16384 also leads to a larger gain from HTML.

XML vs. Indented Format. In Table 1, the a11y representation was provided in an indented format where hierarchy is expressed via space indentation, whereas HTML was provided in XML-tagged format. This raises the possibility that the observed performance differences stem from the format choice rather than the information content of each representation. For instance, if proprietary models are trained more heavily on XML-formatted data, their format preference may manifest in task performance. To examine this possibility, we evaluated a11y and HTML in both XML and indented formats. As shown in Table 2, regardless of format, higher-capability models consistently benefit from HTML while lower-capability models consistently degrade.

Analysis by Task Category. Table 3 shows the per-category change in task success rate when switching from a11y to HTML on WorkArena L1. A consistent pattern emerges across models: HTML improves performance on Filter (list filtering), Sort (list sorting), and Dashboard (chart reading) tasks, while it degrades performance on Form (record creation), Knowledge (KB search), and Catalog (service ordering) tasks. For higher-capability models, the gains on the former categories are large and the losses on the latter are small, resulting in an overall improvement with HTML. For lower-capability models, the opposite pattern holds, leading to an overall degradation. To better understand why higher-capability models achieve large gains on HTML-friendly categories while limiting losses on HTML-unfriendly ones, we analyze agent behavior at the step level in §3.3.

Figure 2: Grounding error counts and their breakdown for a11y and HTML across models (WorkArena L1).

Table 4: Grounding error categories.

Type Description

API Format Errors

Scroll Invalid argument passed to scroll()

Click Invalid argument passed to click()

Select Invalid argument passed to select_option()

Element Availability Errors

Not Found The specified element does not exist in the DOM

Not Visible The specified element is hidden via CSS

Not Enabled The specified element is disabled (e.g., via the disabled attribute)

Wrong Type The specified element does not accept the requested operation

Option Not Found The option value specified in select_option() does not exist in the element

Execution Errors

Intercepted A click operation is intercepted by an overlapping element

Invalid URL The URL specified in goto() is invalid

### 3.3 Grounding Error Analysis

Setup. To further analyze the results of §3.2, we examine grounding errors—cases in which an action is not successfully accepted by the environment or fails a precondition check prior to execution. Specifically, we count actions that raise an exception during Playwright execution as grounding errors, and classify them based on string patterns in the error messages, as described in Table 4. Errors caused by environment-side issues such as network failures or timeouts—where the action itself was correctly formed—are excluded from the analysis. We compare representative models from each capability group: gemini-2.5-flash(budget=128/16384) and gpt-5.1(high/low), for which switching from a11y to HTML improved success rate, against gpt-oss-20b (high), which exhibited the largest performance degradation under HTML.

Error Breakdown. Figure 2 shows the number of grounding errors and their breakdown for a11y and HTML across models. When switching from a11y to HTML, the total number of errors decreases for the higher-capability models gemini-2.5-flash and gpt-5.1, while it increases for the lower-capability model gpt-oss-20b (high). This suggests that switching to HTML affects grounding success, which in turn contributes to the difference in task success rates. We next examine the breakdown of error types to understand the sources of this difference.

- •

Not-found errors occur when the agent specifies the id of an element that does not exist in the DOM. This type of error increases substantially for gpt-oss-20b (high), while remaining nearly unchanged for gpt-5.1. Among Gemini models, gemini-2.5-flash (budget=128) shows a large increase, whereas gemini-2.5-flash (budget=16384) successfully suppresses it. These results suggest that either a more capable model or a larger thinking budget helps suppress hallucinated id references under longer inputs.

- •

Intercepted errors occur when a click operation is intercepted by an overlapping element. This type of error decreases across all models, particularly for higher-capability ones. The HTML used in our experiments includes CSS style information specifying z-index, which determines the stacking order of elements. We hypothesize that higher-capability models leverage this information to infer element overlap and thereby avoid such errors.

Summary. These findings indicate that higher-capability models can reduce grounding errors by exploiting the rich layout information in HTML, whereas lower-capability models experience increased errors due to hallucination under longer inputs.

Table 5: Task success rate (%) by observation history setting and model, evaluated on Workarena L1. Numbers in parentheses indicate the difference from hist0. hist0: no history; hist4/hist9: history from the past 4/9 steps. In the hist9 column, full denotes adding the complete observation history, and diff denotes adding only the diff from the previous step. Values in parentheses after model names indicate reasoning_effort (high/low) or thinking_budget. Input token counts are provided as reference values for gpt-5.1 (high).

hist0 hist4 hist9 full/diff

n_input_tokens 6720 26184 39011 / 13670

Higher-Capability Models (Proprietary)

gpt-5.1-2025-11-13 (high) 55.8 58.8 (+3.0) 58.8 (+3.0) / 58.8 (+3.0)

gpt-5.1-2025-11-13 (low) 49.1 53.0 (+3.9) 50.9 (+1.8) / 53.3 (+4.2)

o3-mini-2025-01-31 (high) 39.7 43.0 (+3.3) 43.3 (+3.6) / 46.1 (+6.4)

gemini-2.5-flash (budget=16384) 45.5 48.2 (+2.7) 50.0 (+4.5) / 48.2 (+2.7)

gemini-2.5-flash (budget=128) 28.5 39.4 (+10.9) 39.4 (+10.9) / 33.3 (+4.8)

Lower-Capability Models (Open-Source)

gpt-oss-120b (high) 46.7 49.1 (+2.4) 48.5 (+1.8) / 46.4 (-0.3)

gpt-oss-120b (low) 33.6 37.6 (+4.0) 40.0 (+6.4) / 39.1 (+5.5)

gpt-oss-20b (high) 46.4 46.7 (+0.3) 48.8 (+2.4) / 49.1 (+2.7)

gpt-oss-20b (low) 20.0 23.9 (+3.9) 23.9 (+3.9) / 24.2 (+4.2)

### 3.4 Observation History and Task Success Rate

We investigate the effect of incorporating past observation history. The observation representation is fixed to a11y, and we compare three settings: no history (hist0), history from the past 4 steps (hist4), and history from the past 9 steps (hist9). For hist9, we further compare two variants: adding the full observation (full) versus adding only the character-level diff from the previous observation (diff). Results are shown in Table 5.

Effect of Observation History. The results show that adding observation history improves performance for most models and settings. Furthermore, o3-mini and gemini-2.5-flash show additional gains from hist9 over hist4, suggesting further room for improvement with longer histories.

Effect of Diff-Based Observation History. Comparing the full and diff variants of hist9, we find that the diff format also generally improves over hist0. Moreover, for gpt-5.1 (low) and o3-mini, the diff format achieves performance comparable to or better than the full format. Since the diff format reduces input token count to approximately one-third, it constitutes an efficient alternative to full history.

Analysis of Action Repetition To investigate why observation history improves performance, we measure the rate at which the agent repeats the same action as in the previous step (repetition rate). As shown in Figure 3, a lower repetition rate is associated with a higher success rate, and adding observation history tends to reduce the repetition rate, particularly when the amount of thinking tokens is small. This suggests that observation history helps the agent better track task progress and avoid redundant actions.

Figure 3: Relationship between action repetition rate and task success rate (WorkArena L1). The x-axis shows the rate of actions identical to the previous step; the y-axis shows the task success rate. The number on each data point indicates the length of observation history included as input.

## 4 Related Work

Observation Representation and Reduction. In web agents, web pages are commonly represented as HTML, which tends to be extremely long, posing challenges in terms of LLM context length constraints and inference efficiency. Consequently, existing work has widely adopted observation reduction and selection strategies. Deng and others (2023) proposed a framework that trains a cross-encoder to score the relevance between the task instruction and each HTML element, feeding only the top- $k$ elements to the LLM. Similarly, Gur and others (2024) adopt a two-stage pipeline in which a dedicated model first extracts and summarizes relevant elements from HTML before passing them to the LLM. Lù et al. (2024) proposed a similar ranking approach but using a dual encoder to score the relevance between DOM elements and the full action history. Zhang and others (2026) proposed generating a Python scoring program from the current sub-task to filter HTML elements programmatically, avoiding direct LLM processing of the full HTML. Besides HTML-based approaches, several studies have adopted the accessibility tree (a11y), a more compact alternative to HTML. Yang and others (2025b) showed that simplifying the a11y by removing task-irrelevant elements improves agent performance. Lee and others (2025) proposed contextualizing a11y observations by describing the role of each element in natural language. Kerboua and others (2025) proposed using a smaller LLM to selectively extract task-relevant lines from a11y observations. Research using screenshots only (Li and others, 2025; Cheng and others, 2024; Hoscilowicz and Janicki, 2025) and work combining screenshots with textual observations (He and others, 2024; Agashe and others, 2025; Lu and others, 2024; Lù et al., 2024) have also been proposed; in all cases, compressed or filtered representations are used for the textual observations.

Observation History and Reduction. Leveraging past observations and actions as history in addition to the current observation has been shown to improve performance (Gao and others, 2025). However, retaining the full history is impractical, and various strategies have been explored to control the amount of history information. Lai and others (2024) reduce redundancy by storing only action logs and local HTML snippets as history. Yang and others (2025b) retain only observations from steps relevant to the currently active subtask.

Summary. In existing work, reducing the amount of information has been the dominant design choice for both single-step observations and observation history. However, whether such reduction remains beneficial for models with strong long-context capabilities or with larger thinking token budgets has not been sufficiently examined.

## 5 Conclusion

We revisited the trend of observation reduction in web agents and demonstrate that observation reduction is not always beneficial for task performance—the optimal observation representation depends on model capability and thinking token budget: higher-capability models with larger thinking budgets benefit from detailed HTML observations, with gains of up to 17.5 percentage points over compact accessibility tree (a11y) representations, while the benefit grows further as the thinking token budget increases. Our error analysis suggests that stronger models are able to exploit layout-related cues available in HTML—such as CSS z-index information—to avoid grounding errors, while weaker models tend to hallucinate element references under longer inputs. Unlike observation representation, incorporating observation history consistently improves performance across almost all models and settings, and a diff-based history representation offers a token-efficient alternative.

These findings motivate the following suggestions for web agent system design.

- •

Adaptively select observation representation. We recommend selecting the observation representation based on model capability and reasoning budget. For high-capability models with larger thinking budgets, the performance gains from richer HTML observations may well outweigh the increased inference cost. Furthermore, as shown in §3.2, the benefit of HTML varies by task category, suggesting that task characteristics should also be taken into account.

- •

Incorporate observation history. We encourage incorporating observation history; when inference cost is a concern, diff-based history offers a practical token-efficient alternative.

## Limitation

- •

Generalizability across websites and domains. Our experiments are conducted on WorkArena L1, a benchmark built on a real-world website (ServiceNow), which provides a degree of practical validity. Nevertheless, whether the findings generalize to other websites and domains remains unverified.

- •

Grounding scheme. Our experiments are limited to id-based grounding; whether similar trends hold under coordinate-based or other grounding schemes is an open question.

- •

Mechanism of HTML benefit. Our error analysis suggests that CSS layout cues contribute to the HTML benefit, but this has not been directly verified through ablation and additional factors may be involved.

- •

Observation history with richer representations. Our history experiments fix the observation representation to a11y, leaving the effect of combining richer observations such as HTML with history unexplored. Furthermore, WorkArena L1 involves tasks of up to 15 steps, and the effect of observation history on longer-horizon tasks remains unknown.

## References

- Agashe et al. (2025) S. Agashe et al. Agent S: an open agentic framework that uses computers like a human. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, External Links: Link Cited by: §1, §4.

- Boisvert et al. (2024) L. Boisvert et al. WorkArena++: towards compositional planning and reasoning-based common knowledge work tasks. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, External Links: Link Cited by: §3.1.

- Cheng et al. (2024) K. Cheng et al. SeeClick: harnessing GUI grounding for advanced visual GUI agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pp. 9313–9332. External Links: Link, Document Cited by: §4.

- de Chezelles et al. (2025) T. L. S. de Chezelles et al. The browsergym ecosystem for web agent research. Transactions on Machine Learning Research. External Links: ISSN 2835-8856, Link Cited by: §3.1.

- Deng et al. (2023) X. Deng et al. Mind2Web: towards a generalist agent for the web. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, External Links: Link Cited by: §1, §4.

- Drouin et al. (2024) A. Drouin et al. WorkArena: how capable are web agents at solving common knowledge work tasks?. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, External Links: Link Cited by: §1, §1, §3.1, §3.1.

- Gao et al. (2025) X. Gao et al. Chain-of-memory: enhancing GUI agents for cross-application navigation. CoRR abs/2506.18158. External Links: Link, Document, 2506.18158 Cited by: §4.

- Gur et al. (2024) I. Gur et al. A real-world webagent with planning, long context understanding, and program synthesis. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, External Links: Link Cited by: §1, §4.

- He et al. (2024) H. He et al. WebVoyager: building an end-to-end web agent with large multimodal models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6864–6890. External Links: Link, Document Cited by: §4.

- Hoscilowicz and Janicki (2025) J. Hoscilowicz and A. Janicki ClickAgent: enhancing UI location capabilities of autonomous agents. In Proceedings of the 26th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pp. 471–476. External Links: Link Cited by: §4.

- Kerboua et al. (2025) I. Kerboua et al. FocusAgent: simple yet effective ways of trimming the large context of web agents. CoRR abs/2510.03204. External Links: Link, Document, 2510.03204 Cited by: §4.

- Lai et al. (2024) H. Lai et al. AutoWebGLM: A large language model-based web navigating agent. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2024, Barcelona, Spain, August 25-29, 2024, pp. 5295–5306. External Links: Link, Document Cited by: §1, §4.

- Lee et al. (2025) D. Lee et al. Learning to contextualize web pages for enhanced decision making by LLM agents. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, External Links: Link Cited by: §1, §4.

- Li et al. (2025) H. Li et al. AutoGUI: scaling GUI grounding with automatic functionality annotations from llms. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pp. 10323–10358. External Links: Link Cited by: §4.

- Lù et al. (2024) X. H. Lù, Z. Kasner, and S. Reddy WebLINX: real-world website navigation with multi-turn dialogue. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, External Links: Link Cited by: §1, §4.

- Lu et al. (2024) Y. Lu et al. OmniParser for pure vision based GUI agent. CoRR abs/2408.00203. External Links: Link, Document, 2408.00203 Cited by: §4.

- Ning et al. (2025) L. Ning et al. A survey of webagents: towards next-generation ai agents for web automation with large foundation models. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pp. 6140–6150. External Links: Link, Document Cited by: §1.

- OpenAI (2025) OpenAI Gpt-oss-120b & gpt-oss-20b model card. CoRR abs/2508.10925. External Links: Link, Document, 2508.10925 Cited by: §3.1.

- Team (2025) G. Team Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. CoRR abs/2507.06261. External Links: Link, Document, 2507.06261 Cited by: §3.1.

- Team (2024) L. Team The llama 3 herd of models. CoRR abs/2407.21783. External Links: Link, Document, 2407.21783 Cited by: §3.1.

- Wang and Liu (2025) X. Wang and B. Liu OSCAR: operating system control via state-aware reasoning and re-planning. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, External Links: Link Cited by: §1.

- Wang (2025) Y. Wang Beyond isolated capabilities: bridging long cot reasoning and long-context understanding. CoRR abs/2507.14849. External Links: Link, Document, 2507.14849 Cited by: §1.

- Yang et al. (2025a) A. Yang et al. Qwen3 technical report. CoRR abs/2505.09388. External Links: Link, Document, 2505.09388 Cited by: §3.1.

- Yang et al. (2025b) K. Yang et al. AgentOccam: A simple yet strong baseline for llm-based web agents. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025, External Links: Link Cited by: §1, §2, §4, §4.

- Yao et al. (2023) S. Yao et al. ReAct: synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, External Links: Link Cited by: §1.

- Zhang et al. (2026) J. Zhang et al. Prune4Web: DOM tree pruning programming for web agent. In Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20-27, 2026, pp. 34710–34718. External Links: Link Cited by: §4.

- Zhu et al. (2025) D. Zhu et al. Chain-of-thought matters: improving long-context language models with reasoning path supervision. CoRR abs/2502.20790. External Links: Link, Document, 2502.20790 Cited by: §1.

Experimental support, please view the build logs for errors. Generated by L A T E xml .

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

- Click the "Report Issue" () button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

We gratefully acknowledge support from our major funders, member institutions, , and all contributors. Major funding support from
