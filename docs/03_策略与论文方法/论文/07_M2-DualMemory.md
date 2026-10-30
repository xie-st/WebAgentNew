# M2: Dual-Memory Augmentation for Long-Horizon Web Agents via Trajectory Summarization and Insight Retrieval

- arXiv: 2603.00503
- URL: https://arxiv.org/html/2603.00503
- 来源：arXiv HTML（LaTeXML）自动转换，公式以 $...$ 内联保留，图表编号可能与原文有出入。

---

M2: Dual-Memory Augmentation for Long-Horizon Web Agents via Trajectory Summarization and Insight Retrieval

##### Report GitHub Issue

Title:

Content selection saved. Describe the issue below:

Description: arXiv is now an independent nonprofit! Learn more Back to arXiv License: CC BY 4.0 arXiv:2603.00503v1 [cs.CV] 28 Feb 2026

# M2: Dual-Memory Augmentation for Long-Horizon Web Agents via

Trajectory Summarization and Insight Retrieval

Dawei Yan Affiliation: Northwestern Polytechnical University, Xi’an, China Affiliation: AI Business, Alibaba Group, Hangzhou, China Haokui Zhang* Affiliation: Northwestern Polytechnical University, Xi’an, China Guangda Huzhang Affiliation: AI Business, Alibaba Group, Hangzhou, China Yang Li Affiliation: AI Business, Alibaba Group, Hangzhou, China Yibo Wang Affiliation: AI Business, Alibaba Group, Hangzhou, China Qing-Guo Chen Affiliation: AI Business, Alibaba Group, Hangzhou, China Zhao Xu Affiliation: AI Business, Alibaba Group, Hangzhou, China Weihua Luo Affiliation: AI Business, Alibaba Group, Hangzhou, China Ying Li Affiliation: Northwestern Polytechnical University, Xi’an, China Wei Dong Affiliation: Xi’an University of Architecture and Technology, Xi’an, China Chunhua Shen Affiliation: Zhejiang University, Hangzhou, China Correspondence to: hkzhang@nwpu.edu.cn

###### Abstract

Multimodal Large Language Models (MLLMs) based agents have demonstrated remarkable potential in autonomous web navigation. However, handling long-horizon tasks remains a critical bottleneck. Prevailing strategies often rely heavily on extensive data collection and model training, yet still struggle with high computational costs and insufficient reasoning capabilities when facing complex, long-horizon scenarios. To address this, we propose M2, a training-free, memory-augmented framework designed to optimize context efficiency and decision-making robustness. Our approach incorporates a dual-tier memory mechanism that synergizes Dynamic Trajectory Summarization (Internal Memory) to compress verbose interaction history into concise state updates, and Insight Retrieval Augmentation (External Memory) to guide the agent with actionable guidelines retrieved from an offline insight bank. Extensive evaluations across WebVoyager and OnlineMind2Web demonstrate that M2 consistently surpasses baselines, yielding up to a 19.6% success rate increase and 58.7% token reduction for Qwen3-VL-32B, while proprietary models like Claude achieve accuracy gains up to 12.5% alongside significantly lower computational overhead.

###### Keywords:

Machine Learning, ICML

Model SFT RL Acc.

UI-TARS-1.5 ✓ ✓ 84.8⋆

OpenAI CUA ✗ ✓ 87.0⋆

M2 (Ours) ✗ ✗ 86.0

Table 1: Comparison with SOTA methods on WebVoyager dataset. $\star$ : Official metrics from the original release; note that some tasks within this benchmark are now infeasible due to website updates.

Figure 1: Average Token cost and accuracy of baseline and M2 across models on WebVoyager.

Figure 2: Overview of the proposed framework. (a) The baseline agent operates with raw context containing redundant visual history and verbose interaction text. This creates high computational overhead and introduces noise that may impair decision-making. Our method M2 incorporates two key mechanisms: In-Mem and Ex-Mem. (b) The In-Mem module prompts the agent to self-summarize historical steps into a concise context. Simultaneously, (c) the Ex-Mem module retrieves actionable insights from historically successful trajectories based on query similarity. (d) This dual-memory approach provides explicit guidance for the current execution step, significantly reducing errors and yielding tangible performance gains.

## 1 Introduction

Recent advancements in MLLMs have catalyzed progress in autonomous web navigation, allowing agents to perform intricate tasks within open-ended web environments (Chen et al., 2024; Xu et al., 2024; Qi et al., 2024; Liu et al., 2025b; Jain et al., 2025; Zhang et al., 2025a). However, real-world web tasks are inherently long-horizon, often requiring numbers of interaction steps to navigate through dynamic web pages, fill forms, and process information. Current mainstream approaches typically employ a “full-context” strategy, concatenating the entire history of HTML screenshots, and interactive text into the prompt (Qin et al., 2025; Li et al., 2025; Huang et al., 2025). This leads to two critical issues: (1) Exorbitant Computational Cost: The accumulation of verbose history and high-resolution screenshots causes the context window to grow rapidly, imposing severe pressure on inference resources. (2) Performance Degradation: An overly long and noisy context often distracts the model, burying critical task-relevant cues under redundant historical information, a phenomenon known as “lost-in-the-middle” (Liu et al., 2024).

To mitigate the context explosion, recent research has explored memory-augmented mechanisms (Yu et al., 2025; Wang et al., 2025b; Cheng et al., 2025; Liu et al., 2025a; Zhou et al., 2025). The prominent direction involves training the model to autonomously maintain a condensed memory state via Supervised Fine-Tuning (SFT), Reinforcement Learning (RL), or constructing complex memory architectures composed of multiple collaborative agents. While these methods effectively reduce context length, they introduce considerable drawbacks: (1) Intensive Data and Training Overhead: Optimizing agents through SFT or RL requires massive computational resources, extensive trajectory datasets for training, and delicate reward shaping mechanisms to ensure effective convergence. (2) High Operational and Deployment Costs: Systems relying on multiple agents suffer from substantial overhead in inter-agent communication, increased complexity in deployment, and elevated invocation costs during real-time execution. Consequently, a training-free and lightweight approach is highly desirable for scalable web agents.

Another line of research focuses on external memory banks and graph-based structures to retrieve relevant context. However, directly migrating these general-purpose memory architectures (Wang et al., 2025a; Kang et al., 2025; Zhang et al., 2025b) to web agents faces distinct domain gaps. First, the interaction nature is different. Unlike conversational agents that need to track multi-turn dialogue history and user personas, web navigation is predominantly execution-oriented. Once the initial instruction is given, there is rarely new user input; tracking detailed interaction history (e.g., every scroll or click) is often redundant unless it contributes to the current subgoal. Second, visual information presents unique challenges. Current multimodal memory systems either store text-only abstractions (Wang and Chen, 2025) or treat visual history as continuous video frames (He et al., 2024a). However, web navigation is visually discontinuous—a single click can lead to a completely different page layout, rendering frame-based consistency assumptions invalid. Furthermore, raw screenshots contain excessive visual noise (e.g., ads, sidebars) irrelevant to the task. Storing raw visual history incurs high token costs with low information density, making simple screenshot buffering inefficient.

To address these challenges, we propose a novel, training-free framework that optimizes context efficiency and execution robustness through a dual-tier memory mechanism named M2. As shown in Fig. 2, it includes two components: (1) Dynamic Trajectory Summarization (Internal Memory) introduces a prompt-driven self-summarization mechanism. Instead of relying on a sliding window of multiple raw observations that consume excessive tokens, the agent is instructed to distill its own thoughts, actions, and page feedback into concise textual abstractions at each step. By maintaining a recursive chain of these summaries and discarding redundant historical screenshots, we ensure the agent retains a clear sense of progress within a lean context window. Complementing this, (2) Insight Retrieval Augmentation (External Memory) provides global strategic foresight. We construct an offline insight bank by distilling actionable interaction rules and defensive hints from a vast collection of successful trajectories. During inference, the agent retrieves the most relevant insights via semantic similarity based on the user query. This allows the agent to benefit from cross-trajectory experience, effectively navigating complex UI structures and avoiding common pitfalls without any additional training or fine-tuning. As demonstrated in Table 1, our proposed M2 framework achieves competitive performance compared to state-of-the-art models that rely on extensive SFT and RL, while completely bypassing the associated training overhead.

Empirical evaluations on WebVoyager (He et al., 2024b) and OnlineMind2Web (Xue et al., 2025) demonstrate that our framework achieves a consistent improvement in both execution success and resource efficiency. The open-source Qwen3-VL-32B (Bai et al., 2025) model exhibits the most substantial gains, with success rates surging by 16.2% on WebVoyager and 19.6% on OnlineMind2Web, while simultaneously reducing token overhead by over 57% in both benchmarks. This efficiency trend extends to frontier proprietary models: Claude-3.7-Sonnet (anthropic, 2025a) and Claude-Sonnet-4 (anthropic, 2025b) achieve accuracy increments of up to 12.5% alongside significant token reductions ranging from 30.3% to 55.0% (see Fig. 1). These results validate that our dual-memory architecture effectively decouples task performance from context growth, enabling high-fidelity navigation with sustainable computational costs.

In summary, the major contributions are as follows:

- •

Training-Free Dual-Memory Architecture: We propose a lightweight framework that integrates recursive internal tracking with external guidance, enabling efficient long-horizon navigation without the need for costly training or cumbersome multi-agent interaction.

- •

Intra-Trajectory Compression and Inter-Trajectory Retrieval: We introduce mechanisms to distill execution history into concise summary chains and retrieve cross-task expert insights, effectively mitigating information overload while enhancing decision robustness.

- •

Scalable Efficacy and Model Parity: We demonstrate that M2 enables sustainable high-fidelity navigation across diverse and dynamic web domains. Our training-free approach enables local models to match proprietary performance with superior token efficiency.

## 2 Method: Dual-Memory Augmentation

### 2.1 Overview

In this section, we present the Dual-Memory Augmentation framework M2designed to address the challenges of long-horizon web navigation.

#### 2.1.1 Formalization of Context

We define a web navigation task as a sequence of steps $t\in\{1,2,\dots,n\}$ , where $n$ denotes the final step of completion. In a conventional setting, at any given step $t$ , the agent operates within a context $C_{t}$ . This context typically consists of the system prompt $P_{sys}$ , the user query $Q$ , and the complete history of previous steps. Each step can be represented as a tuple $S_{t}=(T_{t},A_{t})$ , $T_{t}$ is the generated thought (reasoning), and $A_{t}$ is the executed action. Thus, the vanilla context used in mainstream works is:

$\begin{split}C_{t}=\{&P_{sys},Q,S_{1},S_{2},\dots,S_{t-k-1},\\ &S_{t-k},O_{t-k},\dots,S_{t-1},O_{t-1},O_{t}\}\end{split}$ (1)

where $O_{t}$ is the page observation with window size $k$ . As $t$ increases, the cardinality of $C_{t}$ grows indefinitely, leading to high token costs and the “lost-in-the-middle” effect. For specific details such as $P_{sys}$ , as well as examples of $Q$ , $O_{t}$ , and $S_{t}$ , and the action space of the Agent, please refer to Appendix F.

#### 2.1.2 The Dual-Memory Framework

Our approach redefines the context $C_{t}$ by introducing two specialized memory modules: Internal Memory ( $\mathcal{M}^{int}$ ) and External Memory ( $\mathcal{M}^{ext}$ ). The modified decision-making context at step $t$ is formulated as:

$C^{\prime}_{t}=\{P_{sys},Q,\mathcal{M}_{t}^{int},\mathcal{M}^{ext},O_{t}\}$ (2)

The interaction between these components is governed by the following logic $(O_{t-k:t-1}=\{O_{i}\}_{i=t-k}^{t-1})$ :

- •

Internal Memory $\mathcal{M}_{t}^{int}$ : This module maintains a dynamic trajectory summary. Instead of retaining the raw observations $O_{t-k:t-1}$ and $S_{1:t-1}$ which are often redundant and noisy, $\mathcal{M}_{t}^{int}$ stores a chain of concise state abstractions $s_{1:t-1}$ . Each $s_{t}$ is a distilled representation of $(O_{t},T_{t},A_{t})$ , ensuring that the internal context remains compact regardless of the task length.

- •

External Memory $\mathcal{M}^{ext}$ : This module facilitates cross-task experience retrieval. It provides the agent with task-relevant insights $\mathcal{I}$ retrieved from an offline Insight Bank. These insights, denoted as $\mathcal{M}^{ext}=\{\mathcal{I}_{1},\dots,\mathcal{I}_{i}\}$ , serve as global guidance to prevent the agent from repeating historical errors or falling into common pitfalls.

By substituting the growing list of historical steps with these two components, our framework ensures that $C_{t}$ remains within efficient token consumption while providing both local trajectory awareness and global strategic knowledge.

Figure 3: Comparison between Full Context and In Mem interaction paradigms. The Full Context (left) retains the entire sequence of raw screenshots and exhaustive historical text, leading to high redundancy. In contrast, the In Mem approach (right) employs a specific prompt to synthesize historical steps into a concise textual summary.

### 2.2 Dynamic Trajectory Summarization

Typically, HTML pages are replete with task-irrelevant elements—such as advertisements, sidebars, and navigational components unrelated to the current objective. These elements, when captured in full-page screenshots, dominate the visual information. Conversely, interactive targets like search bars or specific buttons often occupy only a negligible fraction of the screen. Consequently, these critical signals are easily overwhelmed by background noise, a persistent issue that is further exacerbated when processing sequences of multiple screenshots.

In contrast, human cognition retains historical interaction information in a much more concise manner. Rather than memorizing the specific visual details of previous pages, humans tend to form a high-level abstraction of their actions (e.g., ‘I clicked the search bar and entered text’). Inspired by this cognitive mechanism, we propose summarizing historical interaction steps to replace the verbose and potentially distracting visual context, thereby enabling the agent to make more precise decisions.

To maintain a lean context $C_{t}$ while preserving essential historical information, we propose a Dynamic Trajectory Summarization mechanism. This module replaces the raw, high-dimensional observation sequences with a sequence of semantic abstractions.

#### 2.2.1 Prompt-Driven State Abstraction

In our framework, we shift this burden from the raw input buffer to the agent’s self-reasoning capability. We augment the system prompt $P_{sys}$ with specific meta-instructions that mandate the agent to perform self-summarization alongside its primary decision-making. As shown in Fig. 3, the original system prompt is augmented with a new directive that instructs the agent to perform self-summarization. Specifically, at each step $t$ , the agent is required to output a triplet $(T_{t},A_{t},s_{t})$ based on the current context:

$(T_{t},A_{t},s_{t})=\text{Agent}(P^{in}_{sys},Q,\mathcal{M}_{t}^{int},O_{t})$ (3)

where $P^{in}_{sys}$ is the updated system prompt, and $\mathcal{M}_{t}^{int}$ is the internal memory consisting of a sequence of concise textual abstractions $s_{1:t-1}$ .

By integrating the summarization task directly into the execution loop, the agent treats the abstraction of $s_{t}$ as a prerequisite for its next action. The instructions within $P^{in}_{sys}$ guide the agent to distill $s_{t}$ from two perspectives:

- •

Visual Feedback: The observable changes and key information extracted from the page after the action.

- •

Action Execution: A brief record of the action $A_{t}$ performed on $O_{t}$ .

This approach allows the context $C_{t}$ to retain only the current observation $O_{t}$ . The historical trajectory $\{S_{1},S_{2},\dots,S_{t-1}\}$ and the multi observations $\{O_{t-k},\dots,O_{t-1}\}$ is effectively discarded and replaced by the internal memory $\mathcal{M}_{t}^{int}$ which is constructed by $s_{t}$ . This transformation ensures that the context length grows sub-linearly with the task horizon while maintaining a high density of task-relevant information. For details, please refer to the complete $P^{in}_{sys}$ and $s_{t}$ samples we provided in Appendix F.

#### 2.2.2 Iterative Memory Update

The core efficiency of our framework lies in the recursive transition of the internal memory state $\mathcal{M}^{int}$ , which allows the agent to maintain a persistent state without retaining bulky historical context. Unlike vanilla architectures that store the entire history of $O_{t-k:t-1}$ and $S_{1:t-1}$ , our method treats the internal memory as an append-only sequence of distilled summaries.

Formally, at the initialization step $t=1$ , the internal memory is empty, i.e., $\mathcal{M}_{1}^{int}=\emptyset$ . For any subsequent step $t>1$ , the internal memory $\mathcal{M}_{t}^{int}$ is constructed by iteratively appending the summary $s_{t}$ generated in the previous step to the existing memory state:

$\mathcal{M}_{t}^{int}=[s_{1},s_{2},\dots,s_{t-1}]$ (4)

The update rule for the internal memory state can be expressed as:

$\mathcal{M}_{t}^{int}=\mathcal{M}_{t-1}^{int}\oplus s_{t-1}$ (5)

where $\oplus$ denotes the sequence concatenation operator.

Crucially, once $s_{t-1}$ is successfully generated and integrated into $\mathcal{M}_{t}^{int}$ , the observation $O_{t-1}$ , the intermediate Thought $T_{t-1}$ and Action $A_{t-1}$ are permanently discarded from the active context window.

This iterative update mechanism ensures that the memory footprint grows only at the scale of concise text summaries (dozens of tokens) rather than raw webpage observation (typically thousands of tokens). This state-to-state transition effectively transforms long-horizon navigation task into a manageable Markovian-like decision process, where $\mathcal{M}_{t}^{int}$ serves as a sufficient statistic for historical trajectory.

Figure 4: Overview of Insight Retrieval Augmentation pipeline. The workflow begins with a Trajectory Insight Database populated by successful trajectories from diverse models. For a given user query , the system filters insights based on semantic similarity, retrieving the Top- $i$ relevant insights. These insights are then injected into the System Prompt. This mechanism ensures the Web Agent generates goal-centric and error-proof actions, effectively navigating complex UI environments by leveraging cross-trajectory experiences.

### 2.3 Insight Retrieval Augmentation

While internal memory ensures trajectory-level coherence, it cannot provide the agent with prior knowledge of common web navigation patterns or potential pitfalls. To bridge this gap, we design an External Memory module that leverages an Insight Bank $\mathcal{B}$ to provide strategic guidance. The complete pipeline is shown in Fig. 4.

#### 2.3.1 Offline Insight Extraction from Successful Trajectories

The efficacy of our external memory depends on the quality of the distilled insights. To transform raw interaction logs into transferable strategic knowledge, we implement a systematic reverse-engineering process. We utilize a heterogeneous collection of successful trajectories $\mathcal{T}_{succ}$ generated by various frontier models (e.g., Claude-3.7, Claude-4, and Ovis (Lu et al., 2024; Lu et al., 2025)) as the source data.

To perform high-fidelity knowledge distillation, we employ the Ovis-inhouse 80A3B model as a Web Navigation Strategy Abstractor. This model is guided by a specialized extraction prompt $P_{ext}$ designed to identify “High-Leverage Interaction Rules” from successful trajectories. It is worth noting that our framework is model-agnostic; Ovis is selected here primarily for its favorable balance between performance and deployment efficiency, serving as a representative instantiation of a capable vision-language model. The extraction process is formulated as:

$\mathcal{B}=\text{Abstractor}(\mathcal{T}_{succ},P_{ext})$ (6)

The prompt $P_{ext}$ enforces a focus on pure UI interaction logic, strictly prohibiting URL manipulation and trivial action reporting. It requires the abstractor to distill insights across three critical dimensions:

- •

Search & Filter Strategy (Input Logic): This dimension focuses on the construction of precision queries. It identifies optimal keyword prefixes, the use of exact match operators, and the strategic combination of UI filters to narrow down the search space immediately.

- •

Navigation Efficiency (Click Logic): The extractor identifies “Anchor Elements”—specific UI components (e.g., advanced search buttons or hierarchical breadcrumbs) that serve as shortcuts to avoid deep, irrelevant navigation paths and prevent the agent from wandering.

- •

State Validation (Wait/Check Logic): This provides agent with robustness heuristics, such as verifying UI state changes or implementing fallback strategies when a specific query yields no results, thereby avoiding infinite loops and traps.

Crucially, the extraction prompt requires the generalization of task-specific values into universal strategies. For example, specific item names are abstracted into “target items” to ensure the resulting insights $\mathcal{I}$ are applicable to a broad range of semantically similar tasks. The output is formatted as a set of topic-tagged, actionable insights, providing a structured knowledge base for the subsequent retrieval phase. We provide the complete $P_{ext}$ in Appendix F, as well as the distribution of these trajectories and an insight example in Appendix C.

#### 2.3.2 Construction of Insight Bank

The insights extracted are organized into a key-value store, referred to as the Insight Bank $\mathcal{B}$ . Each entry in the bank is represented as a pair $(Q^{hist},\mathcal{I})$ , where $Q^{hist}$ is the historical user query that initiated the successful trajectory, and $\mathcal{I}$ is the corresponding insights.

To enable semantic retrieval, we utilize a pre-trained Sentence Transformer model $\phi(\cdot)$ to encode the historical queries into a high-dimensional vector space:

$\mathbf{v}^{hist}=\phi(Q^{hist})$ (7)

The set of $N$ trajectories in bank can be characterized as $\{(\mathbf{v}_{j}^{hist},\mathcal{I})\}_{j=1}^{N}$ thus serves as a global experience repository that the agent can query during inference.

#### 2.3.3 Similarity-Based Retrieval and Contextual Injection

During execution of a new task with query $Q^{new}$ , agent first computes the query’s embedding $\mathbf{v}^{new}=\phi(Q^{new})$ . We then calculate the semantic similarity between current query and all entries in $\mathcal{B}$ using cosine similarity:

$\text{sim}(Q^{new},Q^{hist})=\frac{\mathbf{v}^{new}\cdot\mathbf{v}_{j}^{hist}}{\|\mathbf{v}^{new}\|\|\mathbf{v}_{j}^{hist}\|}$ (8)

The $i$ most similar queries are identified, and their corresponding insights are retrieved to form the external memory $\mathcal{M}^{ext}=\{\mathcal{I}_{1},\dots,\mathcal{I}_{i}\}$ .

Finally, these insights are injected into the system prompt $P_{sys}$ as “Defensive Hints.” The agent is instructed to treat these insights as strategic suggestions that must be verified against the current observation $O_{t}$ before execution. This injection transforms the agent from a purely reactive entity into a goal-oriented navigator that actively avoids past mistakes, thereby significantly increasing the success rate in complex, long-horizon tasks.

Table 2: Performance Comparison of Claude and Qwen3VL Model across Multiple WebSite on WebVoyager. (Abbreviations: AR: Allrecipes, AM: Amazon, AP: Apple, AX: ArXiv, BN: BBC News, CD: Cambridge Dict., CO: Coursera, ES: ESPN, GH: GitHub, GM: Google Map, HF: Huggingface, WA: Wolfram Alpha.)

Model Config AR AM AP AX BN CD CO ES GH GM HF WA Avg.step Avg.token Avg.acc.

Claude-3.7-Sonnet Normal 71.1 65.9 69.8 74.4 78.6 83.7 83.3 77.3 75.6 63.4 44.2 76.1 14.6 121.2k 72.0

In Mem 73.3 70.7 69.8 62.8 78.6 88.4 83.3 68.2 78.1 63.4 58.1 80.4 20.3 82.9k 72.9

Ex Mem 82.2 78.1 76.7 76.7 83.3 86.1 85.7 68.2 90.2 80.5 67.4 82.6 15.7 164.9k 79.8

In & Ex Mem 84.4 80.5 76.7 86.1 85.7 93.0 95.2 84.1 92.7 82.9 74.4 78.3 16.9 84.5k 84.5 ( $+$ 12.5%)

Claude-Sonnet-4 Normal 82.2 70.7 69.8 79.1 83.3 93.0 85.7 84.1 87.8 78.1 69.8 82.6 15.4 128.2k 80.5

In Mem 84.4 75.6 79.1 81.4 78.6 88.4 85.7 79.6 82.9 70.7 67.4 82.6 16.8 64.5k 79.7

Ex Mem 84.4 85.4 83.7 83.7 92.9 86.1 70.5 85.4 85.4 73.2 72.1 89.1 12.4 120.7k 82.7

In & Ex Mem 82.2 78.1 81.4 81.4 85.7 95.4 92.9 95.5 92.7 87.8 74.4 84.8 16.1 78.1k 86.0 ( $+$ 5.5%)

Qwen3-VL-32B Normal 55.6 51.2 60.5 46.5 73.8 83.7 71.4 47.7 56.1 56.1 30.2 60.9 24.1 215.2k 57.8

In Mem 62.2 75.6 60.5 44.2 73.8 76.7 76.2 70.5 80.5 58.5 37.2 63.0 22.5 73.2k 64.9

Ex Mem 71.1 63.4 67.4 65.1 83.3 88.4 81.0 63.6 70.7 73.2 46.5 67.4 19.2 189.1k 70.1

In & Ex Mem 75.6 70.7 72.1 72.1 90.5 88.4 81.0 77.3 80.5 68.3 48.8 63.0 21.8 92.3k 74.0 ( $+$ 16.2%)

### 2.4 Dual-Memory Integration and Inference

The final decision-making architecture of our agent is a synthesis of local trajectory awareness and global strategic knowledge. By integrating the internal and external memory modules, we reconstruct the operational context $C_{t}$ at each time step $t$ . This integration allows the agent to navigate complex web environments with both hindsight (from past steps) and foresight (from expert insights).

#### 2.4.1 Synthesized Decision Context

As established in the preceding sections, the vanilla context—which suffers from rapid growth in length—is replaced by our dual-memory augmented context $C^{\prime}_{t}$ . For any step $t\in\{1,\dots,n\}$ , the agent generates the current thought $T_{t}$ , action $A_{t}$ , and state summary $s_{t}$ based on the following synthesized formulation:

$(T_{t},A_{t},s_{t})=\text{Agent}(P^{\prime}_{sys},Q,\mathcal{M}_{t}^{int},\mathcal{M}^{ext},O_{t})$ (9)

where:

- •

$\mathcal{M}_{t}^{int}=\{s_{1},\dots,s_{t-1}\}$ provides local trajectory awareness, ensuring agent maintains a compact yet coherent record of its progress within current session.

- •

$\mathcal{M}^{ext}=\text{Top${}_{i}$}(\phi(Q),\mathcal{B})$ provides global heuristic guidance, injecting task-specific “defensive hints” retrieved from the offline Insight Bank to prevent the recurrence of historical failures.

- •

$P^{\prime}_{sys}$ represents the optimized $P_{sys}$ that fits M2 framework. Please refer to Appendix F for complete $P^{\prime}_{sys}$ .

#### 2.4.2 Synergy and Efficiency

The interplay between $\mathcal{M}_{t}^{int}$ and $\mathcal{M}^{ext}$ creates a powerful synergy for long-horizon web navigation. While $\mathcal{M}_{t}^{int}$ anchors agent to current task’s state transitions, $\mathcal{M}^{ext}$ acts as a regularizer, correcting agent’s trajectory when it encounters ambiguous page elements or complex interaction logic.

From an efficiency perspective, this dual-memory structure effectively decodes the long-horizon dependency problem into a sub-linearly growing context window. By discarding raw historical context $O_{t-k:t-1}$ & $S_{1:t-1}$ and retaining only the distilled summaries and Top- $i$ insights, our framework ensures that the token consumption remains stable even as the task depth increases. Consequently, this architecture achieves a superior trade-off between computational cost and task success rate, fulfilling the primary objective of our training-free augmentation approach.

Table 3: Evaluation Results on OnlineMind2Web. Green values in parentheses represent the relative reduction percentage for Tokens and absolute percentage point increase for Accuracy.

Model Config Easy Medium Hard Average

Step Token Acc. (%) Step Token Acc. (%) Step Token Acc (%) Step Token Acc. (%)

Claude-3.7-Sonnet Normal 21.72 219.6k 74.07 29.50 310.9k 58.27 34.51 371.6k 53.52 28.56 300.3k 61.95

In & Ex 19.89 98.0k ( $-$ 55.4%) 79.01 (+4.9%) 27.40 144.0k ( $-$ 53.7%) 65.47 (+7.2%) 31.03 163.9k ( $-$ 55.9%) 61.97 (+8.5%) 26.05 135.2k ( $-$ 55.0%) 68.82 (+6.9%)

Claude-Sonnet-4 Normal 16.23 154.7k 74.07 21.24 205.9k 66.91 21.24 292.1k 61.97 21.54 212.7k 67.70

In & Ex 18.77 95.7k ( $-$ 38.1%) 82.72 (+8.7%) 27.00 141.2k ( $-$ 31.4%) 73.38 (+6.5%) 37.24 207.2k ( $-$ 29.1%) 64.79 (+2.8%) 27.21 144.6k ( $-$ 32.0%) 73.88 (+6.2%)

Qwen3-VL-32B Normal 25.41 220.9k 44.44 35.38 325.8k 30.22 42.11 403.7k 21.13 34.25 315.6k 31.96

In & Ex 24.67 104.0k ( $-$ 52.9%) 65.43 (+21.0%) 29.67 127.1k ( $-$ 61.0%) 52.52 (+22.3%) 37.21 166.0k ( $-$ 58.9%) 33.80 (+12.7%) 30.12 130.2k ( $-$ 58.7%) 51.55 (+19.6%)

## 3 Experiment

In this section, we evaluate the performance of our dual-memory augmented framework across diverse real-time web navigation benchmarks. We describe the experimental configurations, evaluation protocols, and the infrastructure used to simulate real-world user interactions.

### 3.1 Basic Settings

Datasets and Models. Our framework is evaluated on two benchmarks: WebVoyager (12 representative websites) and OnlineMind2Web (291 accessible tasks). We employ a diverse set of multimodal backbones as agents, including Claude 3.7 & 4 computer-use version, and open-source Qwen3-VL-32B.

Infrastructure and Constraints. All experiments are executed on a high-concurrency Kubernetes cluster. Agents interact with virtualized browser environments via kubectl. Each task is restricted to a maximum of 60 interaction steps. While the baseline utilizes a raw observation window of size $k=5$ , our dual-memory approach optimizes this into a sub-linear summary chain.

Implementation and Evaluation. The external Insight Bank is constructed from 55k successful trajectories; while they share the same website distribution as WebVoyager, they involve a distinct set of user queries. For each query, we retrieve the Top-5 insights using sentence transformer and cosine similarity. Following the interaction, GPT-4o (Hurst et al., 2024) determines task success by analyzing the agent’s textual response and last five screenshots of the trajectory. More details on configurations are provided in Appendix B.

### 3.2 Main Results and Analysis

#### 3.2.1 Overall Performance and Generalization

The experimental results on WebVoyager and OnlineMind2Web, summarized in Table 2 and Table 3, demonstrate the consistent superiority of our Dual-Memory Augmentation framework across various model backbones and task complexities. We provide several examples of visualized agent interaction trajectories with corresponding analysis in Appendix E.

Overall Superiority. In all tested scenarios, the combined “In & Ex Mem” configuration achieves the highest success rates. On WebVoyager, Claude-3.7-Sonnet and Claude-Sonnet-4 reach success rates of $84.5\%$ and $86.0\%$ respectively, while the open-source Qwen3-VL-32B sees a significant jump from $57.8\%$ to $74.0\%$ . Notably, our framework effectively bridges the gap between open-source and closed-source models; with dual-memory augmentation, the local Qwen3-VL-32B model ( $74.0\%$ ) even outperforms the vanilla Claude-3.7-Sonnet agent ( $72.0\%$ ), highlighting that robust memory management can compensate for limitations in web navigation.

Robustness Across Task Complexity. Analysis of the OnlineMind2Web dataset (Table 3) reveals that the performance gains are most pronounced in more challenging scenarios. In “Medium” tasks, the success rate of Qwen3-VL-32B increases from $30.22\%$ to $52.52\%$ , a relative improvement of nearly $73\%$ . This suggests that our memory mechanism provides crucial stability when the agents are faced with the deep navigation hierarchies and the complex UI interaction chains.

Cross-Trajectory Knowledge Transfer. A key finding is the efficacy of our Insight Bank. Although the trajectories align with the website distribution of WebVoyager, the extracted insights demonstrate a universal value across different benchmarks and models. Even when applied to the dynamic OnlineMind2Web environment, these insights provide strategic guidance that translates into immediate success rate improvements for Claude and Qwen3VL models alike, proving that high-level website UI interaction rules are transferable across different web domains.

#### 3.2.2 Efficiency and Practicality Analysis

Beyond raw success rates, we investigate the underlying mechanisms of efficiency and decision-making quality provided by our internal and external memory modules.

Efficiency Gains via Internal Memory. As shown in the “Avg.token” columns, the introduction of internal memory ( $\mathcal{M}^{int}$ ) leads to a dramatic reduction in computational cost. By replacing redundant raw observation windows with concise trajectory summaries, the average token consumption per task for Qwen3-VL-32B on WebVoyager drops from $215.2k$ to $92.3k$ , a saving of approximately $57\%$ . This confirms that our training-free framework effectively mitigates the context-window pressure inherent in long-horizon tasks without sacrificing historical context.

Practicality via External Memory. The ablation of memory modules (Table 2) illustrates that external memory ( $\mathcal{M}^{ext}$ ) is the primary driver of decision practicality. For instance, in the Google Maps domain, the accuracy of Claude-3.7-Sonnet increases from $63.4\%$ to $80.5\%$ with the addition of insights. These “defensive hints” prevent the agent from repeating common pitfalls and ensure that navigation remains goal-centric even in visually noisy environments. Furthermore, performance profiling indicates that against an average task duration of 122 seconds on WebVoyager, the latency for a single similarity-based insight retrieval is approximately 6 milliseconds, demonstrating that the computational overhead introduced by $\mathcal{M}^{ext}$ is negligible and highly sustainable for real-time deployment.

Trade-off between Steps and Accuracy. We observe an interesting correlation between average steps and accuracy. In certain configurations, the average steps slightly increase alongside accuracy (e.g., Claude-3.7-Sonnet on WebVoyager). This is not an indication of inefficiency, but rather a reflection of the agent’s enhanced persistence. M2 provides the agent with a clearer sense of progress and strategic guidance, preventing the model from prematurely judging the task as uncompletable and exiting. By maintaining a more stable and compact context, the agent is empowered to explore longer, correct its own course, and eventually reach the target state. We visualize the token consumption of the baseline and M2 on a specific task in Appendix D.

Table 4: Ablation study of M2 components on WebVoyager by Qwen3-VL-32B. S.T. denotes Sentence Transformer (Default).

Ablation Category Settings Acc. (%)

Historical Window $k$ 1 51.6

3 56.0

5 (Default) 57.8

7 52.2

Similarity Function TF-IDF (Salton and Buckley, 1988) 61.1

Word2Vec (Mikolov et al., 2013) 64.4

BM25 (Robertson et al., 1995) 58.0

S.T. (Reimers and Gurevych, 2019) 70.1

Insight Count $i$ 3 64.8

5 (Default) 70.1

7 66.2

In-Mem Components w/o Visual 62.3

w/o Action 54.7

Full 64.9

### 3.3 Ablation Studies

We perform ablation experiments on the specific settings of the M2 framework to systematically evaluate the influence of various parameters on agent performance. (1) Regarding the historical observation window size, performance scales with context depth up to five steps, demonstrating that while moderate temporal cues are essential for resolving state ambiguity, excessive historical length introduces contextual interference that can degrade agent reasoning. (2) In the evaluation of similarity functions, the sentence transformer outperforms traditional lexical methods such as TF-IDF and BM25, indicating that capturing the underlying semantic intent of a query is more effective than simple keyword overlap. (3) For insight retrieval count, five insights provide optimal density of guidance; increasing the number to seven leads to a performance decline, likely due to the introduction of redundant information that distracts the agent. (4) Finally, the analysis of internal memory components demonstrates that while both visual feedback and action history are critical, the removal of action records causes a more significant drop in accuracy, identifying it as the primary element for maintaining trajectory continuity and ensuring robust decision-making.

## 4 Conclusion

In this paper, we introduce the Dual-Memory framework M2, a training-free approach designed to enhance the execution robustness and computational efficiency of autonomous web agents. Our recursive summarization mechanism distills interaction history into concise state updates, while the retrieval of offline expert insights provides the strategic guidance necessary to navigate complex user interfaces without the overhead of data production or training. Extensive evaluations on WebVoyager and OnlineMind2Web demonstrate that our framework significantly improves navigation success across diverse model backbones. Most notably, the open-source Qwen3-VL-32B achieved a 16.2% increase in success rate alongside a 57% reduction in token consumption, effectively bridging the performance gap with proprietary models. These results establish that the dual-memory architecture provides a scalable and sustainable path for deploying high-fidelity web agents in real-world environments.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, specifically in the domain of autonomous web navigation agents. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

- anthropic (2025a) anthropic Claude 3.7 sonnet system card. External Links: Link Cited by: §1.

- anthropic (2025b) anthropic System card: claude opus 4 & claude sonnet 4. External Links: Link Cited by: §1.

- Bai et al. (2025) S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu Qwen3-vl technical report. External Links: 2511.21631, Link Cited by: §1.

- Chen et al. (2024) Q. Chen, D. Pitawela, C. Zhao, G. Zhou, H. Chen, and Q. Wu Webvln: vision-and-language navigation on websites. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38, pp. 1165–1173. Cited by: §1.

- Cheng et al. (2025) W. Cheng, E. Ni, W. Wang, Y. Sun, J. Liu, W. Shen, Y. Chen, B. Shi, and D. Wang MGA: memory-driven gui agent for observation-centric interaction. arXiv preprint arXiv:2510.24168. Cited by: §A.1, §1.

- Fan et al. (2024) Y. Fan, X. Ma, R. Wu, Y. Du, J. Li, Z. Gao, and Q. Li Videoagent: a memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision, pp. 75–92. Cited by: §A.1.

- Ge et al. (2025) Y. Ge, S. Romeo, J. Cai, R. Shu, Y. Benajiba, M. Sunkara, and Y. Zhang Tremu: towards neuro-symbolic temporal reasoning for llm-agents with memory in multi-session dialogues. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 18974–18988. Cited by: §A.1.

- He et al. (2024a) B. He, H. Li, Y. K. Jang, M. Jia, X. Cao, A. Shah, A. Shrivastava, and S. Lim Ma-lmm: memory-augmented large multimodal model for long-term video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13504–13514. Cited by: §1.

- He et al. (2024b) H. He, W. Yao, K. Ma, W. Yu, Y. Dai, H. Zhang, Z. Lan, and D. Yu Webvoyager: building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919. Cited by: §1.

- Huang et al. (2025) J. Huang, Z. Zeng, W. Han, Y. Zhong, L. Zheng, S. Fu, J. Chen, and L. Ma Scaletrack: scaling and back-tracking automated gui agents. arXiv preprint arXiv:2505.00416. Cited by: §1.

- Hurst et al. (2024) A. Hurst, A. Lerer, A. P. Goucher, A. Perelman, A. Ramesh, A. Clark, A. Ostrow, A. Welihinda, A. Hayes, A. Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276. Cited by: §3.1.

- Jain et al. (2025) J. Jain, S. Maheshwari, N. Yu, W. Hwu, and H. Shi AUGUSTUS: an llm-driven multimodal agent system with contextualized user memory. arXiv preprint arXiv:2510.15261. Cited by: §1.

- Kang et al. (2025) J. Kang, M. Ji, Z. Zhao, and T. Bai Memory os of ai agent. arXiv preprint arXiv:2506.06326. Cited by: §1.

- Li et al. (2025) K. Li, Z. Zhang, H. Yin, L. Zhang, L. Ou, J. Wu, W. Yin, B. Li, Z. Tao, X. Wang, et al. WebSailor: navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592. Cited by: §1.

- Li et al. (2024) Z. Li, Y. Xie, R. Shao, G. Chen, D. Jiang, and L. Nie Optimus-1: hybrid multimodal memory empowered agents excel in long-horizon tasks. Advances in neural information processing systems 37, pp. 49881–49913. Cited by: §A.2.

- Liu et al. (2025a) G. Liu, S. Geng, S. Li, H. Cui, S. Zhang, X. Liu, and T. Liu WebCoach: self-evolving web agents with cross-session memory guidance. arXiv preprint arXiv:2511.12997. Cited by: §1.

- Liu et al. (2024) N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang Lost in the middle: how language models use long contexts. Transactions of the Association for Computational Linguistics 12, pp. 157–173. Cited by: §1.

- Liu et al. (2025b) Z. Liu, J. Li, W. X. Zhao, D. Gao, Y. Li, and J. Wen PAL-ui: planning with active look-back for vision-based gui agents. arXiv preprint arXiv:2510.00413. Cited by: §1.

- Long et al. (2025) L. Long, Y. He, W. Ye, Y. Pan, Y. Lin, H. Li, J. Zhao, and W. Li Seeing, listening, remembering, and reasoning: a multimodal agent with long-term memory. arXiv preprint arXiv:2508.09736. Cited by: §A.1.

- Lu et al. (2024) S. Lu, Y. Li, Q. Chen, Z. Xu, W. Luo, K. Zhang, and H. Ye Ovis: structural embedding alignment for multimodal large language model. External Links: 2405.20797, Link Cited by: §2.3.1.

- Lu et al. (2025) S. Lu, Y. Li, Y. Xia, Y. Hu, S. Zhao, Y. Ma, Z. Wei, Y. Li, L. Duan, J. Zhao, Y. Han, H. Li, W. Chen, J. Tang, C. Hou, Z. Du, T. Zhou, W. Zhang, H. Ding, J. Li, W. Li, G. Hu, Y. Gu, S. Yang, J. Wang, H. Sun, Y. Wang, H. Sun, J. Huang, Y. He, S. Shi, W. Zhang, G. Zheng, J. Jiang, S. Gao, Y. Wu, S. Chen, Y. Chen, Q. Chen, Z. Xu, W. Luo, and K. Zhang Ovis2.5 technical report. External Links: 2508.11737, Link Cited by: §2.3.1.

- Mikolov et al. (2013) T. Mikolov, K. Chen, G. Corrado, and J. Dean Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781. Cited by: Table 4.

- Packer et al. (2023) C. Packer, V. Fang, S. Patil, K. Lin, S. Wooders, and J. Gonzalez MemGPT: towards llms as operating systems.. Cited by: §A.1.

- Pink et al. (2025) M. Pink, Q. Wu, V. A. Vo, J. Turek, J. Mu, A. Huth, and M. Toneva Position: episodic memory is the missing piece for long-term llm agents. arXiv preprint arXiv:2502.06975. Cited by: §A.1.

- Qi et al. (2024) Z. Qi, X. Liu, I. L. Iong, H. Lai, X. Sun, W. Zhao, Y. Yang, X. Yang, J. Sun, S. Yao, et al. Webrl: training llm web agents via self-evolving online curriculum reinforcement learning. arXiv preprint arXiv:2411.02337. Cited by: §1.

- Qin et al. (2025) Y. Qin, Y. Ye, J. Fang, H. Wang, S. Liang, S. Tian, J. Zhang, J. Li, Y. Li, S. Huang, et al. Ui-tars: pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326. Cited by: §1.

- Rasmussen et al. (2025) P. Rasmussen, P. Paliychuk, T. Beauvais, J. Ryan, and D. Chalef Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956. Cited by: §A.2.

- Reimers and Gurevych (2019) N. Reimers and I. Gurevych Sentence-bert: sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, External Links: Link Cited by: Table 4.

- Robertson et al. (1995) S. E. Robertson, S. Walker, S. Jones, M. M. Hancock-Beaulieu, M. Gatford, et al. Okapi at trec-3. British Library Research and Development Department. Cited by: Table 4.

- Salama et al. (2025) R. Salama, J. Cai, M. Yuan, A. Currey, M. Sunkara, Y. Zhang, and Y. Benajiba Meminsight: autonomous memory augmentation for llm agents. arXiv preprint arXiv:2503.21760. Cited by: §A.2.

- Salton and Buckley (1988) G. Salton and C. Buckley Term-weighting approaches in automatic text retrieval. Information processing & management 24 (5), pp. 513–523. Cited by: Table 4.

- Sarch et al. (2024) G. Sarch, L. Jang, M. Tarr, W. W. Cohen, K. Marino, and K. Fragkiadaki Vlm agents generate their own memories: distilling experience into embodied programs of thought. Advances in Neural Information Processing Systems 37, pp. 75942–75985. Cited by: §A.2.

- Tan et al. (2025a) H. Tan, Z. Zhang, C. Ma, X. Chen, Q. Dai, and Z. Dong MemBench: towards more comprehensive evaluation on the memory of llm-based agents. arXiv preprint arXiv:2506.21605. Cited by: §A.1.

- Tan et al. (2025b) Z. Tan, J. Yan, I. Hsu, R. Han, Z. Wang, L. Le, Y. Song, Y. Chen, H. Palangi, G. Lee, et al. In prospect and retrospect: reflective memory management for long-term personalized dialogue agents. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8416–8439. Cited by: §A.2.

- Wang and Chen (2025) Y. Wang and X. Chen Mirix: multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957. Cited by: §A.1, §1.

- Wang et al. (2025a) Y. Wang, R. Takanobu, Z. Liang, Y. Mao, Y. Hu, J. McAuley, and X. Wu Mem- $\{$ $\backslash$ alpha $\}$ : learning memory construction via reinforcement learning. arXiv preprint arXiv:2509.25911. Cited by: §1.

- Wang et al. (2025b) Z. Wang, L. Yang, X. Tang, S. Zhou, D. Chen, W. Jiang, and Y. Li History-aware reasoning for gui agents. arXiv preprint arXiv:2511.09127. Cited by: §A.2, §1.

- Wheeler and Jeunen (2025) S. Wheeler and O. Jeunen Procedural memory is not all you need: bridging cognitive gaps in llm-based agents. In Adjunct Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and Personalization, pp. 360–364. Cited by: §A.2.

- Xu et al. (2025) W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang A-mem: agentic memory for llm agents. arXiv preprint arXiv:2502.12110. Cited by: §A.1.

- Xu et al. (2024) Y. Xu, D. Lu, Z. Shen, J. Wang, Z. Wang, Y. Mao, C. Xiong, and T. Yu Agenttrek: agent trajectory synthesis via guiding replay with web tutorials. arXiv preprint arXiv:2412.09605. Cited by: §1.

- Xue et al. (2025) T. Xue, W. Qi, T. Shi, C. H. Song, B. Gou, D. Song, H. Sun, and Y. Su An illusion of progress? assessing the current state of web agents. arXiv preprint arXiv:2504.01382. Cited by: §1.

- Yu et al. (2025) H. Yu, T. Chen, J. Feng, J. Chen, W. Dai, Q. Yu, Y. Zhang, W. Ma, J. Liu, M. Wang, et al. MemAgent: reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259. Cited by: §1.

- Zhang et al. (2025a) D. Zhang, B. Rama, J. Ni, S. He, F. Zhao, K. Chen, A. Chen, and J. Cao Litewebagent: the open-source suite for vlm-based web-agent applications. arXiv preprint arXiv:2503.02950. Cited by: §1.

- Zhang et al. (2025b) G. Zhang, M. Fu, G. Wan, M. Yu, K. Wang, and S. Yan G-memory: tracing hierarchical memory for multi-agent systems. arXiv preprint arXiv:2506.07398. Cited by: §A.2, §1.

- Zhang et al. (2025c) Z. Zhang, Q. Dai, X. Chen, R. Li, Z. Li, and Z. Dong MemEngine: a unified and modular library for developing advanced memory of llm-based agents. In Companion Proceedings of the ACM on Web Conference 2025, pp. 821–824. Cited by: §A.1.

- Zhou et al. (2025) Z. Zhou, A. Qu, Z. Wu, S. Kim, A. Prakash, D. Rus, J. Zhao, B. K. H. Low, and P. P. Liang MEM1: learning to synergize memory and reasoning for efficient long-horizon agents. arXiv preprint arXiv:2506.15841. Cited by: §1.

## Appendix A Related Work

### A.1 Context Management and Memory Optimization in LLM Agents

Effective context management is a fundamental prerequisite for deploying LLM based agents in long-horizon web navigation tasks. Recent literature has extensively explored strategies to optimize memory representation and context utilization (Ge et al., 2025; Tan et al., 2025a; Pink et al., 2025; Xu et al., 2025; Zhang et al., 2025c). To manage extensive contexts, MemGPT (Packer et al., 2023) utilizes a tiered storage mechanism to handle unbounded data, while MIRIX (Wang and Chen, 2025) orchestrates six distinct memory types for persistent reasoning. In multimodal domains, VideoAgent (Fan et al., 2024) and M3-Agent (Long et al., 2025) employ structured, entity-centric frameworks to track visual and auditory states dynamically. Similarly, MGA (Cheng et al., 2025) leverages structured memory to model GUI interactions as independent states, effectively decoupling long-chain dependencies.

Current context strategies typically suffer from two drawbacks: the reliance on decoupled summarization agents which increases inference cost, and the dependence on learning-based models that require extensive training. We overcome these by proposing a lightweight, training-free alternative. Our method employs in-context prompt abstraction to filter visual noise and retain navigation cues, eliminating the need for heavy external modules or parameter updates.

### A.2 Experience Learning and Retrieval-Augmented Generation for LLM Agents

By leveraging paradigms like Experience Learning or Retrieval-Augmented Generation (RAG) to retrieve relevant insights, agents can transcend immediate context limitations to avoid recurring pitfalls and enhance generalization (Salama et al., 2025; Wheeler and Jeunen, 2025; Rasmussen et al., 2025). Focusing on abstraction and reflection, ICAL (Sarch et al., 2024) transforms suboptimal trajectories into cognitive abstractions via self-correction, while Optimus-1 (Li et al., 2024) utilizes an experience pool of summarized interactions to enhance planning. Incorporating reinforcement learning, RMM (Tan et al., 2025b) iteratively refines memory retrieval using online RL, and HAR (Wang et al., 2025b) employs hybrid rewards to synthesize error correction guidelines. Furthermore, G-Memory (Zhang et al., 2025b) integrates high-level insights with fine-grained trajectories through a bi-directional retrieval mechanism.

Existing frameworks primarily target textual tasks, often lacking the procedural specificity required for web navigation. Moreover, they typically rely on factual retrieval or online interactions. Addressing these limitations, we propose a semantic Insight Retrieval Augmentation module. This mechanism utilizes an offline bank of successful trajectories to retrieve actionable insights via task similarity, offering robust guidance in a training-free manner.

## Appendix B Implementation Details

Benchmarks and Datasets. We evaluate our framework on two primary benchmarks: WebVoyager and OnlineMind2Web. Due to regional access and verification restrictions, we select a representative subset of 12 websites from WebVoyager for evaluation. For OnlineMind2Web, which involves real-time web interaction, we successfully conduct tasks across 291 websites, discarding 9 tasks where the target URLs were inaccessible at the time of testing. These datasets provide a rigorous testbed for long-horizon reasoning and cross-domain UI navigation.

Agent Backbones. To demonstrate the model-agnostic nature of our dual-memory approach, we employ three multimodal models as agents: Claude-3.7-Sonnet, Claude-Sonnet-4, and the open-source Qwen3-VL-32B. All models are tasked with operating web pages and executing user instructions within a training-free prompting framework.

Infrastructure and Environment. Experiments are conducted on a high-concurrency platform built on Kubernetes (K8s). We deploy a cluster of 64 concurrent nodes, each hosting a virtual computer environment. The agents interact with these environments via kubectl to perform browser actions and capture visual feedback. To balance task complexity and computational cost, we limit each task to a maximum of 60 steps.

Dual-Memory Configuration. For the internal memory, while the baseline maintains a sliding window of the last $k=5$ screenshots, our framework replaces this with a summary chain $\mathcal{M}^{int}$ , retaining only the current observation $O_{t}$ . The external memory is powered by an Insight Bank distilled from 55k successful trajectories. We utilize the all-MiniLM-L6-v2 Sentence Transformer to calculate cosine similarity between the current query $Q$ and historical trajectories, retrieving the Top-5 most relevant insights as strategic hints.

Evaluation Protocol. Following the interaction, task success is determined by a GPT-4o judge. The judge evaluates the model’s final response and the last five screenshots of the trajectory to verify goal completion. This visual-textual evidence ensures a robust and objective assessment of the agent’s performance in real-world navigation scenarios.

Figure 5: Distribution of the 55k trajectory data across 12 distinct web domains in the Insight Bank.

Figure 6: Cumulative token consumption comparison on an Apple.com navigation task (16 steps). The agent uses Qwen3-VL-32B.

## Appendix C Insight Bank Statistics & Details

To ensure the generalization capability and robustness of our external memory, we construct a large-scale Insight Bank derived from approximately 55k successful execution trajectories. These trajectories span 12 diverse web domains, covering categories such as e-commerce, academic research, coding platforms, and utility tools. The user queries for all these trajectories are generated by LLM and are strictly distinct from the benchmarks used in the experiments.

As illustrated in Fig. 5, the data distribution is strategically balanced to prevent overfitting to specific website layouts. This diversity ensures that the extracted insights cover a wide spectrum of interaction logic, from static information retrieval to dynamic form manipulation.

Beyond the quantitative scale, qualitative analysis of stored insights is critical for effective retrieval augmentation. Unlike raw trajectory logs which record specific interactions, our Insight Bank stores generalized, high-leverage interaction rules. To illustrate this, consider a complex task on Apple.com requiring the configuration of a 16-inch MacBook Pro with specific specifications (M4 Max chip, 32GB memory, etc.). Instead of memorizing the specific click sequence for this singular product, the extracted insights provide universal UI strategies applicable to similar shopping scenarios:

- •

Search Strategy: When the exact query yields no results, strip the query to the core noun and use the sidebar model filter to drill down manually.

- •

Interaction Order: Always apply the “Sort by Date/Price” before selecting specific filters. On many dynamic sites, changing the sort order triggers a page refresh that inadvertently resets active filters, causing the agent to lose progress.

- •

State Validation: After clicking “Add to Cart/Bag,” do not proceed immediately; verify the “Cart Icon” badge number has incremented. If it hasn’t changed within 3 seconds, the click was likely intercepted by an overlay—trigger the click again.

- •

Navigation Efficiency: If a page fails to load or display properly, use the browser’s “Back” key (Alt+Left) to return to the previous page and resume the configuration process, rather than refreshing the page repeatedly.

By retrieving and injecting such structured, logic-driven insights into the prompt, our method enables the agent to reference explicit operational goals and avoid recurring navigation traps during inference.

## Appendix D Token Consumption Growth Curve

To rigorously quantify the efficiency gains of our proposed framework, we conduct a step-by-step token consumption analysis on a representative task by Qwen3-VL-32B. The task required the agent to navigate Apple.com to retrieve specific technical specifications and software compatibility information: “Find information on Apple website, and tell me the device weight of Apple Vision Pro and list 5 Built-in Apps it supports.”

Fig. 6 illustrates the cumulative token usage for both the Original Interaction mode (full-history concatenation) and our M2 Interaction mode (dual-memory augmentation) over a 16-step trajectory.

The comparison reveals three distinct phases of token consumption dynamics:

##### 1. Initial Overhead (Step 1):

As observed in the initial phase, the M2 framework incurs a higher starting cost compared to the baseline. Specifically, at Step 1, M2 consumes approximately $\sim$ 3.6k tokens, exceeding the Original method by $\sim$ 1.7k tokens. This overhead is intentional and attributable to the Insight Retrieval Augmentation module, which injects retrieved guidelines and defensive instructions into the system prompt before the first action is taken. This upfront investment is designed to “prime” the agent with expert knowledge.

##### 2. The Crossover Point (Steps 3–5):

The efficiency advantage of the baseline is short-lived. The curves intersect between Step 3 and Step 4. By Step 4, the cumulative cost of the Original method accelerates rapidly as it begins to stack raw history (screenshots and entire text) from previous steps. In contrast, M2 employs Dynamic Trajectory Summarization, discarding raw observations in favor of concise textual state updates. Consequently, the consumption of the original method surpasses M2 almost immediately after the several interaction loop begins.

##### 3. Long-Term Growth Trends (Step 5–16):

The divergence becomes drastic as the trajectory lengthens.

- •

Original Interaction (Rapid Growth): The baseline exhibits a super-linear growth trend. By Step 16, the context window is burdened with full interaction history, driving the total consumption to 106k tokens. This exponential bloat not only increases cost but also introduces significant visual noise.

- •

M2 Interaction (Gradual Growth): In contrast, our method maintains a stable growth rate. By condensing history into summaries, the marginal cost per step remains constant ( $\sim$ 3.7k tokens). By Step 16, the total consumption is only 58k tokens, representing a 45.3% reduction compared to the baseline.

This analysis confirms that while M2 requires a minimal cold-start “token investment” to prime the agent with expert knowledge, it effectively solves the context explosion problem, making it a highly scalable solution for long-horizon web navigation.

Figure 7: Comparison of agent responses between the original interaction form and the M2 interaction form.

Figure 8: Qualitative comparison on the GitHub task. Left (Original): Agent gets trapped in a loop of irrelevant repository results due to context redundancy. Right (M2): Agent navigates hierarchically to the documentation page and uses a precise search, maintaining a clear task progress via memory summarization.

Figure 9: Qualitative comparison on a geographic disambiguation task. Left (Original): The agent suffers from context hallucination, ignoring the established UK starting point and selecting a US destination (North Plymouth, MA), resulting in a routing failure. Right (M2): Guided by the external insight to prioritize the first dropdown suggestion and maintaining internal context consistency, the agent correctly identifies the local UK destination.

## Appendix E Qualitative Analysis

### E.1 Comparison of Response Structure Visualization

Fig. 7 illustrates the structural differences in agent responses between the baseline and our proposed method.

Original Interaction Condition (Left): The baseline agent follows a standard ReAct-style output format. The response consists solely of a Thought block, where the agent reasons based on the immediate visual observation, followed directly by the Action parameters. The context relies implicitly on the raw history of previous turns.

Dual-Memory Augment Condition (Right): In contrast, the M2 framework introduces two distinct structural enhancements to the response format:

- •

Insight-Aware Reasoning: Within the Thought block, the agent explicitly references the external memory (e.g., “According to the Reference Trajectory Insights…”), demonstrating that high-level strategic guidance is actively incorporated into the planning process before action selection.

- •

Structured Summarization: A unique Summary field is generated alongside the action. This field adheres to a strict template ([Brief current page state] $\to$ [Brief action taken]), which serves as the compressed internal memory update for the subsequent step, replacing the need to store raw observation logs.

### E.2 Case Study: GitHub Copilot FAQ Retrieval

We analyze a retrieval task: “Open GitHub Copilot’s FAQs to find the official answer to when Copilot chat can be used on mobile.” (Fig. 8).

Baseline Failure (Context Overload): The baseline agent falls into a “global search trap,” repeatedly querying the homepage. Specifically, the agent consistently triggers the “Search all of GitHub” action and receives a static “No results found” page; interpreting this visual invariance as a system non-response, the agent mistakenly retries the click indefinitely, obscuring state awareness in the context window.

M2 Success (Insight & Summary-Driven Navigation): In contrast, M2 bypasses the noisy global search in favor of hierarchical navigation (Copilot $\to$ Resources $\to$ Documentation). This is enabled by M2, which maintains a clean execution log and professional guidance. By filtering out visual noise, the concise context allows the agent to track its progress accurately and eventually execute a precise local search within the documentation to retrieve the answer.

### E.3 Case Study: Disambiguating Geographic Locations

We examine a navigation task susceptible to geographic ambiguity: “Find a parking lot in Gloucester and book a ride from there to North Plymouth, view the map to understand the route better.” (Fig. 9).

Baseline Failure (Context Hallucination): The baseline agent initially identifies the starting point in Gloucester, UK. However, when searching for the destination “North Plymouth,” it suffers from a contextual hallucination. Ignoring the established UK context, it selects the second suggestion, “North Plymouth, MA, USA.” This results in an infeasible trans-Atlantic routing error.

M2 Success (Insight & Context Synergy): The M2 agent successfully resolves this ambiguity through two synergistic mechanisms:

- •

Insight-Driven Selection Strategy (External Memory): Guided by the specific insight “Preferably select the first suggestion from the dropdown list to ensure accuracy,” the agent prioritizes the top-ranked suggestion provided by the search engine (which naturally prioritizes proximity to the current map view/start point).

- •

Contextual Consistency (Internal Memory): The agent’s reasoning explicitly validates the choice: “matches the intended destination in the UK (as Gloucester is in the UK).” By adhering to the specific context rather than relying on intrinsic bias, it correctly selects “North Plymouth, UK,” enabling a valid route calculation.

Figure 10: Original system prompt used by the agent.

Figure 11: Update of Dynamic Trajectory Summarization compared to the original system prompt.

Figure 12: Prompt $P_{ext}$ used by Abstractor to extract insight.

Figure 13: System prompts $P^{\prime}_{sys}$ used in the final M2 framework.

## Appendix F Prompt Templates

### F.1 Original System Prompt $P_{sys}$ and Action Space

As shown in Fig. 10, the system prompt $P_{sys}$ establishes model’s role as an autonomous web browsing agent tasked with resolving user queries through sequential interaction with web interfaces. At each step, the agent receives a multimodal observation consisting of a webpage screenshot and semantic text, grounded by the specific viewport resolution. The action space is designed to simulate human interactions, encompassing mouse operations (e.g., HOVER, LEFT_CLICK, DRAG), keyboard inputs (TYPE, PRESS_KEY), and navigational controls (SCROLL, WAIT, and page transitions). To ensure execution, the model is constrained to generate a strictly formatted four-part output: a Thought summarizing the reasoning process, an Action Element specifying the target coordinates (e.g., <point>(x,y)</point>), an Action Type selected from the predefined set, and an Action Value containing necessary parameters such as input text or scroll direction.

### F.2 $P^{in}_{sys}$ and $s_{t}$ for Dynamic Trajectory Summarization

To accommodate the integration of the dynamic memory mechanism, we refine the standard system instructions into a history-aware prompt, denoted as $P^{in}_{sys}$ (Fig. 11). Unlike the baseline which relies solely on immediate observations, $P^{in}_{sys}$ explicitly conditions the agent’s reasoning on a “Historical Summary”—a compressed chronological log of prior interactions ( $\mathcal{M}_{t}^{int}$ ). This summary enforces a structured format where each entry $s_{t}$ encapsulates the pre-action state and the operation performed, exemplified by: $s_{t}=$ [On Apple Watch Series 11 landing page with "Buy" button visible] $\to$ [Click the "Buy" button to access pricing details]. Furthermore, $P^{in}_{sys}$ incorporates strict negative constraints (Rule 6) that compel the agent to cross-reference the current observation against the historical trace to actively prune redundant explorations and prevent execution loops. Finally, the output space is augmented to require the generation of the current step’s summary alongside the action, ensuring the memory stream is autonomously maintained.

### F.3 $P_{ext}$ for Insight Retrieval Augmentation

The insight extraction prompt $P_{ext}$ (Fig. 12) configures the Model to act as a “Web Navigation Strategy Abstractor,” tasked with distilling generalizable “High-Leverage Interaction Rules” from historical successful trajectories. Unlike standard summarization, this prompt enforces a Pure UI Interaction constraint, strictly prohibiting URL manipulation to ensure the extracted insights focus on intelligent interface usage—such as identifying “Anchor Elements” for navigation shortcuts or constructing precise search queries. The analysis is stratified into three specific dimensions: Search & Filter Strategy (optimizing input logic), Navigation Efficiency (optimizing click paths), and State Validation (visual logic for error avoidance). To ensure transferability, the prompt mandates the generalization of specific values and the exclusion of trivial actions. The output is constrained to a strict, machine-parsable text format, consisting of newline-separated insights prefixed with topic tags (e.g., [Search Strategy]), enabling seamless storage and retrieval within the Insight Bank.

### F.4 $P^{\prime}_{sys}$ applicable to M2

Compared to the initial baseline prompt, the final version $P^{\prime}_{sys}$ (Fig. 13) incorporates substantial modifications to support the dual-memory architecture and ensure robust execution. First, the observation space (Rule 1) is augmented to include both “Reference Trajectory Insights” (External Memory) and a “Historical Summary” (Internal Memory), providing the agent with global strategic guidance and local state context. Second, a defensive verification mechanism is explicitly codified (Rule 2), instructing the agent to critically evaluate retrieved insights against the current visual screenshot and prioritize actual observations over historical suggestions to prevent context mismatch. Third, the reasoning process is constrained by history-aware negative constraints (Rule 7), effectively barring the agent from repeating ineffective actions or re-exploring visited states identified in the summary. Finally, the output space (Rule 8) is expanded to require the generation of a structured Summary for the current step, enabling the iterative maintenance of the internal memory stream without external intervention.

Experimental support, please view the build logs for errors. Generated by L A T E xml .

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

- Click the "Report Issue" () button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

We gratefully acknowledge support from our major funders, member institutions, , and all contributors. Major funding support from
