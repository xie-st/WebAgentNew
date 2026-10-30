# Flash-Searcher: Fast and Effective Web Agents via DAG-Based Parallel Execution

- arXiv: 2509.25301
- URL: https://arxiv.org/html/2509.25301
- 来源：arXiv HTML（LaTeXML）自动转换，公式以 $...$ 内联保留，图表编号可能与原文有出入。

---

Flash-Searcher: Fast and Effective Web Agents via DAG-Based Parallel Execution

##### Report GitHub Issue

Title:

Content selection saved. Describe the issue below:

Description: arXiv is now an independent nonprofit! Learn more Back to arXiv License: CC BY-NC-SA 4.0 arXiv:2509.25301v1 [cs.AI] 29 Sep 2025 \correspondence

Wangchunshu Zhou at ; Xitong Gao at \checkdata[Code]https://github.com/OPPO-PersonalAI/Flash-Searcher

# Flash-Searcher: Fast and Effective Web Agents via DAG-Based Parallel Execution

August 24, 2026

###### Abstract

Large language models (LLMs) have demonstrated remarkable capabilities in complex agent reasoning tasks when equipped with external tools. However, current frameworks predominantly rely on sequential processing, leading to inefficient execution particularly for tasks requiring extensive tool interaction. This paper introduces Flash-Searcher, a novel parallel agent reasoning framework that fundamentally reimagines the execution paradigm from sequential chains to directed acyclic graphs (DAGs). Flash-Searcher decomposes complex tasks into subtasks with explicit dependencies, enabling concurrent execution of independent reasoning paths while maintaining logical constraints. Through dynamic workflow optimization, our framework continuously refines the execution graph based on intermediate results, effectively integrating summary module. Comprehensive evaluations across multiple benchmarks demonstrate that Flash-Searcher consistently outperforms existing approaches. Specifically, it achieves 67.7% accuracy on BrowseComp and 83% on xbench-DeepSearch, while reducing agent execution steps by up to 35% compared to current frameworks. Furthermore, when distilling this parallel reasoning pipeline into single models, we observe substantial performance gains across diverse backbone architectures, underscoring the generalizability of our methodology. Our work thus represents a significant advance in agent architecture design, offering a more scalable and efficient paradigm for complex reasoning tasks.

## 1 Introduction

Recent advances in tool-augmented agents and multi-agent systems (MAS) (Dorri et al., 2018; Canese et al., 2021; Zhou et al., 2023c; Zhou et al., 2024; Zhu et al., 2025a; Zhu et al., 2025b; Qiu et al., 2025; Roucher et al., 2025; Tang et al., 2025; Team, 2025) have demonstrated remarkable capabilities in complex problem-solving tasks, showcasing how collaborative agent frameworks can effectively address challenges requiring diverse reasoning abilities and tool manipulation. These systems leverage specialized agents with distinct roles, enabling sophisticated planning, reasoning, and tool utilization to solve tasks that would be challenging for single-agent approaches. Concurrently, research efforts have focused on Tool-Integrated Reasoning (TIR) (Jin et al., 2025a; Li et al., 2025c; Li et al., 2025d; Wu et al., 2025a; Sun et al., 2025; Zhang et al., 2025a; Zheng et al., 2025; Xue et al., 2025) approaches, which aim to incorporate the capabilities of tool execution or multi-agent systems into a single model through specialized training methodologies.

Figure 1: Overview of Flash-Searcher: Framework and Key Components.

Despite their impressive performance, both MAS and TIR approaches face significant limitations when addressing general complex tasks. Multi-agent systems suffer from inefficient tool utilization, excessively long reasoning chains, and prolonged execution times due to sequential processing and redundant communication, while TIR methods encounter reasoning efficiency bottlenecks with chains frequently exceeding context window limitations. These challenges intensify in complex scenarios requiring deep research capabilities, where MAS and TIR methodologies incorporate additional verification mechanisms (reflection, self-critique, iterative refinement) to enhance reliability—at the cost of substantially increased computational overhead. When solving complex tasks, current agent frameworks typically require more than 20 interaction steps (Wang et al., 2025; Roucher et al., 2025; Hu et al., 2025), with execution times stretching to hours. This creates a sharp tension between solution quality and computational efficiency, severely limiting practical viability in user-responsive applications. When confronted with complex tasks inducing unavoidable latency, do users deem the better performance necessary enough to justify tolerating or paying for these delays?

To address these critical challenges, we introduce Flash-Searcher, a novel parallel agent reasoning framework that fundamentally reimagines how agents collaborate to solve complex tasks. Building upon recent empirical advances in reasoning models, our approach leverages these models’ enhanced capabilities in simultaneously managing multiple cognitive threads. As illustrated in Figure 1, unlike traditional approaches that adhere to strict sequential processing, Flash-Searcher decomposes the original task into multiple parallel execution paths, orchestrated via carefully designed agent workflows. This parallelization allows multiple reasoning paths to progress simultaneously while intelligently managing tool calls across different execution branches. The Flash-Searcher framework redefines the efficiency-effectiveness frontier in complex task solving through key innovations: 1) adaptive decomposition and parallelization of tasks into concurrent subtasks with dynamic strategy adjustment, 2) dependency-aware reasoning graph management to model information dependencies and 3) optimize critical paths/information flow, and proactive information retrieval with knowledge sharing to anticipate downstream needs and reduce redundant interaction steps.

Our extensive evaluations demonstrate that Flash-Searcher achieves state-of-the-art performance across multiple challenging benchmarks. our Flash-Searcher (with GPT-5mini) reduces the average agent execution steps by 35% (11.2 → 7.4 steps) and shortens the overall execution time by ~65% compared to OAgents. Despite this dramatic efficiency improvement, Flash-Searcher (with GPT-5) achieves an impressive average performance of 82.5% on GAIA benchmark. Furthermore, on more challenging benchmarks such as xbench, HLE and BrowseComp, Flash-Searcher achieves performance metrics of 83.0, 44.0 and 67.7 respectively, surpassing current state-of-the-art methods. Furthermore, to validate the generalizability of our approach, we constructed Flash-Searcher execution trajectories based on collected web agent data and conducted post-training on the Qwen-2.5 family of open-source models. This lightweight adaptation achieves 68.0 performance on the xbench-DeepSearch benchmark, a 29.3 improvement over WebDancer, verifying the effective transfer of the parallel agent paradigm to open-source models with minimal additional training.

In summary, our contributions are as follows:

- •

We present a novel parallel agent reasoning framework that substantially reduces execution steps while achieving SOTA performance across various benchmarks.

- •

High-quality parallel reasoning trajectories, systematically curated and constructed for model post-training, significantly boost performance on complex evaluation tasks.

- •

Experimental results demonstrate the effectiveness of lightweight post-training in propagating parallel agent strategies to open-source models - achieving comparable results to multi-agent systems.

- •

We fully open-source pipeline and datasets of Flash-Searcher to catalyze research on search agents and models.

## 2 Related Work

### 2.1 Multi-agent System

Recent research has highlighted the effectiveness of multi-agent systems in addressing complex real-world challenges through collaborative agent frameworks. These systems typically employ multiple specialized agents with distinct roles, thereby supporting advanced planning, multi-turn reasoning, tool utilization, and environment interaction (Zhou et al., 2023c; Zhou et al., 2024; Jin et al., 2025b; Zhu et al., 2025a; Zhu et al., 2025b; Mai et al., 2025; Hu et al., 2024; Tang et al., 2025; Shi et al., 2025; Tang et al., 2025; Zhou et al., 2023b). Early multi-agent systems such as CAMEL (Li et al., 2023) showed that dialog between agents can elicit stepwise reasoning through role-playing. Subsequent frameworks, including MetaGPT (Hong et al., 2024) and ChatDev (Qian et al., 2023), formalized this approach by implementing structured execution pipelines with dedicated roles such as manager, designer, and coder. Other approaches, like Magnetic-One (Fourney et al., 2024) and Smolagents (Roucher et al., 2025), incorporate a central planner that dynamically delegates subtasks to specialized tool-based agents. AgentVerse (Chen et al., 2023) refines collaborative reasoning via a recruitment–decision–execution–evaluation cycle, enhancing reflection and coordination. Workforce (Hu et al., 2025) decouples planning, coordination, and execution into modular agents, enabling efficient domain transfer through plug-and-play workers. Alita (Qiu et al., 2025) proposes autonomous tool exploration via iterative trial-and-error, expanding capabilities by transforming multi-attempt tasks into single-attempt ones. However, beyond performance, the latency in these complex multi-agent frameworks remains understudied.

### 2.2 Efficient Framework

To address the efficiency bottlenecks inherent in existing agent frameworks, Tool-Integrated Reasoning (TIR) has recently emerged as a prominent research direction. Early efforts primarily adopted prompt-based strategies—such as Search-o1 (Li et al., 2025c)—which employ static templates to instantiate fixed Thought–Action–Observation loops, thereby enabling rudimentary tool-augmented reasoning. More recent work has pivoted toward post-training paradigms (Jin et al., 2025a; Li et al., 2025d; Wu et al., 2025a; Li et al., 2025a; Tao et al., 2025; Sun et al., 2025; Xue et al., 2025; Li et al., 2025b; Nguyen et al., 2025), where agents are refined via task-specific fine-tuning to enhance performance. Despite their empirical gains, these approaches typically enforce narrowly scoped execution workflows, which severely limit their adaptability and scalability in open-domain, real-world environments. These challenges have motivated a broader effort to improve the efficiency and scalability of reasoning-enabled agents. Recent advances have focused on two key directions: optimizing agent pipelines and parallelizing search processes. Efficient Agents (Wang et al., 2025) conducts a comprehensive analysis of core agent modules (workflow design, tool invocation, and memory architecture) to systematically balance performance and cost. Similarly, ParallelSearch (Zhao et al., 2025) trains models to detect parallelizable query structures, decomposing complex queries into independent sub-queries for retrieval tasks, resulting in significant performance gains in search-based tasks. However, existing systems remain constrained by isolated reasoning-execution loops or and the prolonged cycles introduced by multi-step verification, highlighting the need for more efficient approaches to agent.

## 3 Method

### 3.1 Preliminaries

Tool-Augmented Agents. Tool-augmented agents enhance the capabilities of LLMs by seamlessly integrating external tools to perform actions such as information retrieval, mathematical computation, and code execution. This paradigm mitigates the inherent limitations of parametric knowledge through a structured tool-calling pipeline. Formally, the agent–environment interaction is modeled as a Markov decision process, wherein each tool invocation induces a state transition driven by environmental feedback. At timestep $t$ , the agent selects a tool-calling action $a_{t}\in\mathcal{A}$ —where $\mathcal{A}$ denotes the action space comprising available tools—based on the current state $s_{t}$ , and receives an observation $o_{t}\sim\mathcal{P}(\cdot\mid s_{t},a_{t})$ from the tool environment. The state transition function is defined as:

$s_{t+1}=g\left(s_{t},a_{t},o_{t}\right),$ (1)

where $g:\mathcal{S}\times\mathcal{A}\times\mathcal{O}\rightarrow\mathcal{S}$ represents a state update function incorporating task, action history, and structured tool outputs into the new state representation $s_{t+1}\in\mathcal{S}$ .

Multi-Agent Systems. Consider a set of agents indexed by $\mathcal{I}$ , where each agent is denoted as $a_{i}$ for $i\in\mathcal{I}$ . Each agent maintains a local state $s^{i}_{t}$ and possesses specialized capabilities. The global system state at time $t$ is defined as $S_{t}=\{s^{1}_{t},s^{2}_{t},\ldots,s^{n}_{t},c_{t}\}$ , which aggregates all local states along with a shared context $c_{t}$ . Agents coordinate through inter-agent communication protocols to optimize a common objective function $\mathcal{U}(S_{t})$ . The evolution of the global state follows:

$S_{t+1}=f\left(\{a^{i}_{t}\}_{i\in\mathcal{I}},S_{t},O_{t}\right),$ (2)

where $a^{i}_{t}\in\mathcal{A}$ denotes the action executed by agent $i$ at timestep $t$ , $O_{t}=\{o^{i}_{t}\}_{i\in\mathcal{I}}$ represents the collection of observations from all agents, and $f$ integrates individual actions, the current global state, and observations to produce the next global state.

Existing approaches often adopt sequential execution with reflection and verification, prolonging task completion. Complex tasks may require $40+$ interactions, introducing substantial latency. This sequential dependency creates a fundamental quality–efficiency trade-off, hindering real-world deployment.

Figure 2: The pipeline of Flash-Searcher.

### 3.2 Flash-Searcher: Parallel Agent Reasoning Framework

To overcome the inherent inefficiencies of sequential execution in conventional agent frameworks, we introduce Flash-Searcher, a novel parallel reasoning framework that reformulates complex task solving as structured concurrency. Our approach transcodes the traditional linear workflow into a dynamic directed acyclic graph (DAG) plan, achieving substantial efficiency gains while preserving execution coherence. The full pipeline of Flash-Searcher is illustrated in Figure 2.

##### DAG-based Plan Construction.

Given a composite task $T$ , Flash-Searcher employs a decomposition function $\mathcal{D}$ that identifies constituent subtasks and their interdependencies, yielding a DAG-based plan:

$\mathcal{D}(T)=G_{\text{plan}}=(V,E),$ (3)

where $V=\{t_{1},t_{2},\dots,t_{n}\}$ denotes subtasks and $E\subseteq V\times V$ captures prerequisite relations. Each directed edge $(t_{i},t_{j})\in E$ encodes that $t_{i}$ must precede $t_{j}$ .

Parallel Inferential Execution & Tool Orchestration. At execution step $t$ , Flash-Searcher selects candidate subtasks from the pending set $\mathcal{P}_{t}\subseteq V$ :

$\mathcal{E}(G_{t},\mathcal{P}_{t})=\{v_{i}\in\mathcal{P}_{t}\mid\varphi(v_{i},G_{t},s_{t})=1\},$ (4)

where $\varphi(\cdot)$ is a readiness predicate. Unlike strict topological scheduling, $\varphi$ permits aggressive parallelization: a subtask $v_{i}$ may be scheduled if either (i) all its prerequisites are complete, or (ii) partial execution can provide auxiliary signals for dependency verification. Thus, $\varphi$ formalizes cross-validation as a hybrid criterion, blending dependency satisfaction and heuristic consistency checks. During execution, multiple subtasks $\mathcal{E}(G_{t},\mathcal{P}_{t})$ are processed in parallel via tool or agent invocations. The system integrates observations into the reasoning state:

$s_{t+1}=\mathcal{f}\!\left(s_{t},\{a^{(k)}_{t}\}_{k=1}^{m},\{o^{(k)}_{t}\}_{k=1}^{m}\right),$ (5)

where $a^{(k)}_{t}$ and $o^{(k)}_{t}$ denote the action and observation of the $k$ -th parallel execution, and $\mathcal{f}$ integrates the results via structured aggregation and performs state transitions based on the aggregated information.

Adaptive Progress Tracking & Summarization. To reflect execution progress, Flash-Searcher periodically updates the DAG-based plan every $\Delta$ steps:

$G_{\text{plan}}^{t+\Delta}=\mathcal{R}\!\left(G_{\text{plan}}^{t},\mathcal{C}_{t},\mathcal{P}_{t},s_{t}\right),$ (6)

where $\mathcal{C}_{t}$ is the set of completed subtasks. The refinement rule $\mathcal{R}$ eliminates resolved nodes, revalidates unresolved dependencies based on cross-validation outcomes, and dynamically inserts new decomposition nodes if needed. The interval $\Delta$ can be flexibly specified: a smaller $\Delta$ increases the frequency of plan updates, ensuring faster task adaptation and responsiveness; a larger $\Delta$ suppresses excessive optimization, reducing computational overhead in complex or stable tasks.

By integrating DAG-based decomposition, controlled aggressive parallelization, and periodic DAG optimization, Flash-Searcher mitigates the sequential bottleneck of existing reasoning architectures. The framework strikes a balance between theoretical soundness and practical efficiency, ensuring reproducibility and scalability in complex real-world tasks. The full Flash-Searcher pipeline is formally presented in Algorithm 1.

Algorithm 1 Flash-Searcher Framework

1: Composite task $T$ See eq. 3 2: $G_{\text{plan}}\leftarrow\mathcal{D}(T)$ Initial DAG plan 3: Initialize $s_{0}$ , $\mathcal{P}_{0}\leftarrow V$ , $\mathcal{C}_{0}\leftarrow\emptyset$ State setup 4: $t\leftarrow 0$ Time index 5: while $\mathcal{P}_{t}\neq\emptyset$ do Pending subtasks remain 6: $\mathcal{E}_{t}\leftarrow\{v\in\mathcal{P}_{t}\mid\varphi(v,G_{t},s_{t})=1\}$ See eq. 4 7: Execute subtasks in $\mathcal{E}_{t}$ in parallel Structured concurrency 8: Collect results $\{o^{(k)}_{t}\}$ and update $s_{t+1}=\mathcal{f}(s_{t},\{a^{(k)}_{t}\},\{o^{(k)}_{t}\})$ See eq. 5 9: $\mathcal{C}_{t+1}\leftarrow\mathcal{C}_{t}\cup\text{Completed subtasks}$ Update completed set 10: $\mathcal{P}_{t+1}\leftarrow\mathcal{P}_{t}\setminus\mathcal{C}_{t+1}$ Update pending set 11: if $t\bmod\Delta=0$ then Periodic refinement 12: $G_{t+1}\leftarrow\mathcal{R}(G_{t},\mathcal{C}_{t+1},\mathcal{P}_{t+1},s_{t+1})$ See eq. 6 13: end if 14: $t\leftarrow t+1$ Advance time 15: end while 16: return Final state $s_{T}$ Solution of $T$

## 4 Experiment

### 4.1 Flash-Searcher Framework

#### 4.1.1 Setup

Benchmarks. We evaluate Flash-Searcher on four challenging benchmarks for information retrieval and reasoning:

- •

GAIA (Mialon et al., 2023): A comprehensive benchmark for evaluating complex task-solving capabilities. For this benchmark, we mainly use the text-only validation set (103 tasks), which requires deep information retrieval and complex reasoning. Notably, the full validation set is solely used in Figure 4 for fair comparison; all other evaluations are based on the text-only validation subset.

- •

BrowseComp (Wei et al., 2025): Large-scale benchmark comprising 1,266 tasks designed to test internet-scale information retrieval with hard-to-find information needs and sophisticated browsing strategies.

- •

xbench-DeepSearch (Xbench-Team, 2025): Professional benchmark with 100 tasks simulating real-world search scenarios, emphasizing multi-round refinement and cross-source information integration.

- •

HLE (Phan et al., 2025): A frontier benchmark covering over a hundred subjects, designed to address the limited difficulty of existing benchmarks. We follow the setting in AFM (Li et al., 2025b) and use HLE-500 for evaluations.

Framework Configuration. Flash-Searcher employs a minimalist yet powerful tool configuration optimized for parallel execution. Our framework integrates two core components: a Search Tool implemented with the Serper API (Serper, 2025) for retrieving structured search results, and a Crawl Tool leveraging the Jina Reader (Jina, 2025) for content extraction. The crawl tool incorporates automatic summarization using the same backbone language model, ensuring consistent information representation while significantly reducing cognitive load. This streamlined design enables efficient parallel tool orchestration across reasoning branches while maintaining trajectory simplicity and operational coherence. More details can be found in Appendix C.

Metrics. We employ the LLM-as-Judge paradigm (Zheng et al., 2023; Wu et al., 2025a) for automated evaluation, utilizing GPT-4.1-mini as the judge model. Each agent output of different benchmarks receives a binary correctness assessment from the judge model. For final performance reporting, we default to presenting Pass@1 results: this metric quantifies the proportion of tasks where the agent generates a correct output on the first attempt. Specifically, the Pass@1 score for each individual benchmark is calculated based on its binary assessment results, and these per-benchmark Pass@1 scores are ultimately aggregated to report the overall performance. The standardized prompt for judgment is detailed in Section H.1.

#### 4.1.2 Main Results

We present a comprehensive evaluation of Flash-Searcher against state-of-the-art closed-source and open-source agent frameworks across four challenging benchmarks: BrowseComp, xbench-DeepSearch, GAIA, and HLE. As illustrated in Figure 4, our method achieves highly competitive performance, matching or exceeding existing approaches while demonstrating superior efficiency and scalability. These results underscore the effectiveness of our DAG-based architecture in handling diverse task complexities.

Figure 3: Performance of Flash-Searcher with different backbones.

In Figure 3, our Flash-Searcher when integrated with GPT-5 achieves a competitive performance of 67.7% on the BrowseComp benchmark. This result not only demonstrates a substantial advantage over state-of-the-art open-source frameworks (e.g., BrowseMaster (Pang et al., 2025), which attains 30.0%) but also approaches the performance of the leading closed-source solution, specifically the OpenAI ChatGPT agent (68.9%). Even with less powerful backbone models such as GPT-5-mini, our framework achieves 35.3%, demonstrating the effectiveness of our parallel reasoning approach regardless of the underlying model. For xbench-DeepSearch, Flash-Searcher also shows remarkable performance, with our GPT-5 variant achieving 83%, surpassing both BrowseMaster (66%) and Metaso DeepResearch (64%). This substantial improvement highlights the particular strength of our approach in deep research scenarios that demand extensive information gathering and complex reasoning. Besides, On the GAIA benchmark, Flash-Searcher with lightweight, resource-efficient GPT-5-mini backbone achieves 80.6%, exceeding even strong closed-source systems like Alita (75.2%) and Manus (73.3%). Additionally, our method demonstrates exceptional capability on the HLE benchmark, achieving a state-of-the-art 44.0% with GPT-5, substantially outperforming all other frameworks.

Figure 4: Performance comparison of agent frameworks on BrowseComp, xbench-DeepSearch, GAIA and HLE benchmarks. All results are reported using Pass@1 metric.

These results demonstrate that our Flash-Searcher, a parallel reasoning framework, effectively addresses the multifaceted challenges inherent to information retrieval tasks. The consistent performance of Flash-Searcher across diverse backbone models spanning different architectures and capability levels further validates the robustness of our approach.

### 4.2 Flash-Searcher Learning

#### 4.2.1 Setup

Dataset. To train our parallel reasoning agent, we construct a high-quality dataset derived from multiple sources including WebWalker (Wu et al., 2025b), ASearcher (Gao et al., 2025), WebShaper (Tao et al., 2025), and CoA (Li et al., 2025b). Our final dataset consists of 3354 effective DAG-based reasoning trajectories. Each trajectory incorporates periodic DAG workflow reviews and is formatted as a multi-turn dialogue, enabling effective context window extrapolation and long-range dependency modeling. This format specifically enhances the model’s ability to manage complex reasoning graphs while maintaining coherent conversation flow. More details can be found in Section F.1

Training Configurations. We maintain consistent evaluation metrics and benchmarks with the framework experiments in Section 4.1.1. All training is implemented using the Llama-Factory framework (Zheng et al., 2024). We employ supervised fine-tuning to develop robust parallel reasoning capabilities. Specifically, for all trained models, the maximum dialogue length is set to 131,072 tokens, the learning rate is set to $10^{-5}$ , and training is conducted for four epochs. The full training parameters and detailed data formatting specifications are comprehensively documented in Section F.2.

#### 4.2.2 Agent Model Results

To validate the effectiveness of our parallel reasoning approach beyond framework implementation, we distilled Flash-Searcher’s parallel reasoning capabilities into standalone agent models through lightweight supervised fine-tuning. Table 1 presents a comprehensive comparison of these agent models against existing state-of-the-art methods across four challenging benchmarks.

Our experimental analysis demonstrates that lightweight supervised fine-tuning effectively facilitates the transfer of Flash-Searcher’s parallel reasoning capabilities to standalone agent models, consistently achieving state-of-the-art (SOTA) performance across diverse benchmarks and model backbone scales. Specifically, on the Qwen-2.5-32B backbone, Flash-Searcher establishes a new performance ceiling. It outperforms the strongest prior method by 3.3% on BrowseComp, 5.0% on xBench-DeepSearch, and 2.0% on GAIA. Despite forgoing code interpreter tools, Flash-Searcher achieves state-of-the-art performance at 19.4% on HLE, surpassing tool-augmented baselines and affirming the general effectiveness of Flash-Searcher in handling general complex tasks. This result underscores Flash-Searcher’s inherent reasoning robustness, as it delivers strong performance without relying on extensive tools.

Scaling Flash-Searcher to 72B yields consistent and meaningful performance gains across all benchmarks, demonstrating that our parallel reasoning framework scales gracefully with model capacity. Notably, the most substantial improvements occur on complex, multi-step reasoning tasks such as BrowseComp and xbench-DeepSearch, with 5% gains, suggesting that increased parameter scale enhances the model’s ability to coordinate and refine reasoning steps. Even on HLE, the performance affirms that Flash-Searcher internalizes structured reasoning without relying on external tools. This behavior confirms that our lightweight fine-tuning paradigm not only transfers reasoning capabilities effectively but also unlocks deeper potential as backbone capacity grows, making it suited for scalable, general-purpose agent deployment.

Notably, these results are achieved through lightweight supervised fine-tuning without RL or tool reliance. This confirms that parallel reasoning is a learnable and scalable inductive bias, efficiently transferred via minimal supervision. Flash-Searcher thus emerges as a simple, robust, and parameter-efficient solution for real-world agents.

Table 1: Performance comparison of agent models on BrowseComp, xbench-DeepSearch, and GAIA benchmarks. All results are reported using Pass@1 metric. Gray-font values correspond to results reported in the associated reports.

Method Backbone BrowseComp xbench-DeepSearch GAIA HLE

Cognitive Kernel-Pro Qwen-3-8B - - 43.7 -

WebDancer QwQ-32B 3.8 39.0 50.5 7.2

WebThinker-RL 2.8 24.0 48.5 -

SimpleDeepSearcher - - 50.5 -

WebShaper - - 53.3 12.2

SFR-DR - - 52.4 17.1

WebDancer Qwen-2.5-32B 2.5 38.7 40.7 -

SimpleDeepSearcher - - 40.8 -

WebShaper - - 52.4 -

WebSailor 10.5 53.3 53.2 10.8

AFM-RL 11.1 58.0 55.3 18.0

Flash-Searcher 14.4 63.0 57.3 19.4

WebSailor Qwen-2.5-72B 12.0 55.0 55.4 -

WebShaper - - 60.1 -

Flash-Searcher 18.9 68.0 61.2 20.2

## 5 Efficiency Analysis

We present a comprehensive efficiency analysis of Flash-Searcher using the GPT-5-mini backbone, evaluating its execution efficiency and framework improvements compared to existing agent systems. The distribution plot in Figure 5(a) demonstrates BrowseComp benchmark requiring the highest number of both metrics. This reflects the varying complexity demands across different benchmark types. Figure 5(b) reveals Flash-Searcher’s operational efficiency through tool calls per execution step. The tight interquartile range, particularly evident in the GAIA benchmark, indicates consistent and predictable tool utilization patterns. This evidence directly supports Flash-Searcher’s core claim: its DAG-based architecture optimizes tool utilization efficiency while cutting extraneous execution steps. Specifically, the architecture’s built-in parallel tool invocation enables concurrent execution of complementary tools, eliminating sequential bottlenecks that cause redundant steps in linear pipelines.

(a) Tool calls vs. execution steps.

(b) Tool calls per execution step of Flash-Searcher.

Figure 5: Efficiency analysis of Flash-Searcher on four benchmarks: (5(a)) shows the correlation between tool calls and steps; (5(b)) characterizes the distribution of tool calls per step.

To fairly evaluate the execution efficiency of Flash-Searcher, we compare Flash-Searcher against OAgents (Zhu et al., 2025a) and OWL-Roleplaying (Hu et al., 2025) with their original configurations (Details in Appendix G). The experimental results are presented in Figure 6, which demonstrates significant efficiency improvements of our approach across four benchmarks.

As shown in Figure 6(a), Flash-Searcher outperforms OAgents on all four benchmarks, achieving higher task success rates and efficiency gains—with this advantage growing more pronounced as task complexity increases (BrowseComp > xbench-DeepSearch > HLE > GAIA). This validates Flash-Searcher’s adaptability to complex scenarios, laying the foundation for subsequent efficiency analysis. Figure 6(b) further demonstrates that Flash-Searcher (with GPT-5-mini backbone) reduces agent steps by 35% versus OAgents and 30% versus OWL-Roleplaying on GAIA benchmark, enabled by its parallel reasoning architecture. This efficiency gain stems from the DAG-based workflow’s ability to execute concurrent reasoning paths, which effectively mitigates the sequential bottleneck of traditional methods. Figure 5 illustrates the distribution of tool calls and steps for Flash-Searcher: despite fewer total steps, our approach maintains higher per-step tool utilization efficiency (average 3.00 tool calls per step, compared to 0.83 for OAgents and 0.85 for OWL-Roleplaying), confirming more productive and effective reasoning iterations.

(a) Execution steps and tool calls comparison.

(b) Efficiency comparison of frameworks on GAIA.

Figure 6: Efficiency comparison of agent frameworks on four benchmarks.

The core innovation lies in our DAG-based parallel execution mechanism, which directly addresses the fundamental limitation of redundant tool invocation cycles in sequential reasoning approaches. By coordinating information needs across parallel branches, we eliminate duplicate searches while maintaining reasoning diversity. As in Figure 6(a), our framework simultaneously enhances both efficiency and performance, effectively resolving the longstanding efficiency-effectiveness trade-off in agent systems.

Although agent execution duration is inherently influenced by external factors such as API rate limits, Flash-Searcher consistently achieves a 35% reduction in execution steps under comparable environmental conditions. This reduction directly translates into lower end-to-end latency and improved throughput, offering a significant efficiency advantage. Such gains are especially critical in latency-sensitive applications and high-throughput deployment settings, where conventional sequential agent architectures encounter substantial scalability and responsiveness bottlenecks.

## 6 Conclusion

In this work, we introduce Flash-Searcher, a novel parallel agent reasoning framework that overcomes the sequential bottlenecks of conventional tool-augmented agents through structured concurrency. By reformulating task solving as dynamic scheduling over DAGs, Flash-Searcher enables fine-grained parallel execution while rigorously preserving logical coherence and correctness. Extensive experiments across BrowseComp, xbench-DeepResearch, GAIA, and HLE demonstrate that Flash-Searcher achieves state-of-the-art performance, attaining a score of 67.7% on BrowseComp, alongside substantial gains in computational efficiency through reduced latency and improved resource utilization. Our results, further corroborated by distilled agent variants, establish parallel reasoning as a foundational paradigm for building efficient, scalable, and robust AI systems capable of mastering complex real-world tasks.

## 7 Contributions

Contributors

-

Tianrui Qin ’ Qianben Chen ’

Sinuo Wang ’ He Xing ’

King Zhu ’ He Zhu ’

Dingfeng Shi ’ Xinxin Liu ’

Ge Zhang ’ Jiaheng Liu ’

Yuchen Eleanor Jiang ’

Corresponding Authors

-

Xitong Gao ’ Wangchunshu Zhou ’

## References

- Besta et al. (2024) Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI conference on artificial intelligence, 2024.

- Canese et al. (2021) Lorenzo Canese, Gian Carlo Cardarilli, Luca Di Nunzio, Rocco Fazzolari, Daniele Giardino, Marco Re, and Sergio Spanò. Multi-agent reinforcement learning: A review of challenges and applications. Applied Sciences, 11(11):4948, 2021.

- Chen et al. (2023) Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chen Qian, Chi-Min Chan, Yujia Qin, Yaxi Lu, Ruobing Xie, et al. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents. arXiv preprint arXiv:2308.10848, 2(4):6, 2023.

- Dorri et al. (2018) Ali Dorri, Salil S Kanhere, and Raja Jurdak. Multi-agent systems: A survey. Ieee Access, 6:28573–28593, 2018.

- Fourney et al. (2024) Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, et al. Magentic-one: A generalist multi-agent system for solving complex tasks. arXiv preprint arXiv:2411.04468, 2024.

- Gao et al. (2025) Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous rl. arXiv preprint arXiv:2508.07976, 2025.

- Hendrycks et al. (2020) Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

- Hong et al. (2024) Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, et al. Metagpt: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations, ICLR, 2024.

- Hu et al. (2025) Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Qiguang Chen, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Bernard Ghanem, Ping Luo, and Guohao Li. Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation, 2025. URL https://arxiv.org/abs/2505.23885.

- Hu et al. (2024) Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, et al. Os agents: A survey on mllm-based agents for computer, phone and browser use, 2024.

- Jin et al. (2025a) Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025a.

- Jin et al. (2025b) Yiyang Jin, Kunzhao Xu, Hang Li, Xueting Han, Yanmin Zhou, Cheng Li, and Jing Bai. Reveal: Self-evolving code agents via iterative generation-verification, 2025b. URL https://arxiv.org/abs/2506.11442.

- Jina (2025) Inc. Jina. Jina reader, 2025. URL https://jina.ai/reader/.

- Li et al. (2023) Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large language model society. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

- Li et al. (2025a) Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. Websailor: Navigating super-human reasoning for web agent, 2025a. URL https://arxiv.org/abs/2507.02592.

- Li et al. (2025b) Weizhen Li, Jianbo Lin, Zhuosong Jiang, Jingyi Cao, Xinpeng Liu, Jiayu Zhang, Zhenqiang Huang, Qianben Chen, Weichen Sun, Qiexiang Wang, et al. Chain-of-agents: End-to-end agent foundation models via multi-agent distillation and agentic rl. arXiv preprint arXiv:2508.13167, 2025b.

- Li et al. (2025c) Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint arXiv:2501.05366, 2025c.

- Li et al. (2025d) Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou. Webthinker: Empowering large reasoning models with deep research capability. arXiv preprint arXiv:2504.21776, 2025d.

- Liu et al. (2023) Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. Llm+ p: Empowering large language models with optimal planning proficiency. arXiv preprint arXiv:2304.11477, 2023.

- Mai et al. (2025) Xinji Mai, Haotian Xu, Weinong Wang, Yingying Zhang, Wenqiang Zhang, et al. Agent rl scaling law: Agent rl with spontaneous code execution for mathematical problem solving. arXiv preprint arXiv:2505.07773, 2025.

- Mialon et al. (2023) Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations, 2023.

- Nguyen et al. (2025) Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming Xiong, and Shafiq Joty. Sfr-deepresearch: Towards effective reinforcement learning for autonomously reasoning single agents. arXiv preprint arXiv:2509.06283, 2025.

- Pan et al. (2025) Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, and Alane Suhr. Learning adaptive parallel reasoning with language models. Conference on Language Modeling, 2025.

- Pang et al. (2025) Xianghe Pang, Shuo Tang, Rui Ye, Yuwen Du, Yaxin Du, and Siheng Chen. Browsemaster: Towards scalable web browsing via tool-augmented programmatic agent pair. arXiv preprint arXiv:2508.09129, 2025.

- Phan et al. (2025) Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025.

- Qian et al. (2023) Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software development. arXiv preprint arXiv:2307.07924, 2023.

- Qiu et al. (2025) Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, et al. Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution. arXiv preprint arXiv:2505.20286, 2025.

- Roucher et al. (2025) Aymeric Roucher, Albert Villanova del Moral, Thomas Wolf, Leandro von Werra, and Erik Kaunismäki. ‘smolagents‘: a smol library to build great agentic systems. https://github.com/huggingface/smolagents, 2025.

- Sel et al. (2023) Bilgehan Sel, Ahmad Al-Tawaha, Vanshaj Khattar, Ruoxi Jia, and Ming Jin. Algorithm of thoughts: Enhancing exploration of ideas in large language models. arXiv preprint arXiv:2308.10379, 2023.

- Serper (2025) Inc. Serper. Serper api, 2025. URL https://serper.dev/.

- Shi et al. (2025) Dingfeng Shi, Jingyi Cao, Qianben Chen, Weichen Sun, Weizhen Li, Hongxuan Lu, Fangchen Dong, Tianrui Qin, King Zhu, Minghao Yang, et al. Taskcraft: Automated generation of agentic tasks. arXiv preprint arXiv:2506.10055, 2025.

- Sun et al. (2025) Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, et al. Simpledeepsearcher: Deep information seeking via web-powered reasoning trajectory synthesis. arXiv preprint arXiv:2505.16834, 2025.

- Tang et al. (2025) Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, Ge Zhang, Jiaheng Liu, Xingyao Wang, Sirui Hong, Chenglin Wu, Hao Cheng, Chi Wang, and Wangchunshu Zhou. Agent kb: Leveraging cross-domain experience for agentic problem solving. In ICML 2025 Workshop on Collaborative and Federated Agentic Workflows, 2025.

- Tao et al. (2025) Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webshaper: Agentically data synthesizing via information-seeking formalization, 2025. URL https://arxiv.org/abs/2507.15061.

- Team (2025) MiroMind AI Team. Miroflow: An open-source agentic framework for deep research. https://github.com/MiroMindAI/MiroFlow, 2025.

- Wang et al. (2025) Ningning Wang, Xavier Hu, Pai Liu, He Zhu, Yue Hou, Heyuan Huang, Shengyu Zhang, Jian Yang, Jiaheng Liu, Ge Zhang, et al. Efficient agents: Building effective agents while reducing cost. arXiv preprint arXiv:2508.02694, 2025.

- Wei et al. (2025) Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.

- Wu et al. (2025a) Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Yong Jiang, Pengjun Xie, et al. Webdancer: Towards autonomous information seeking agency. arXiv preprint arXiv:2505.22648, 2025a.

- Wu et al. (2025b) Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, et al. Webwalker: Benchmarking llms in web traversal. arXiv preprint arXiv:2501.07572, 2025b.

- Xbench-Team (2025) Xbench-Team. Xbench-deepsearch, 2025. URL https://xbench.org/agi/aisearch.

- Xue et al. (2025) Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Zejun Ma, and Bo An. Simpletir: End-to-end reinforcement learning for multi-turn tool-integrated reasoning. https://simpletir.notion.site/report, 2025. Notion Blog.

- Yao et al. (2023) Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

- Zhang et al. (2025a) Dingchu Zhang, Yida Zhao, Jialong Wu, Baixuan Li, Wenbiao Yin, Liwen Zhang, Yong Jiang, Yufeng Li, Kewei Tu, Pengjun Xie, et al. Evolvesearch: An iterative self-evolving search agent. arXiv preprint arXiv:2505.22501, 2025a.

- Zhang et al. (2025b) Shiqi Zhang, Xinbei Ma, Zouying Cao, Zhuosheng Zhang, and Hai Zhao. Plan-over-graph: Towards parallelable llm agent schedule. arXiv preprint arXiv:2502.14563, 2025b.

- Zhao et al. (2025) Shu Zhao, Tan Yu, Anbang Xu, Japinder Singh, Aaditya Shukla, and Rama Akkiraju. Parallelsearch: Train your llms to decompose query and search sub-queries in parallel with reinforcement learning. arXiv preprint arXiv:2508.09303, 2025.

- Zheng et al. (2023) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

- Zheng et al. (2024) Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. arXiv preprint arXiv:2403.13372, 2024.

- Zheng et al. (2025) Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. arXiv preprint arXiv:2504.03160, 2025.

- Zhou et al. (2023a) Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning acting and planning in language models. arXiv preprint arXiv:2310.04406, 2023a.

- Zhou et al. (2023b) Wangchunshu Zhou, Yuchen Eleanor Jiang, Peng Cui, Tiannan Wang, Zhenxin Xiao, Yifan Hou, Ryan Cotterell, and Mrinmaya Sachan. Recurrentgpt: Interactive generation of (arbitrarily) long text, 2023b. URL https://arxiv.org/abs/2305.13304.

- Zhou et al. (2023c) Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, et al. Agents: An open-source framework for autonomous language agents. arXiv preprint arXiv:2309.07870, 2023c.

- Zhou et al. (2024) Wangchunshu Zhou, Yixin Ou, Shengwei Ding, Long Li, Jialong Wu, Tiannan Wang, Jiamin Chen, Shuai Wang, Xiaohua Xu, Ningyu Zhang, et al. Symbolic learning enables self-evolving agents. arXiv preprint arXiv:2406.18532, 2024.

- Zhu et al. (2025a) He Zhu, Tianrui Qin, King Zhu, Heyuan Huang, Yeyi Guan, Jinxiang Xia, Yi Yao, Hanhao Li, Ningning Wang, Pai Liu, Tianhao Peng, Xin Gui, Xiaowan Li, Yuhui Liu, Yuchen Eleanor Jiang, Jun Wang, Changwang Zhang, Xiangru Tang, Ge Zhang, Jian Yang, Minghao Liu, Xitong Gao, Wangchunshu Zhou, and Jiaheng Liu. Oagents: An empirical study of building effective agents, 2025a. URL https://arxiv.org/abs/2506.15741.

- Zhu et al. (2025b) King Zhu, Hanhao Li, Siwei Wu, Tianshun Xing, Dehua Ma, Xiangru Tang, Minghao Liu, Jian Yang, Jiaheng Liu, Yuchen Eleanor Jiang, Changwang Zhang, Chenghua Lin, Jun Wang, Ge Zhang, and Wangchunshu Zhou. Scaling test-time compute for llm agents, 2025b. URL https://arxiv.org/abs/2506.12928.

## Appendix A Limitations and Future Work

While our approach demonstrates significant improvements in agent capabilities, several limitations should be acknowledged.

Our primary focus in this work has been enhancing the execution efficiency of agents, necessitating a careful balance between performance and computational resources. To ensure fair comparisons and practical deployment scenarios, we imposed execution step limitations across all evaluations–40 steps for Flash-Searcher framework and model variants. These constraints, while necessary for efficiency considerations, prevented the complete resolution of certain complex queries, particularly evident in approximately 25% (for framework) and 75% (for framework) of the BrowseComp where additional reasoning steps may yield correct solutions. Furthermore, our crawl tool truncates retrieved web content before generating a summary, which introduces an additional source of information loss that may impact final performance. To further validate this observation, we conducted additional evaluations of Flash-Searcher with an extended number of reasoning steps; the detailed results are provided in Appendix E. It is worth noting that in unconstrained settings where computational cost is not a primary concern, Flash-Searcher would likely achieve even higher performance metrics.

Furthermore, we observed suboptimal performance on mathematical reasoning tasks in benchmarks like HLE, primarily due to the absence of code execution tools. This design choice was deliberate, as parallel code tool invocations would significantly increase model output volume, severely impacting the efficiency benefits of our parallel reasoning architecture. The substantial overhead in managing concurrent code execution environments would counteract the performance gains achieved through our approach. We believe that mathematical reasoning performance could be substantially improved with appropriate computational tools, but integrating them required architectural trade-offs beyond the scope of this work.

Our Flash-Searcher architecture is inherently compatible with supplementary reflection and verification mechanisms, which could further enhance accuracy and reliability. Such extensions represent promising directions for future work, particularly in deployment scenarios where resource efficiency can be traded for increased precision. An especially promising direction involves multi-agent architectures where specialized code execution agents could be invoked to solve mathematical sub-tasks while maintaining the efficiency advantages of our parallel reasoning approach. This hybrid architecture would preserve the computational benefits of our framework while addressing the current limitations in mathematical reasoning capabilities.

Additionally, while our parallel reasoning approach significantly improves efficiency, there remain opportunities to develop more sophisticated orchestration mechanisms that could dynamically allocate reasoning resources based on task complexity. Further research could also explore the integration of our methodology with emerging model architectures and specialized domain knowledge to address increasingly complex multi-step reasoning challenges.

Despite these limitations, we believe our work represents an important step toward more efficient and capable agent systems, establishing a foundation for future innovations in this rapidly evolving field.

## Appendix B Discussion of DAG/Graph Reasoning Methods

### B.1 Relationship to Graph/Tree-Structured Reasoning

While our work builds upon recent graph-based reasoning frameworks, key distinctions exist in purpose and implementation. Graph of Thoughts (GoT) [Besta et al., 2024] models reasoning steps as graph structures but emphasizes symbolic reasoning rather than tool execution. Tree of Thoughts (ToT) [Yao et al., 2023] explores branching reasoning paths through tree structures but prioritizes depth-first search over parallelization. Algorithm of Thoughts (AoT) [Sel et al., 2023] provides algorithmic guidance for reasoning, whereas our framework optimizes parallel tool execution specifically.

### B.2 Distinctions from Parallel Reasoning and Planning Frameworks

Recent works explore parallel reasoning and planning for LLM-based agents, yet differ significantly from our approach. Learning Adaptive Parallel Reasoning (LAPR) [Pan et al., 2025] introduces parallelization for language model reasoning but focuses primarily on model-internal computation rather than coordinating external tool calls. Plan-over-Graph (PoG) [Zhang et al., 2025b] shares conceptual similarities with our DAG approach but emphasizes strict dependency enforcement, whereas our framework intentionally relaxes these constraints to maximize parallel execution efficiency while ensuring result validity through cross-validation.

While Language Agent Tree Search (LATS) [Zhou et al., 2023a] and LLM+P [Liu et al., 2023] effectively integrate planning and acting through tree search and optimal planning techniques, they lack explicit mechanisms for parallel tool execution. Our framework complements these efforts by specializing DAG structures specifically for efficient information retrieval across multiple sources.

Our primary contribution is an efficient dynamic DAG-based planning framework that optimizes execution trajectories in real time. It enables two core capabilities: (1) parallel tool invocation for faster computation, and (2) cross-validation across dependent subtasks to preserve result integrity—resolving the key efficiency-accuracy trade-off in complex workflows. Beyond efficiency, the framework addresses a critical LLM limitation: context length constraints. By continuously summarizing intermediate states and refining paths via real-time outcomes, it ensures lengthy multi-step workflows remain tractable without losing information fidelity.

## Appendix C Experiment Details

### C.1 Benchmarks.

We evaluate the effectiveness of Flash-Searcher on four challenging benchmarks that require sophisticated information retrieval and reasoning capabilities:

- •

GAIA [Mialon et al., 2023]: As a milestone benchmark for General AI Assistants, it constructs real-world questions that necessitate fundamental capabilities including reasoning, multi-modality handling, web browsing, and tool-use proficiency. To ensure rigorous and comparable evaluation, we conduct experiments primarily on the text-only validation subset of GAIA, which consists of 103 carefully curated cases — this subset specifically highlights the challenges of disambiguating ambiguous queries and synthesizing multi-source information. Additionally, for fair comparison with existing works, we further evaluate on the full validation set (165 cases) of GAIA. Following the framework of OAgents [Zhu et al., 2025a], we additionally integrate text, image, and audio tools into our evaluation pipeline to align with the multi-modality and tool-use design goals of the full validation set.

- •

BrowseComp [Wei et al., 2025]: A rigorous benchmark comprising 1,266 questions designed to measure persistent web browsing capabilities for finding hard-to-find, entangled information. While avoiding challenges like long-form generation, it specifically tests an agent’s ability to formulate effective queries, navigate search results, extract relevant information, and synthesize coherent answers through sophisticated browsing strategies.

- •

xbench-DeepSearch [Xbench-Team, 2025]: A professionally curated benchmark focusing specifically on deep-search capabilities in Chinese contexts, featuring 100 expert-written questions requiring multi-round search refinement and cross-source integration. Designed to isolate and evaluate the Planning → Search → Reasoning → Summarization pipeline of agent systems.

- •

HLE [Phan et al., 2025]: To address the saturation of existing benchmarks (e.g., MMLU [Hendrycks et al., 2020], where SOTA LLMs now exceed 90% accuracy), HLE is proposed as a benchmark of 2,500 highly difficult questions across dozens of subjects, serving as a "final" closed-ended test for broad academic capabilities. Developed by experts via multi-stage review (pre-filtering, graduate/ expert validation, public feedback), it is multi-modal (text-only/image-accompanied), supports automated verification (multiple-choice/exact-match), and its questions are original, lookup-resistant, and emphasize advanced math for deep reasoning. Following AFM’s setup [Li et al., 2025b], we use the HLE500 subset to evaluate model performance on high-difficulty reasoning.

These benchmarks collectively provide a comprehensive evaluation landscape, assessing both the efficiency and effectiveness of our parallel agent reasoning framework across diverse information retrieval scenarios of varying complexity.

### C.2 Tool Configurations.

To ensure streamlined and efficient agent workflows and models, Flash-Searcher employs a minimalist but powerful tool configuration focused on maximizing information retrieval capabilities while maintaining trajectory simplicity:

For external tools, we deliberately constrain our framework to just two essential components:

- •

Search Tool: We implement this tool using the Serper API [Serper, 2025] to support agents in retrieving web-based information for knowledge-intensive tasks. By default, each API call returns 5 relevance-ranked results, structured to include core elements: descriptive titles (for rapid relevance screening), concise content snippets (to pre-assess information utility), and direct URLs (for deep exploration of primary sources). This configuration strikes a balance between comprehensiveness—ensuring access to high-value sources—and computational efficiency, avoiding information overload that could hinder agent decision-making.

- •

Crawl Tool: Implemented using the Jina Reader [Jina, 2025], this tool enables agents to extract and process content from specific web pages. To enhance efficiency and maintain trajectory conciseness, our crawl tool incorporates an automatic summarization mechanism that extracts and condenses the most relevant information from web pages. Specifically, considering the constraints of model context window length and the cost control of API calls in large-scale experiments, we introduce a content truncation strategy for web pages: only the first 60,000 characters of each web page are selected as the input for the summarization mechanism to perform information extraction and condensation. This design balances the trade-off between information coverage and practical implementation costs, while it should be noted that the truncation may lead to the loss of potential valuable information in the latter part of long web pages—thus, our current results still leave room for further optimization $\smiley{}$ . This approach significantly reduces cognitive load on the agent by eliminating the need to straightly process extensive raw HTML content.

The summarization component within the crawl tool utilizes the same language model as our backbone agent, ensuring consistency in understanding and representation across the framework. This architectural decision not only streamlines the information flow but also reduces potential misalignments between different components of the system.

By adopting this focused tool configuration, Flash-Searcher achieves a balance between capability and efficiency. The framework provides agents with sufficient tools to tackle complex information retrieval tasks while avoiding the overhead and complexity associated with managing numerous specialized tools. This approach is particularly advantageous in our parallel execution context, where multiple tool calls can be orchestrated simultaneously across different branches of the reasoning graph.

### C.3 Model List.

In our experiments, we employed a diverse set of state-of-the-art LLMs. The evaluated LLMs include GPT-5 (Reasoning effort: medium; version: 2025-08-07), GPT-5-mini (Reasoning effort: medium; version: 2025-08-07), GPT-5-nano (2025-08-07), GPT-4.1, DeepSeek-v3.1 (w/o thinking), and GLM-4.5 (Default). In all experiments, we maintained consistent hyperparameters across comparable settings, with temperature set to 1.0. All models were accessed via reliable API endpoints with consistent system prompts to ensure fair comparison.

### C.4 Parameters of Flash-Searcher.

To ensure the reproducibility and clarity of the Flash-Searcher framework’s implementation, this section details all key hyperparameters and configuration settings used in its execution. These parameters collectively govern critical behaviors of the framework, such as the scope of concurrent optimization objectives, the granularity of step-wise task execution, the constraints on tool utilization, and the rules for progress tracking and information retrieval. Specific configurations are summarized in Table 2 below.

Table 2: Parameter configurations for Flash-Searcher Framework.

Parameter Description Value

Parallel goals Number of concurrent objectives 5

Goal path length Predefined steps per goal 5

Max tool calls per step Maximum tool invocations per step 5/10

Max steps Total step budget for task execution 40

Summary interval Steps between progress summaries 7–9

Search retrievals per query Results returned per search call 5

Max length of extracted content Max characters extracted by crawl_page 60,000

In addition to the framework-level execution parameters detailed above, the inference process of the Flash-Searcher models—responsible for decision-making (e.g., goal prioritization, tool selection) and content generation (e.g., progress summarization, query formulation)—relies on a set of critical model-specific inference parameters. These parameters directly influence the model’s reasoning depth, output stability, and computational efficiency, and are tightly aligned with the framework’s execution constraints (e.g., step budget, tool call limits) to ensure coherent end-to-end performance. To support efficient and scalable inference, we adopt the vllm framework (a high-throughput LLM serving framework optimized for GPU acceleration) and deploy the system on a hardware cluster consisting of 8 NVIDIA A800 GPUs. Specific inference configurations (model-specific) and hardware-framework settings are summarized in Table 3 and Table 4 below, respectively.

Table 3: Inference parameter configurations for Flash-Searcher Models.

Parameter Description Value

Context length Maximum context tokens 131072 (32B) / 65536 (72B)

Max steps Total conversation length 40

Max output tokens per call Maximum generated tokens per inference step 8192

Temperature, top-k, top-p Probabilistic generation controls Default

Other inference tool parameters are designed to maintain consistency with the framework’s execution settings. Meanwhile, the inference configurations are tailored to match the model size (32B/72B parameters) and context length requirements, avoiding memory bottlenecks during long-sequence reasoning. These cross-parameter alignments are critical for avoiding misalignment between the model’s reasoning process and the execution environment, thereby ensuring reproducibility and stability of the Flash-Searcher system’s performance across different task instances.

Table 4: Configurations for Flash-Searcher inference.

Configuration Description Value

Inference framework Serving framework for inference vllm v0.10.1.1

GPU type Hardware accelerator model NVIDIA A800 (80GB)

Tensor parallelism GPU partitioning strategy 8

RoPE scaling Extending context length Dynamic (factor=4.0 for 32B; factor=2.0 for 72B)

Model of Crawl Tool Model for crawling data summary GPT-5-mini

### C.5 Detailed Results of Flash-Searcher.

To comprehensively evaluate the effectiveness of the proposed Flash-Searcher framework, we conduct extensive experiments on four representative benchmarks for agent systems. The performance is quantified using the widely adopted Pass@1 metric, which measures the proportion of tasks successfully completed by the agent in a single attempt. Table 5 and Figure 7 present the detailed performance comparison between Flash-Searcher and existing state-of-the-art agent frameworks or models. For fairness and reference, values displayed in gray font are directly quoted from the original reports of the compared methods.

Table 5: Performance comparison of agent frameworks on BrowseComp, xbench-DeepSearch, and GAIA benchmarks. All results are reported using Pass@1 metric. Gray-font values correspond to results reported in the associated reports. Note that Flash-Searcher achieve 83.0 for full validation set.

Method Backbone BrowseComp xbench-DeepSearch GAIA HLE

Closed-Source Frameworks

OpenAI ChatGPT agent - 68.9 - 41.6

OpenAI DeepResearch - 51.5 - 67.4 26.6

Metaso DeepResearch MetaLLM etc. 12.0 64 - -

Skywork DeepResearch Claude-Sonnet-3-7 etc. - - 82.4 25.9

Kimi Researcher Kimi k-series etc. - 69 - 26.9

Manus Claude etc. - - 73.3 -

Alita Claude-Sonnet-4 - - 75.2 -

Open-Source Frameworks

Smolagents OpenAI-o1 - - 49.7 -

A-World Gemini-2.5-Pro - - 71.0 -

Cognitive Kernel-Pro Claude-Sonnet-3-7 - - 66.1 -

OWL–Workforce Claude-Sonnet-3-7 - - 69.7 -

OAgents GPT-4.1 13.7 47 58.3 15.4

BrowseMaster DeepSeek-R1-0528 30.0 66 68.0 -

MiroFlow GPT-5 33.2 72 82.4 29.5

Flash-Searcher GPT-5 67.7 83 82.5 44.0

GPT-5-mini 35.3 71 80.6 32.2

GPT-4.1 31.4 57 69.9 39.6

DeepSeek-V3.1 18.6 70 62.1 22.8

GLM-4.5 - 63 63.1 -

GPT-5-nano - 61 54.4 -

(a) Performance comparisons.

(b) Scaling analysis of Flash-Searcher.

Figure 7: Performance and scaling analysis of Flash-Searcher. (7(a)): Flash-Searcher-32B consistently outperforms the SOTA method across all four benchmarks with Qwen-2.5-32B. (7(b)): Scaling Flash-Searcher from 32B to 72B parameters yields consistent gains.

## Appendix D Additional Ablations

## Appendix E Model Ablations on Execution Step Constraints

Figure 8: Performance of Flash-Searcher models under different reasoning step constraints.

To validate the observation that execution step limitations constrain the resolution of complex queries, we conduct ablations on the Flash-Searcher models, focusing on the impact of extended maximum reasoning steps. Specifically, we evaluate model performance on BrowseComp-100 (a subset of BrowseComp) under an extended maximum step limit of 80.

Figure 8 summarizes the performance of Flash-Searcher model variants under the two step limits. Across all Flash-Searcher models, extending the maximum number of steps from 40 to 80 yields consistent and measurable performance gains: Flash-Searcher-32B improves by 5.0 points, while Flash-Searcher-72B achieves a 7.0-point increase. These results confirm that the performance bottleneck observed in the 40-step setting arises from insufficient reasoning steps rather than fundamental model limitations.

These results directly verify our initial observation: increasing the number of allowed reasoning steps enables Flash-Searcher to fully unpack complex task logic, thereby improving solution accuracy. This supports the feasibility of trading computational resources for precision in resource-unconstrained deployment scenarios.

## Appendix F Flash-Searcher Model Training

### F.1 Training Dataset

Our training dataset is constructed by curating subsets of four well-established public agent-focused datasets: AFM [Li et al., 2025b]11 1 AFM Dataset: https://huggingface.co/datasets/PersonalAILab/AFM-WebAgent-SFT-Dataset, ASearcher [Gao et al., 2025]22 2 ASearcher Dataset: https://huggingface.co/datasets/inclusionAI/ASearcher-train-data, WebShaper [Tao et al., 2025]33 3 WebShaper Dataset: https://github.com/Alibaba-NLP/WebAgent/blob/main/WebShaper, and WebWalkerQA [Wu et al., 2025b]44 4 WebWalkerQA Datasethttps://huggingface.co/datasets/callanwu/WebWalkerQA. The subsets contain 1355, 628, 500, and 2597 examples, respectively. For AFM, Asearch, and WebWalkerQA-silver datasets, we applied a filtering process based on the execution trajectory length of baseline ReAct frameworks. Specifically, we selected only those examples that required more than 8 steps to complete, as these represent more complex reasoning and action sequences that better demonstrate agent capabilities.

Table 6: Composition of the training dataset after filtering and trajectory generation.

Dataset Original Size Correct Samples

AFM 1,355 1212

Asearch 628 457

WebShaper 500 405

WebWalkerQA 2,597 1767

Total 5080 3354 (Removed data with formatting issues)

We leveraged our Flash-Searcher framework (with GPT-5 as the backbone) to generate trajectories for pre-filtered examples. To ensure training data reliability—critical for effective model learning—we used a judge model (GPT-4.1-mini) to validate trajectory answer correctness, retaining only factually accurate ones. This initial filtering yielded 1212, 457, 405, and 1767 candidate trajectories from AFM, ASearcher, WebShaper, and WebWalkerQA, respectively. We further conducted systematic format inspections to exclude trajectories with structural flaws (e.g., incomplete turn segmentation, invalid dialogue hierarchy, missing action labels)—a step to reduce noise in supervised fine-tuning (SFT). After this two-stage screening (correctness + format), we ultimately retained 3354 valid trajectories for training. Table 6 summarizes the final training dataset composition, including source dataset and trajectory attribute breakdowns.

These trajectories were formatted into SFT-compatible multi-turn dialogues via the LLaMA-Factory framework [Zheng et al., 2024]. Specifically, the detailed structure of the multi-turn dialogue format (including role definitions, dialogue turn segmentation, and task-related context embedding) is illustrated in Figure 10, which standardizes the conversion of trajectory data into instruction-response pairs for SFT training.

### F.2 Parameters

We performed supervised fine-tuning (SFT) using the LLaMA-Factory framework with selected hyperparameters to optimize model performance. Table 7 presents the key parameters used during our training process. We employed a cosine learning rate schedule with warmup to stabilize the early training phase. To address memory constraints while training on the 32B/72B parameter model, we utilized gradient accumulation and parameter-efficient fine-tuning techniques. The training was conducted on 64 NVIDIA A800 GPUs (80GB each) with DeepSpeed ZeRO-3 optimization to manage memory usage efficiently.

Table 7: Training hyperparameters for supervised fine-tuning.

Parameter Value

Learning Rate 1e-5

Training Epochs 6

Gradient Accumulation Steps 2

Warmup Ratio 0.1

Gradient Accumulation Steps 2

LR Scheduler Cosine with Warmup

Context length 131072 (for 32B) / 65536 (for 72B)

### F.3 Model training curves

In this section, we provide detailed training curves for the Flash-Searcher when applied to the Qwen-2.5-32B-Instruct and Qwen-2.5-72B-Instruct models. These curves illustrate the evolution of key metrics throughout the training process, validating the stability and convergence properties of our approach.

(a) Qwen-2.5-32B.

(b) Qwen-2.5-72B.

Figure 9: Training curves for Flash-Searcher models. Both models demonstrate stable convergence without signs of overfitting.

### F.4 Example of Training Data

We present a multi-turn dialogue format exampes for SFT training, explicitly illustrating the three core components of each dialogue unit: system prompt (task constraints), user instruction (task-specific requirement), and agent response (standardized output).

Figure 10: Example of the multi-turn dialogue format for SFT training. Each dialogue unit consists of three core components: (1) System prompt (task constraints), (2) User instruction (task-specific requirement), and (3) Agent response (standardized output).

## Appendix G Other Framework Setups

For our comparative analysis, we employ two state-of-the-art agent frameworks: OAgents [Zhu et al., 2025a] and OWL-Roleplaying [Hu et al., 2025]. We maintain their original configurations to ensure fair comparison with our approach.

For OAgents, both the Code-Agent and Search-Agent components utilize GPT-4.1 as their backbone model. Similarly, OWL-Roleplaying is implemented with two backbone variants: GPT-4.1 and OpenAI-o3. All other parameters, prompting strategies, and execution workflows for both frameworks are kept identical to their original implementations. Our experiments are conducted using the official repositories 55 5 OAgents: https://github.com/OPPO-PersonalAI/OAgents; OWL: https://github.com/camel-ai/owl of these frameworks to ensure reproducibility and consistency with published results.

## Appendix H Prompts

### H.1 Llm-as-judge Prompt

### H.2 Flash-Searcher Framework

#### H.2.1 System Prompt

#### H.2.2 DAG Plan Prompt

#### H.2.3 Summary Prompt

#### H.2.4 Execution Prompt

### H.3 Flash-Searcher Model

#### H.3.1 Training and Inference Prompts

## Appendix I Case Study

Experimental support, please view the build logs for errors. Generated by L A T E xml .

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

- Click the "Report Issue" () button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

We gratefully acknowledge support from our major funders, member institutions, , and all contributors. Major funding support from
