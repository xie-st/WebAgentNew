# AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management

- arXiv: 2512.10371
- URL: https://arxiv.org/html/2512.10371
- 来源：arXiv HTML（LaTeXML）自动转换，公式以 $...$ 内联保留，图表编号可能与原文有出入。

---

AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context Management Skip to main content

##### Report GitHub Issue

Title:

Content selection saved. Describe the issue below:

Description: Why HTML? Back to Abstract Download PDF License: arXiv.org perpetual non-exclusive license arXiv:2512.10371v2 [cs.AI] 08 May 2026

# AgentProg: Empowering Long-Horizon GUI Agents with Program-Guided Context ManagementThanks: Corresponding Author: Yuanchun Li (liyuanchun@air.tsinghua.edu.cn)

#1004Conference: The 24th Annual International Conference on Mobile Systems, Applications and Services; June 21–25, 2026; Cambridge, United KingdomThe 24th Annual International Conference on Mobile Systems, Applications and Services (MobiSys ’26), June 21–25, 2026, Cambridge, United KingdomDOI: 10.1145/3745756.3809245ISBN: 979-8-4007-2027-7/26/06CCS: Human-centered computing Ubiquitous and mobile computingCCS: Computing methodologies Artificial intelligence Shizuo Tian Affiliation: Tsinghua University, Beijing, China , Hao Wen Affiliation: Tsinghua University, Beijing, China , Yuxuan Chen Affiliation: Tsinghua University, Beijing, China , Jiacheng Liu Affiliation: Peking University, Beijing, China , Shanhui Zhao Affiliation: Tsinghua University, Beijing, China , Guohong Liu Affiliation: Tsinghua University, Beijing, China , Ju Ren Affiliation: Tsinghua University, Beijing, China , Yunxin Liu Affiliation: Tsinghua University, Beijing, China and Yuanchun Li Affiliation: Tsinghua University, Beijing, China © cc

###### Abstract.

The rapid development of mobile GUI agents has stimulated growing research interest in long-horizon task automation. However, building agents for these tasks faces a critical bottleneck: the reliance on ever-expanding interaction history incurs substantial context overhead. Existing context management and compression techniques often fail to preserve vital semantic information, leading to degraded task performance. We propose AgentProg, a program-guided approach for agent context management that reframes the interaction history as a program with variables and control flow. By organizing information according to the structure of program, this structure provides a principled mechanism to determine which information should be retained and which can be discarded. We further integrate a global belief state mechanism inspired by Belief MDP framework to handle partial observability and adapt to unexpected environmental changes. Experiments on AndroidWorld and our extended long-horizon task suite demonstrate that AgentProg has achieved state-of-the-art success rates on these benchmarks. More importantly, it maintains robust performance on long-horizon tasks while baseline methods experience catastrophic degradation. Our system is open-sourced at https://github.com/MobileLLM/AgentProg.

###### Keywords:

Mobile GUI Agents, Mobile Task Automation, Long-Horizon Agents, Context Management ††cc-license: by

## 1. Introduction

Building intelligent assistants that can operate mobile devices by following natural language instructions has long been a central goal of mobile systems (Azim et al., 2016; Li and Riva, 2018; Wen et al., 2024) and artificial intelligence researchers (Sun et al., 2022; Li et al., 2020; Liu et al., 2018). Graphical User Interface (GUI) agents aim to simulate human user operations on graphical interfaces (e.g., clicking, scrolling, and inputting text) to interact with devices and complete complex tasks within existing software ecosystems. Mobile GUI environments present unique challenges compared to web or desktop settings: (1) severe partial observability due to the constrained screen size of mobile devices as only a small portion of UI elements is visible, and critical device states are usually hidden from the screen, making the observability problem more significant than on web or desktop platforms; (2) noisy accessibility trees as many mobile applications provide incomplete, misleading or even no accessibility tree support, while a complete DOM tree is often readily accessible in web environments, making symbolic program approaches that depend on structured elements unreliable; (3) fragmented UI ecosystems, where the system share panels, permission dialogs, or even interaction patterns are different across different OS versions and device manufacturers, challenging the generalization ability of agents. Recent advances in large language models (LLMs) and vision-language models (VLMs) have shown promising results in this domain. Building upon these foundation models, researchers have developed rich agent frameworks that enable agents to successfully complete relatively simple tasks through interactive decision-making processes (Zhang and Zhang, 2024; Zhang et al., 2023; Wen et al., 2025; Xu et al., 2025; Ye et al., 2025b; Gonzalez-Pumariega et al., 2025; Li et al., 2025b).

However, despite their success on simple tasks, existing methods face significant challenges when scaling to long-horizon scenarios—tasks that require dozens or even hundreds of steps to complete. The ability to complete such long tasks is usually more desirable, as they are often cumbersome and difficult for human users. For instance, an example long task is to review all events on today’s calendar (including sending messages, creating notes, etc.) and generate follow-up steps on all scheduled items, or to navigate between multiple shopping and note apps to compare and record product prices. These challenges manifest in two critical dimensions. First, model performance degrades significantly on extended tasks, as language models struggle to maintain coherence and task-relevant reasoning over protracted sequences (Liu et al., 2024; An et al., 2024). Second, the computational cost grows substantially as the interaction history increases, with each decision requiring to process an increasingly long context.

Existing approaches attempt to address the context management problem through various techniques, such as context window (Qin et al., 2025), summarization (Rawles et al., 2024; Ye et al., 2025b) or hierarchical planning (Agashe et al., 2025; Li et al., 2025a). While these methods partially alleviate the computational burden, they face a fundamental limitation: they lack a principled framework for determining which information is essential for future steps and which can be safely discarded. This often results in the loss of crucial state information, leading to task failure (Zhang et al., 2025; Schroeder et al., 2025; Zhou et al., 2025). Moreover, existing methods demonstrate a critical gap in generalization—an agent’s ability to solve individual atomic tasks does not directly transfer to solving compositional or iterative long-horizon tasks, revealing deeper limitations in their reasoning capabilities (Liu et al., 2025). The core challenge in long-horizon task execution lies in effective context management during task execution. An agent must maintain a comprehensive understanding of the task progress and dynamically assess the evolving environment to make informed decisions. This requires task-aware context management to preserve the essential state while discarding irrelevant details.

Based on a pilot test of mobile GUI agents on long-horizon tasks (detailed in Section 2.3), we observed three key difficulties that hinder more accurate and robust task execution.

- (1)

Task Planning and Decomposition. Long-horizon tasks often involve composition, repetition, and recursion of multiple subtasks. Existing methods typically decompose tasks into sequences or trees (Wei et al., 2022; Yao et al., 2023), but these structures are overly simplistic for complex scenarios, necessitating frequent re-planning that introduces overhead and instability. For instance, tasks like “read today’s todo list and complete each item” cannot be fully planned upfront since the items are unknown initially, leading to repeated re-planning that increases error risks.

- (2)

Interaction History Management. Due to the limited context length of LLMs, agents must determine what information to retain or forget in the task execution process. Current approaches lack task-aware context management, struggling to preserve critical intermediate results (e.g., restaurant details across a reservation task) while discarding irrelevant navigation history.

- (3)

Environment State Comprehension. Agents must maintain an accurate understanding about environment states throughout execution. Software environments are usually dynamic and partially-observable with hidden UI elements, unexpected state changes, and transient failures, which are hard to handle for agents in the long task process. For instance, without proper environment understanding, agents may incorrectly believe tasks are complete when operations actually failed, causing workflow derailment that becomes critical in long-horizon scenarios.

Our inspiration to address these challenges stems from traditional programs, which are capable of executing extremely long tasks with limited memory capacity. A program is naturally a representation of task decomposition. Its “context management” strategy is the program itself, which seamlessly handles information retaining/discarding in the system heap and stack through the mechanisms of functions, threads, global/local variables, etc. In principle, the task execution process of a GUI agent can also be represented as a program, and the agent context can be maintained in the same way as the CPU memory during program execution.

Therefore, we propose AgentProg, a program-guided context management approach for long-horizon GUI agent tasks. AgentProg reframes the agent’s interaction history—traditionally represented as a flat, ever-expanding sequence — as a program with variables and control flow. This program structure, generated based on the task requirements, provides a principled and task-aware mechanism for context management. By organizing information according to the program’s structure, AgentProg can systematically determine which information to retain and which to discard at each step, enabling task-specific context management that preserves crucial details.

A key question in our agent-program analogy is how to handle environment dynamicity. Letting the agent follow a program in the task process may lead to poor adaptivity, since the programs are generated before observing the actual environment at each step. We introduce a domain-specific language named Semantic Task Program (STP) to handle this problem. Unlike traditional programs that are based on precise symbolic instructions (e.g., name = getElement(’user Name’).text), STP uses fuzzy, natural-language-style instructions (e.g., get user name as {name}) to construct the program. The STP instructions capture necessary information for interaction context management, while they are interpreted adaptively at runtime based on the actual environment state. Furthermore, we incorporate a Global Belief State module into the STP execution process to address the inherent partial observability of GUI environments, a characteristic absent in traditional programming paradigms.

We evaluate AgentProg on AndroidWorld (Rawles et al., 2024), a challenging and widely-used benchmark for mobile device-control agents, and introduce an extended task suite that we construct based on AndroidWorld to specifically test agent performance on long-horizon scenarios. Our experiments reveal that while existing methods perform well on the original AndroidWorld benchmark, they exhibit substantial performance degradation on our extended task suite, highlighting their fundamental limitations in handling long-horizon reasoning. In contrast, our program-guided approach demonstrates robust performance across both benchmarks. The context organization ability of AgentProg also helps it achieve the new state of the art on AndroidWorld.

Our main contributions are summarized as follows:

- (1)

We identify three key challenges for GUI agents on long-horizon tasks, and propose program-guided context management as a principled approach to address these challenges.

- (2)

We design and implement AgentProg, a mobile GUI agent based on program-guided context management. We introduce Semantic Task Program and global belief state to address the dynamicity and partial observability problems in program-driven task execution.

- (3)

AgentProg has been evaluated on AndroidWorld and AW-Extend and achieved the new state of the art.

## 2. Related Work and Motivation

### 2.1. GUI Agent

Graphical User Interface (GUI) agents are autonomous systems designed to interact with software applications through their visual interfaces, mimicking human user behavior to accomplish complex tasks. A GUI agent typically takes as input a natural language task instruction (e.g., “Send an email to John about the meeting”) and the current screen state, which includes visual information (screenshots) and optionally structural information (UI element hierarchies such as accessibility trees or HTML DOMs) (Nguyen et al., 2025). The agent outputs a sequence of actions such as clicking on specific UI elements, scrolling, typing text, or navigating between applications to accomplish the specified task within existing software ecosystems.

The decision-making process of a GUI agent can be formalized as a partially observable Markov decision process (POMDP) (Sondik, 1978). At each timestep $t$ , the agent observes the current screen state $o_{t}$ and selects an action $a_{t}$ based on the task goal $g$ and the interaction history $h_{<t}=\{(o_{0},a_{0}),\allowbreak(o_{1},a_{1}),\dots,(o_{t-1},a_{t-1})\}$ . The environment then transitions to a new state, providing observation $o_{t+1}$ . This process continues until the task is completed or a maximum number of steps is reached. The core challenge lies in how agents manage the ever-growing interaction history $h_{<t}$ while maintaining accurate understanding of the current environment state.

### 2.2. Related Work

GUI Agents with Foundation Models. Recent advances in large language models and multimodal large language models have significantly enhanced their capabilities in language understanding and cognitive processing, enabling effective interpretation of human instructions, detailed planning, and complex task execution (Xu et al., 2024a). Early approaches combined LLMs with domain-specific knowledge to achieve smartphone automation (Wen et al., 2024; Zhang et al., 2023), while subsequent methods introduced refined control mechanisms for mobile applications through visual perception and multimodal reasoning (Yan et al., 2023; Zhang et al., 2023; Zhang et al., 2024). Vision-centric approaches have been proposed that identify and locate UI elements directly from visual inputs without relying on XML metadata or system accessibility trees (Cheng et al., 2024; Zheng et al., 2024; Gou et al., 2025; Wu et al., 2024b; Wang et al., 2024c; Li et al., 2025a; Li et al., 2025b; Wang et al., 2025b).

Context Management for GUI Agents. Managing interaction history is crucial for GUI agents to maintain task progress within token budget constraints. Existing approaches include sliding window methods that retain only the most recent $N$ observations (Qin et al., 2025), summarization methods that condense each step into minimal tokens (Rawles et al., 2024; Gao et al., 2025; Tan et al., 2024), and hierarchical planning methods that decompose tasks into sub-task sequences (Li et al., 2025b; Ye et al., 2025b; Gonzalez-Pumariega et al., 2025; Zhang et al., 2024; Mozannar et al., 2025). However, these methods face limitations: sliding window incurs increasing context, summarization lacks explicit planning for complex tasks and may lose critical early information, and hierarchical planning introduces overhead requiring multiple re-planning rounds and also faces with the challenge of forgetting important information.

Program-Based GUI Agents. While standard GUI agents utilize a fixed set of programmatic APIs (e.g., click, type, scroll) as their action space, they typically treat these low-level APIs merely as atomic tool invocations. Recently, a distinct paradigm shift in LLM agents has emerged from using rigid, discrete tool calls to employing executable code as the unified action space for agents (Wang et al., 2024a; Shen et al., 2025; Li et al., 2025c; Liang et al., 2023). To bridge the gap between low-level actions and high-level code generation, recent GUI agents have empowered agents to proactively construct high-level functions, encapsulating primitive operations into meaningful skills (Wang et al., 2025a; Wu et al., 2024a). Other approaches establish a persistent program context or terminal environment, enabling the agent to engage in continuous interaction through script synthesizing (Wen et al., 2025; Song et al., 2025a; Gur et al., 2024). However, current task script generation methods (Wen et al., 2025) typically require precise API calls and strict syntax adherence, relying heavily on extensive exploration of mobile apps to build application documentation. Consequently, these scripts are prone to failure when encountering unseen applications or when dealing with out-of-distribution scenarios where documentation is unavailable.

### 2.3. Motivation and Challenges

Figure 1. Performance Comparison on AndroidWorld vs. AW-Extend. a11y refers to the Accessibility Tree observation space; SoM denotes Set-of-Mark; Mobile-Ag-v3 denotes Mobile-Agent-v3.

Figure 2. Failure mode in existing methods (Mobile-Agent-v3 (Ye et al., 2025b), UI-TARS (Qin et al., 2025), M3A (Rawles et al., 2024)) on AW-Extend.

Despite the progress in GUI agents, existing methods face fundamental limitations when handling long-horizon tasks, which require dozens or hundreds of sequential steps to complete. To better understand these challenges, we construct AW-Extend, an extended task suite based on AndroidWorld (Rawles et al., 2024) that specifically tests agent performance on compositional and iterative tasks. Our extended suite comprises 19 tasks that are possibly common in realistic scenarios across two categories: Compositional Tasks that combine sub-tasks into a compositional task requiring critical information maintenance across applications, and Iterative Tasks that scale sub-task counts to $n=10$ or $n=20$ to test robustness and memory filtering capabilities. The detail of this task suite is illustrated in Section 4.

Our preliminary analysis reveals critical failure reasons in existing methods, as illustrated in Figure 1:

Lack of Overall Planning. In the “MarkorTodoList” task, Mobile-Agent-v3 prematurely terminates after completing only the first two sub-tasks, mistakenly believing all three items from the note have been addressed, as shown in Figure 2. This indicates a fundamental deficiency in holistic task planning and progress tracking.

Failure to Remember Critical Information. Despite employing context management schemes, existing agents fail to preserve task-relevant information at critical moments. In our tests, agents consistently pass incorrect parameters during sub-task execution because they do not retain all task items when initially opening the note, as shown in Figure 2. Their context management mechanisms lack sensitivity to task-critical information.

Poor Understanding of Dynamic Environments. Our iterative tasks involving 10-20 sub-tasks expose fragility in handling unexpected situations—accidental app closures, dialog popups, or wrong operations. Moreover, due to the partial observability of GUI environments, changes in phone data are sometimes implicit, such as system settings, date and time, and in-app data. Existing methods struggle with such environmental variations, resulting in poor robustness during extended task execution. In our tests, the agent was tasked with deleting all events on this Saturday, but failed to correctly understand the date in the phone environment. Consequently, it deleted all Saturday events across different weeks, leading to task failure.

## 3. Our Approach: AgentProg

We present AgentProg, a program-guided framework for long-horizon task execution, as shown in Figure 3. It addresses three fundamental challenges in agent-environment interaction. First, to tackle the challenge of planning and decomposing complex long-horizon tasks, we introduce Semantic Task Program (STP), a domain-specific language that extends natural language instructions with structured control flow (loops, branches, functions) and explicit variable operations. Second, to provide agents with appropriate context without redundancy, we develop a program-guided context management approach that operates along control flow-based pruning and data flow management. Third, to maintain accurate environment understanding despite partial observability and outdated beliefs, we propose a Global Belief State mechanism that continuously validates and updates the agent’s hypotheses about environment mechanisms and hidden variables throughout execution.

Together, these components form a two-stage framework: Semantic Task Program Generation, where the agent performs global planning to create a structured program with identified critical variables, followed by Semantic Task Program Execution, where the agent incrementally interprets the program while maintaining belief state consistency and managing context efficiently. This design enables stable, efficient execution of long-horizon tasks without the context overflow and information loss problems that plague conventional approaches.

Figure 3. The workflow of AgentProg.

### 3.1. Semantic Task Program: A DSL for Long-Horizon Task Planning

We design Semantic Task Program (STP), a domain-specific language (DSL) that represents long-horizon tasks. Unlike rigid programming languages, STP balances expressiveness with semantic tolerance. It enables agents to plan and execute complex tasks effectively. More details and examples of STP are listed in Section A.

Design Principles. STP is built on three core principles: (1) Natural Language Foundation: Each program step is primarily expressed in natural language instructions, ensuring readability and maintaining generality across different environments; (2) Structured Control Flow: The language incorporates essential programming constructs including loops, conditionals, and functions to express iterative and recursive task patterns concisely; (3) Explicit Data Flow: Variables are used to explicitly pass information between steps, allowing the agent to identify and preserve critical information throughout task execution.

Syntax and Constructs. An STP consists of a sequence of steps, where each step can be: (1) Action Step: A natural language instruction describing what should be accomplished, such as “Open the note ‘todo_list.md’ in the Markor application”; (2) Variable Declaration and Assignment: Explicit recording of critical information using variables enclosed in braces, e.g., “record them as {task_list}”; (3) Control Flow Statements: Loop constructs, conditional branches, and function definitions that organize execution flow, e.g., “Iterate over {task_list}”; (4) Variable References: Using declared variables in subsequent steps, e.g., “for each {task_item}, …”.

Semantic Tolerance. A key distinguishing feature of STP is its semantic tolerance. Current task script generation methods (Wen et al., 2025) require precise API calls and strict syntax. Consequently, they are highly sensitive to environment dynamics and software updates. Semantic Task Program allows for variations in execution details. The natural language instructions provide high-level guidance without being tightly coupled to specific environment implementations. This tolerance enables the agent to adapt to different environments and handle unexpected situations during execution, while the structured control flow ensures the overall task logic remains intact.

Comparison to Symbolic DSLs. Symbolic DSLs (e.g., AutoDroid-V2) often suffer from syntax errors and require extensive app-specific knowledge (Wen et al., 2025). When mobile applications provide poor or even no accessibility tree support, those methods are prone to failure. This critical limitation on mobile platform motivates our natural-language-style design, where STP leverages LLM’s strong semantic priors, ensuring robust generalization via visual observations without rigid syntax constraints or limitations of accessibility tree.

### 3.2. STP Generation and Execution

AgentProg aims to enhance agent performance in long-horizon tasks through a two-stage framework for memory management as Figure 3 shows: 1) Semantic Task Program Generation: The agent generates a globally planned Semantic Task Program based on task requirements, where the program extends natural language with support for control flow constructs (e.g., branching, loops) and variable operations (declaration, assignment); 2) Semantic Task Program Execution: The agent incrementally interprets and executes the Semantic Task Program, taking environment state and current program state as input, and producing actions while updating the program state.

Generation Phase. The agent plans a feasible program according to specifications. It uses control flow to orchestrate execution and identifies critical variables for preservation. The natural language components ensure the program remains general and robust.

Execution Phase. Instead of rigidly executing pre-defined scripts, AgentProg adopts a dynamic instruction grounding paradigm. The Semantic Task Program serves as a high-level guide, where AgentProg acts as a runtime interpreter that translates the current high-level program instruction into executable low-level actions based on the real-time environment state. This mechanism decouples the task logic from specific execution details, preventing context overflow while maintaining flexibility.

Specifically, AgentProg operates in two alternating modes:

1) Action Generation. In this mode, the agent focuses on instantiating the high-level intent of the current program step into concrete execution logic. Based on the instruction pointed to by the Program Counter (PC) and the current GUI observation, the agent generates a snippet of low-level Python code to interact with the environment. AgentProg utilizes LLM to ground abstract natural language instructions into a sequence of atomic Python APIs (e.g., start_app(<app>), click(<element>), swipe_down(), input

(<text>)). Crucially, this allows the agent to dynamically adapt the low-level operations (e.g., handling pop-ups or layout changes) while adhering to the high-level plan. The agent records these executed Python scripts to track data flow, while repetitive execution histories (e.g., inside loops) are folded to save context.

2) Program Counter Update. After the generated Python code is executed and the environment updates, the agent switches to this mode to manage the control flow. It decides how to move the Program Counter—whether to advance to the next step, iterate a loop, or jump to a branch—based on the execution result of the previous step. The Program Counter is dynamically determined by the LLM rather than following a deterministic transition, enabling the agent to skip, repeat or branch based on runtime observations.

These two modes alternate strictly: AgentProg translates the current instruction into Python code (Action Generation), executes it, and then decides where to go next in the program (PC Update). Throughout this process, AgentProg maintains a structured context containing the static program plan and the dynamic variables and low-level history, ensuring all decisions are globally consistent yet locally adaptive.

### 3.3. Program-Guided Context Management

During the Semantic Task Program Execution phase, AgentProg adopts a program-guided approach to manage the context input for the agent. This mechanism addresses the trade-off between information retention and context length. By leveraging the explicit structure of the STP, AgentProg acts as a filter: it retains execution-critical information while systematically discarding redundant history. This management operates along two dimensions: control flow pruning and data flow persistence.

Figure 4. Program-guided context management with context pruning, history retrieval and variable management.

#### 3.3.1. Execution Tree-Guided Context Pruning.

To manage the control flow context, we organize the STP execution history into a dynamic tree structure. As illustrated in Figure 4, each executed step expands a corresponding node in this tree.

Tree Construction Rules. The node expansion logic is strictly bound to the program structure. We categorize nodes into three types:

- •

Sequential Nodes: If a step follows a linear sequence, it becomes a direct child of the previously executed node.

- •

Conditional Nodes: For branching statements (e.g., if-else), only the node corresponding to the actually executed branch becomes a child of the condition node. Unexecuted branches are never added.

- •

Loop Nodes: For loop structures, each iteration expands a new loop statement node. These are added as children of the loop entry point, preserving the iteration history hierarchically.

Path-Based Pruning. During inference, the agent does not see the entire raw history. Instead, we compute the active path from the root node to the currently executing node. Only the step information and observations along this active path are retained in the context; all other branches and completed loop iterations that are no longer relevant to the current state are discarded.

This approach significantly reduces context overhead, especially in tasks involving complex logic. It prevents the agent from being confused by redundant information (e.g., stale states from a previous loop iteration or irrelevant actions from a non-taken branch), allowing it to focus solely on the current logical path.

#### 3.3.2. Program-based Historical Step Retrieval

While pruning removes irrelevant context, AgentProg also employs a retrieval mechanism to enhance stability for iterative tasks. When the execution revisits a specific program step—a common scenario in loops or recursive segments—AgentProg activates a retrieval mechanism. Unlike generic similarity search, this mechanism uses the unique program step ID to precisely identify preceding executions of the exact same step. The operational outcomes of these past iterations—including actions taken and resulting state changes—are selectively injected into the current context as references. This allows the agent to “remember” its own recent success or failure patterns. By providing this structured historical awareness, AgentProg significantly improves consistency in long-horizon repetitive tasks, mitigating the risks of oscillation or repeated errors that often plague agents lacking step-specific memory.

#### 3.3.3. Data Flow and Variable Management

Beyond control flow, AgentProg ensures the persistence of critical information through explicit variable management. This addresses the challenge where purely context-based agents often lose track of key data (e.g., a phone number retrieved 50 steps ago). During the Semantic Task Program Generation phase, the agent explicitly identifies and declares essential variables. These variables—such as extracted todo items, reservation times, or user preferences—act as “anchors”. They are mandated to persist throughout the execution, ensuring that task-critical data is never inadvertently discarded by the pruning mechanism.

During the Execution phase, the agent retains the flexibility to define auxiliary variables via Python code. This programmatic manipulation compensates for the limitations of raw LLM reasoning. For instance, an agent can extract multiple items into a list variable, iterate through them systematically, and use counters to track progress. This combination of declarative planning and imperative execution ensures robust information flow: critical data remains accessible, while the internal state adapts dynamically to evolving task requirements.

#### 3.3.4. Advantages of Program-Guided Context Management.

AgentProg binds context directly to the program’s execution structure, effectively addressing these limitations. From a control flow perspective, the STP captures repetition and branching patterns upfront. This allows the agent to systematically prune irrelevant branches and loop iterations without the need for error-prone re-planning cycles. From a data flow perspective, explicit variable declaration serves as an information anchor, ensuring that critical data (e.g., user preferences or extracted items) persists throughout execution regardless of context length. This dual mechanism enables AgentProg to maintain a compact, noise-free context while guaranteeing the persistence of task-critical information.

### 3.4. Global Belief State

Figure 5. Dynamic global belief state management in partially observable environments.

While the Action Generation phase produces executable Python scripts, ensuring their faithful execution requires active monitoring. The Global Belief State integrates directly into this low-level execution loop, acting as a runtime verifier that checks if the assumptions made during code generation remain valid as each atomic command executes. This validation is critical because mobile GUI environments are inherently partially observable. Critical system states—such as clipboard content, navigation back-stack, or off-screen selected items, etc.—are often invisible in the current screenshot but decisive for future actions.

Inspired by the Belief MDP framework (Kaelbling et al., 1998), we introduce a Global Belief State mechanism to bridge this gap. It operates as a runtime monitor through a continuous Predict-Verify-Align cycle: 1) Maintain Hypothesis (Predict): Unlike passive memory, the agent maintains an active set of hypotheses about the current environment status, including the UI context and data validity. 2) Runtime Verification (Verify): During the execution of low-level Python codes, AgentProg continuously compares the real-time visual observation against its belief state. 3) Anomaly Correction (Align): If the observation contradicts the belief, AgentProg identifies a Belief-Reality Gap. It immediately marks the previous state as invalid (e.g., “Context Lost”), and switches from execution mode to a recovery routine (e.g., navigating back to the target app) to realign its belief with reality.

Case Study. Consider a scenario where an agent is filling out a contact form. Initial Belief: “Editing contact ‘John Doe’; Name and Email fields are filled.” Event: Due to an accidental touch or crash, the app exits to the home screen. Without Belief State: A traditional agent might hallucinate that the step succeeded and attempt to click a "Save" button that no longer exists, leading to a cascade of errors. With Belief State: AgentProg detects the conflict: the visual observation (Home Screen) refutes the belief hypothesis (Editing View). It immediately invalidates the "Name/Email filled" status, updates the belief to "Form Context Lost," and generates a corrective plan to restart the app and re-fill the information. This mechanism ensures that the agent never proceeds on false premises.

## 4. AW-Extend Benchmark

To systematically evaluate agent performance on the long-horizon challenges identified above, we constructed AW-Extend, an extended task suite that builds upon the original AndroidWorld benchmark. By adapting the existing AndroidWorld evaluation and verification system, we ensure our extended tasks maintain the same rigorous automated assessment standards while specifically targeting long-horizon scenarios.

Existing mobile GUI benchmarks (Rawles et al., 2024; Xu et al., 2024b; Wang et al., 2024b) primarily focus on evaluating isolated functionalities within individual applications, lacking coverage of realistic complex scenarios that involve coordinating multiple applications, integrating diverse functionalities across apps, or executing extended sequences of operations within a single application. Recently, several long-horizon benchmarks have emerged across different platforms, including ColorBench (Song et al., 2025b) for mobile, VeriWeb (Liu et al., 2026) and RealWebAssist (Ye et al., 2025a) for web. However, these long-horizon benchmarks are either offline benchmarks, which scale easily but cannot capture real execution dynamics (e.g., unexpected dialogs, environment state changes) and introduce offline-to-online gap, or target non-mobile platforms where partial observability and accessibility limitations are less severe.

AW-Extend is an online and long-horizon benchmark for mobile agent evaluation. It runs on live Android emulators with handcrafted environment setup and automated state-based evaluation scripts to ensure realistic environment dynamics. AW-Extend tasks require over 30 steps on average, compared to 13.0 for ColorBench and 17.6 for RealWebAssist, making them substantially more challenging for context management. AW-Extend utilizes realistic task composition and batching (e.g.,"add multiple contacts and message each one"), consistent with methodologies in ColorBench (Song et al., 2025b) and VeriWeb (Liu et al., 2026). Our extended suite comprises 19 tasks covering two categories:

Compositional Tasks combine multiple atomic operations into realistic workflows that require maintaining task context across different applications. These tasks are characterized by strong dependencies between subtasks, demanding that agents retain and effectively utilize critical information throughout the execution process. Representative examples include: processing a To-Do List from notes by completing each item sequentially (creating calendar events, adding contacts, managing expenses); batch operations such as creating multiple contacts and messaging each one; and cross-application interactions like querying note content via text message and replying accordingly.

Iterative Tasks systematically scale the sub-task count from the standard $n=3$ in AndroidWorld to $n=10$ and $n=20$ , testing both agent robustness and the ability to filter irrelevant information from context. Unlike compositional tasks where subtasks are interdependent, iterative tasks involve weakly-correlated operations, making them particularly effective for evaluating an agent’s capacity to suppress interference from irrelevant historical context while maintaining consistent task execution. These tasks encompass managing expense records, calendar events, menu items, and note operations.

We adapted AndroidWorld’s environment configuration and evaluation code to create these tasks, enabling assessment through AndroidWorld’s standard interface, including environment setup, automated evaluation based on emulator state, and trajectory recording.

## 5. Experiments

### 5.1. Experimental Settings

Benchmarks. We conduct experiments on AndroidWorld (Rawles et al., 2024) and AW-Extend. AndroidWorld is a comprehensive and widely used benchmark for evaluating Android GUI agents across diverse real-world mobile applications with 116 tasks. AW-Extend is an extension we construct with 19 additional long horizon tasks described in Section 4.

Baselines. We compare our method with several state-of-the-art approaches on two benchmarks. On AndroidWorld benchmark, as shown in Table 1, we include a wide range of methods from related work. We re-evaluated M3A method (Rawles et al., 2024), for both its a11y tree and SoM variants, using Gemini-2.5-Pro, as the original work reported results based on GPT-4-Turbo. All other baseline scores on AndroidWorld are cited from their respective original publications. On AW-Extend benchmark, we select a focused set of baselines, each employing a distinct context management strategy to handle long-horizon tasks. As detailed in Table 2, these include:

- •

M3A (Rawles et al., 2024), which generates a textual summary after each step to manage its context.

- •

UI-TARS (Qin et al., 2025), which uses a sliding window approach, discarding the oldest screenshots in a first-in-first-out (FIFO) manner when the context exceeds a limit. We follow its official setting, retaining a maximum of five screenshots.

- •

Mobile-Agent-v3 (Ye et al., 2025b), which employs hierarchical planning method. A planner first decomposes the main task into a list of sub-tasks, and an executor then completes them sequentially. The planner maintains the list of sub-tasks continuously during the task execution.

For baselines on AW-Extend benchmark, we use Gemini-2.5-Pro (Google, 2025) as the backbone model for M3A (Rawles et al., 2024), UI-TARS-1.5-API for UI-TARS (Qin et al., 2025), and both GUI-Owl-32B and GUI-Owl-7B for Mobile-Agent-v3 (Ye et al., 2025b). We note that Mobile-Agent-v3 is designed around a specialized model (GUI-Owl) with built-in grounding capabilities, making it incompatible with direct backbone substitution to general-purpose LLMs. We evaluate it using its reported best settings to serve as a representative baseline for hierarchical planning methods.

Metrics. We adapt the official AndroidWorld evaluation metric (Success Rate) to measure task completion performance. For AW-Extend, we adapt AndroidWorld’s evaluation codebase to support assessment of our custom task set.

Implementation Details. We implement AgentProg using Gemini-2.5-Pro (Google, 2025) as the backbone planning model, and leverage UI-TARS-1.5-API (Qin et al., 2025) to locate the UI elements.

### 5.2. Analysis of Success Rate

Table 1. Success Rate (%) on AndroidWorld.

Method AndroidWorld (%)

MobileGPT (Lee et al., 2024) 23.0

AutoDroid-V2 (Wen et al., 2025) 26.0

M3A (a11y, GPT-4-Turbo) (Rawles et al., 2024) 30.6

M3A (a11y, Gemini-2.5-Pro) (Rawles et al., 2024) 31.0

M3A (SoM, GPT-4-Turbo) (Rawles et al., 2024) 25.4

M3A (SoM, Gemini-2.5-Pro) (Rawles et al., 2024) 39.7

GLM-4.1V-9B-Thinking (Hong et al., 2025) 41.7

UI-TARS (UI-TARS-7B) (Qin et al., 2025) 33.0

UI-TARS (UI-TARS-1.5-API) (Qin et al., 2025) 64.2

V-Droid (Dai et al., 2025) 59.5

MobileUse (Li et al., 2025b) 62.9

UI-Venus (Gu et al., 2025) 65.9

Agent S3 (Gonzalez-Pumariega et al., 2025) 68.1

Mobile-Agent-v3 (GUI-Owl-7B) (Ye et al., 2025b) 66.4

Mobile-Agent-v3 (GUI-Owl-32B) (Ye et al., 2025b) 73.3

AgentProg 78.0

Table 2. Success Rate (%) on AW-Extend.

Method AW-Extend (%)

M3A (a11y) 23.7

M3A (SoM) 28.9

Mobile-Agent-v3 (GUI-Owl-7B) 26.3

Mobile-Agent-v3 (GUI-Owl-32B) 28.9

UI-TARS (UI-TARS-1.5-API) 36.8

AgentProg 68.4

Overall Results. As shown in Table 1, our method achieves a success rate of 78.0% on the AndroidWorld benchmark, surpassing the previous state-of-the-art Mobile-Agent-v3 by 4.7%. This demonstrates significant advantages in general mobile automation tasks. On AW-Extend benchmark shown in Table 2, we compare our program-guided context management approach against three representative baseline context management strategies: Summarization (M3A), sliding window (UI-TARS), and hierarchical planning (Mobile-Agent-v3). Our method achieves a success rate of 68.4%, substantially outperforming all baseline approaches. These results demonstrate that our program-guided context management framework offers substantial advantages over existing methods, particularly for long-horizon tasks that require maintaining context effectively over extended interaction sequences.

Figure 6. Success Rate (%) across difficulty levels on AndroidWorld.

Performance Across Task Difficulty Levels. Figure 6 presents the success rates of different methods stratified by task difficulty levels on AndroidWorld. In Figure 6, M3A uses SoM as input and Gemini-2.5-Pro as backbone model. MobileUse results are reported from the original paper, while M3A and AgentProg results are from our evaluation. Other existing methods have not reported performance across different difficulty levels on AndroidWorld. We observe that success rates decline progressively as task difficulty increases across all methods. Notably, AgentProg consistently outperforms baselines at every difficulty level. Besides, the magnitude of improvement is most pronounced for medium tasks (25.0% over MobileUse), followed by hard tasks (13.1% improvement), with the smallest gain on easy tasks (9.8% improvement). This pattern suggests that existing methods can handle simple tasks reasonably well, but face substantial challenges with medium and hard tasks. AgentProg effectively addresses this gap, particularly excelling where current methods struggle most. This demonstrates that the program-driven framework with belief state tracking in AgentProg is particularly effective for complex tasks that demand advanced planning capabilities, robust memory of critical information, and sophisticated environment understanding. The widening performance gap at higher difficulty levels validates that these capabilities become increasingly critical as task complexity grows.

Figure 7. Success Rate (%) across difficulty task types on AW-Extend.

Performance Across Different Task Types. AW-Extend benchmark comprises two distinct task categories: iterative tasks and compositional tasks. We analyze the performance of different methods across these categories, as illustrated in Figure 7. Compositional tasks combine multiple related subtasks, requiring agents to retain information from one subtask while executing another, thereby testing their ability to preserve critical memory. In contrast, iterative tasks involve executing similar operations multiple times with weak inter-task dependencies, primarily challenging the capability of agents to filter out irrelevant memory interference while maintaining robust task completion. Among only baseline methods, UI-TARS achieves the best overall performance, primarily due to its strong results on compositional tasks. This advantage stems from its sliding window approach, which avoids summarizing historical information, allowing the model to access information from previous subtasks when executing the current one. However, both UI-TARS and Mobile-Agent-v3 underperform M3A on iterative tasks. For UI-TARS, this may be attributed to the sliding window accumulating substantial irrelevant memory in the context, which interferes with model performance. The more complex architecture of Mobile-Agent-v3 compared to M3A introduces greater instability in context management during task execution, reducing robustness for iterative operations. M3A outperforms other baselines on iterative tasks, likely because its summary-based context management offers a simpler, more stable architecture that maintains lower failure rates when executing similar operations repeatedly. AgentProg surpasses all baseline methods, demonstrating significant advantages on both iterative and compositional tasks. The program-guided context management approach enables robust completion of both task types: it effectively preserves important information through program variables while eliminating redundant and irrelevant memory by parsing program structure. This dual capability allows AgentProg to excel where existing methods face trade-offs between memory retention and interference reduction.

Impact of Global Belief State. We further investigate the contribution of the global belief state component by comparing AgentProg with a variant that removes this module. The results reveal substantial performance improvements achieved by incorporating belief state tracking: the success rate improves from 53.9% to 78.0% on AndroidWorld (increased by 24.1%), and from 35.1% to 68.4% on AW-Extend (increased by 33.3%). These significant gains underscore the critical role of global belief state in long-horizon tasks. The global belief state mechanism enhances the agent’s understanding of environmental states, enabling it to detect and recover from unexpected situations immediately. This capability is particularly valuable for maintaining execution stability, which becomes increasingly crucial as task horizons extend. The more pronounced performance improvement on AW-Extend further validates that belief state tracking becomes even more essential for complex, multi-step tasks. By maintaining a structured representation of the current environment state, the global belief state allows AgentProg to make more informed decisions, reduce cascading errors, and ultimately achieve more robust task completion across diverse scenarios.

Impact of Execution Tree. Removing Execution Tree causes severe degradation on AW-Extend, dropping from 68.4% to 39.5%. On AndroidWorld, performance also decreases substantially from 78.0% to 61.6%. Without the Execution Tree, the agent retains all historical steps in a flat sequence, causing irrelevant context from completed loop iterations and non-taken branches to accumulate rapidly. This is particularly disadvantageous for iterative tasks in AW-Extend, where dozens of similar subtask histories overwhelm the agent’s context and interfere with decision-making. The result confirms that control flow-based context pruning is indispensable for long-horizon tasks.

Impact of Explicit Variables. Removing Explicit Variables leads to a performance drop from 78.0% to 64.2% on AndroidWorld and from 68.4% to 50.0% on AW-Extend. Without declared variables, the agent loses its “anchors” for preserving task-critical data (e.g., a list of to-do items or extracted contact information) across STP execution phase. The impact is smaller than removing the Execution Tree because the low-level Python code generation still allows the LLM some autonomy to define and track variables implicitly.

STP Error Analysis. To assess the robustness of Semantic Task Program generation, we analyze STP errors across all tasks in AndroidWorld. We define an STP error as a case where the generated program contains incorrect task decomposition, missing steps, or invalid assumptions about app workflows that cannot be recovered during execution. Out of all generated STPs, only 2.5% contain such errors, indicating that the STP generation process is highly reliable. We attribute this robustness to STP specifies core business logic (e.g., CRUD actions) without over-decomposing tasks into low-level operations, delegating the handling of unseen states and implicit prerequisites to the execution phase. Moreover, our architecture is compatible with existing re-planning strategies (Shinn et al., 2023; Sun et al., 2023): Global Belief State mechanism can detect misalignment between STP assumptions and the dynamic environment at runtime, so it is possible for AgentProg to identify the failure cause and provide feedback for adaptive STP correction. Integration with re-planning modules remains a promising direction for future investigation.

A Runtime Recovery Example of Global Belief State. Global Belief State helps detect and mitigate situations where the high-level STP makes incorrect assumptions about unseen app states. We illustrate this with the BrowserMaze task from AndroidWorld, where the agent must open a maze HTML file in Chrome and navigate to the bottom-right goal. The generated STP assumes an obstacle-free grid and plans a straightforward strategy: “repeatedly click down until no further movement, then repeatedly click right.” However, the maze contains obstacles that block this linear path. During execution, GBS tracks the agent’s position and detects a Belief-Reality Gap: the expected position after a right click does not change, indicating a blocked cell. Rather than continuing the failing strategy, GBS updates its belief to reflect the obstacle and triggers an adaptive rerouting: the agent alternates between down and right clicks and successfully reaches the bottom-right goal. This case demonstrates how GBS enables robust execution by detecting misalignment between STP assumptions and the actual environment at runtime. Further details and visualizations are provided in Appendix B.

### 5.3. Analysis of Latency and Cost

Table 3. Per-Task Token Consumption and Latency for Successful Attempts on AW-Extend. “Static Prefix” represents the static prefix prompt tokens that remain constant across invocations, “Dynamic” represents the prompt tokens that continuously change during task execution, and “Output” represents the total number of tokens generated during task execution.

Agent Task Set Static Prefix (k) Dynamic (k) Output (k) Latency (s)

UI-TARS Overall 8.1 315.5 3.2 312

Iterative 8.5 329.6 3.3 344

Compositional 7.6 294.3 3.0 263

Mobile-Agent-v3 Overall 7.9 809.4 22.1 1604

Iterative 8.9 917.6 26.1 2033

Compositional 5.8 592.9 14.2 746

AgentProg Overall 1026.4 301.3 179.7 2662

Iterative 1275.3 370.3 225.1 2910

Compositional 777.4 232.4 134.3 2164

Token Consumption and Latency. As shown in Table 3, we compare the token consumption and latency per task across AgentProg and two baseline methods: Mobile-Agent-v3 and UI-TARS, which outperform other baselines in AndroidWorld and AW-Extend respectively. First, we analyze token consumption by distinguishing between static prefix and dynamic prompt tokens since AgentProg is based on prompt engineering with substantial system prompts, a large portion of the prefix can be cached to reduce cost and improve efficiency. This distinction provides meaningful insights into the true efficiency of different approaches. Second, we compare only tasks where all three methods succeeded, as failed tasks exhibit inconsistent token patterns—some fail due to premature termination, leading to low token counts, while others fail after excessive iterative trial-and-error operations, leading to high token counts. Including failed tasks would obscure the true efficiency characteristics of each method.

Table 3 shows that UI-TARS demonstrates the lowest latency and minimal token consumption because its backbone model trained on extensive data with short static prompts and performs no additional context processing (e.g., summarization), resulting in minimal output tokens and the fastest execution speed. In contrast, Mobile-Agent-v3, while also using a trained model with relatively short static prefix (53 tokens per call), requires more tokens than UI-TARS due to its summarization and reflection process. Since AgentProg is implemented through prompt engineering, it contains a substantial static prompt prefix (12.5k tokens per call), resulting in a much higher proportion of cached tokens in the input, which reduces the cost of input tokens. Notably, AgentProg achieves significantly lower dynamic token counts compared to Mobile-Agent-v3, with performance comparable to UI-TARS. This demonstrates the effectiveness of AgentProg’s context management approach over hierarchical planning method in maintaining compact context. However, AgentProg generates significantly more output tokens compared to both baselines, as it produces extensive intermediate content for context management including belief state, and many operations involve program counter update and variable manipulations. In terms of latency, AgentProg exhibits longer execution time than Mobile-Agent-v3, primarily attributable to the increased output token generation. All three methods exhibit consistent trends across task categories: iterative tasks involve numerous repeated subtasks and thus require higher token consumption, while compositional tasks contain fewer combined subtasks and consequently demand fewer tokens.

Figure 8. Dynamic context tokens in 50 steps.

Context Tokens Over Steps. As shown in Figure 8, we analyze the dynamic context growth patterns of different methods over extended task execution in 50 steps. We compute the average dynamic tokens at each step across all agent trajectories on the AW-Extend benchmark. UI-TARS performs minimal processing of historical information, only discarding excess screenshots when necessary. Consequently, a notable drawback of this approach is the continuous growth of context length over time. UI-TARS exhibits rapid dynamic context growth in the first 5 steps, primarily because each step adds a new screenshot to the context. From step 6 onwards, the growth rate slows as UI-TARS begins discarding the oldest screenshot while adding a new one, with context expansion mainly driven by the textual portion of historical information.

Mobile-Agent-v3 also achieves relatively slow dynamic token growth thanks to its hierarchical planning mechanism, which decomposes complex tasks into subtasks and progressively removes information during execution. However, Mobile-Agent-v3 maintains dynamic tokens at a consistently high level (approximately 17k tokens) throughout task execution. In contrast, AgentProg consistently maintains dynamic tokens at a lower level (approximately 9k tokens), demonstrating advantages over both baselines. During 50 steps, the dynamic context token count in AgentProg remains nearly constant, indicating the effectiveness of our program-guided management in maintaining a compact representation of context. This stability is achieved through our approach’s ability to represent critical information in program variables and global belief state, while systematically pruning redundant and irrelevant historical details by parsing program structure. AgentProg scales efficiently to long-horizon tasks, ensuring consistent performance even as task complexity and duration increase, while requiring much fewer dynamic tokens than existing methods.

Table 4. Per-task average steps on AW-Extend.

Task Set AgentProg Mobile-Agent-v3 UI-TARS

All Attempts

Overall 129.6 134.0 225.5

Iterative 138.5 169.1 224.9

Compositional 104.6 34.0 227.0

Successful Attempts Only

Overall 37.7 51.2 43.6

Iterative 36.8 57.8 50.0

Compositional 39.5 38.0 34.0

Average Steps per Task. Table 4 presents the average number of steps required per task for AgentProg and baselines on the AW-Extend dataset. Step counts should be interpreted with caution, as different methods require varying numbers of model queries per step: Mobile-Agent-v3 queries the language model three times per step and UI-TARS queries once per step while AgentProg uses two queries per step. The trends in step counts largely match those observed in token consumption as shown in Table 3. When considering all attempts, the compositional task results show an anomalous pattern where Mobile-Agent-v3 uses very few steps, attributable to premature task termination due to forgotten subtasks. When examining only successful attempts, AgentProg requires fewer steps than baselines across the overall tasks and iterative tasks, with comparable performance on composition tasks. This demonstrates that AgentProg achieves better performance while maintaining step efficiency. The higher step counts in “All Attempts” group compared to “Successful Attempts Only” group can be attributed to two factors: first, tasks requiring fewer steps are inherently easier to complete successfully; second, failed attempts in “All Attempts” involve substantial iterative trial-and-error operations that inflate the step count.

## 6. Discussion

Accuracy-Latency Trade-off. As shown in Table 3, AgentProg incurs higher latency and cost compared to baseline methods. This overhead stems primarily from the STP execution, including Program Counter Update, Action Generation, and Global Belief State maintenance. However, we argue that AgentProg primarily targets complex, long-horizon tasks suited for background execution, where users are far more sensitive to accuracy than latency, as fast agents become impractical if they require frequent human intervention for failures. We anticipate future model-level optimizations (e.g., faster inference speeds, distilled models) and system optimizations such as merging the Program Counter Update and Action Generation modes into a single LLM call where possible.

On-Device Deployment. Currently, AgentProg relies on cloud-based LLMs, which introduce network latency and privacy concerns. A promising future direction is deploying AgentProg with on-device Small Language Models (SLMs) via model distillation (Wen et al., 2025). Our program-guided context management reduces the input dynamic context size and the static context can be eliminated entirely through distillation, which is compatible with the limited context lengths of on-device SLMs. Besides, AgentProg supports a hybrid deployment strategy. STP Generation phase requiring stronger reasoning capabilities and more knowledge can remain on the cloud, while the STP Execution phase can be distilled into an on-device SLM, offering a flexible trade-off among latency, privacy, and accuracy.

Generalization to Broader Long-Horizon Agents. Although AgentProg is evaluated on mobile GUI tasks, the program-guided context management approach can be applied to other long-horizon agent settings, including web navigation, desktop automation, and tool-use environments. We leave the empirical validation of AgentProg on non-mobile settings to future work.

## 7. Conclusion

In this work, we introduce AgentProg to address the context bottleneck in long-horizon tasks. By reframing interaction history into a structured Semantic Task Program, AgentProg enables principled context pruning and critical variable retention. Additionally, our Global Belief State mechanism maintains an active mental model to handle partial observability and environmental dynamics. Experiments on AndroidWorld and our AW-Extend benchmark demonstrate that AgentProg significantly outperforms state-of-the-art baselines. Ultimately, our work establishes a scalable foundation for robust agent deployment in complex real-world scenarios.

## Acknowledgements

This research was supported in part by the National Natural Science Foundation of China under Grant No. 62432004 and 62272261, the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China under Grant No. JYB2025XD

XM122, Wuxi Research Institute of Applied Technologies, Tsinghua University under Grant 20242001120, Tsinghua University (AIR)–AsiaInfo Technologies (China) Inc. Joint Research Center and a grant from the Guoqiang Institute, Tsinghua University.

## References

- Agashe et al. (2025) Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. 2025. Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents. arXiv:2504.00906 [cs.AI] https://arxiv.org/abs/2504.00906

- An et al. (2024) Chenxin An, Jun Zhang, Ming Zhong, Lei Li, Shansan Gong, Yao Luo, Jingjing Xu, and Lingpeng Kong. 2024. Why does the effective context length of LLMs fall short? arXiv preprint arXiv:2410.18745 (2024).

- Azim et al. (2016) Tanzirul Azim, Oriana Riva, and Suman Nath. 2016. ULink: Enabling User-Defined Deep Linking to App Content. In Proceedings of the 14th Annual International Conference on Mobile Systems, Applications, and Services (Singapore, Singapore) (MobiSys ’16). Association for Computing Machinery, New York, NY, USA, 305–318. doi:10.1145/2906388.2906416

- Cheng et al. (2024) Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. 2024. SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents. arXiv:2401.10935 [cs.HC]

- Dai et al. (2025) Gaole Dai, Shiqi Jiang, Ting Cao, Yuanchun Li, Yuqing Yang, Rui Tan, Mo Li, and Lili Qiu. 2025. Advancing mobile gui agents: A verifier-driven approach to practical deployment. arXiv preprint arXiv:2503.15937 (2025).

- Gao et al. (2025) Xinzge Gao, Chuanrui Hu, Bin Chen, and Teng Li. 2025. Chain-of-Memory: Enhancing GUI Agents for Cross-Application Navigation. arXiv:2506.18158 [cs.AI] https://arxiv.org/abs/2506.18158

- Gonzalez-Pumariega et al. (2025) Gonzalo Gonzalez-Pumariega, Vincent Tu, Chih-Lun Lee, Jiachen Yang, Ang Li, and Xin Eric Wang. 2025. The Unreasonable Effectiveness of Scaling Agents for Computer Use. arXiv:2510.02250 [cs.AI] https://arxiv.org/abs/2510.02250

- Google (2025) Google. 2025. Gemini 2.5 Pro - Google DeepMind. https://deepmind.google/models/gemini/pro/.

- Gou et al. (2025) Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. 2025. Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=kxnoqaisCT

- Gu et al. (2025) Zhangxuan Gu, Zhengwen Zeng, Zhenyu Xu, Xingran Zhou, Shuheng Shen, Yunfei Liu, Beitong Zhou, Changhua Meng, Tianyu Xia, Weizhi Chen, et al. 2025. Ui-venus technical report: Building high-performance ui agents with rft. arXiv preprint arXiv:2508.10833 (2025).

- Gur et al. (2024) Izzeddin Gur, Hiroki Furuta, Austin V Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. 2024. A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=9JQtrumvg8

- Hong et al. (2025) Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, et al. 2025. GLM-4.1 V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning. arXiv preprint arXiv:2507.01006 (2025).

- Kaelbling et al. (1998) Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. 1998. Planning and acting in partially observable stochastic domains. Artif. Intell. 101, 1–2 (May 1998), 99–134.

- Lee et al. (2024) Sunjae Lee, Junyoung Choi, Jungjae Lee, Munim Hasan Wasi, Hojun Choi, Steve Ko, Sangeun Oh, and Insik Shin. 2024. MobileGPT: Augmenting LLM with Human-like App Memory for Mobile Task Automation. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking (Washington D.C., DC, USA) (ACM MobiCom ’24). Association for Computing Machinery, New York, NY, USA, 1119–1133. doi:10.1145/3636534.3690682

- Li et al. (2025c) Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Yida Zhao, Liwen Zhang, Litu Ou, Dingchu Zhang, Xixi Wu, Jialong Wu, et al. 2025c. Websailor-v2: Bridging the chasm to proprietary agents via synthetic data and scalable reinforcement learning. arXiv preprint arXiv:2509.13305 (2025).

- Li et al. (2025a) Ning Li, Qiqiang Lin, Zheng Wu, Xiaoyun Mo, Weiming Zhang, Yin Zhao, Xiangmou Qu, Jiamu Zhou, Jun Wang, Congmin Zheng, et al. 2025a. ColorAgent: Building A Robust, Personalized, and Interactive OS Agent. arXiv preprint arXiv:2510.19386 (2025).

- Li et al. (2025b) Ning Li, Xiangmou Qu, Jiamu Zhou, Jun Wang, Muning Wen, Kounianhua Du, Xingyu Lou, Qiuying Peng, and Weinan Zhang. 2025b. MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation. arXiv preprint arXiv:2507.16853 (2025).

- Li and Riva (2018) Toby Jia-Jun Li and Oriana Riva. 2018. Kite: Building Conversational Bots from Mobile Apps. In Proceedings of the 16th Annual International Conference on Mobile Systems, Applications, and Services (Munich, Germany) (MobiSys ’18). Association for Computing Machinery, New York, NY, USA, 96–109. doi:10.1145/3210240.3210339

- Li et al. (2020) Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge. 2020. Mapping Natural Language Instructions to Mobile UI Action Sequences. In Annual Conference of the Association for Computational Linguistics (ACL 2020). https://www.aclweb.org/anthology/2020.acl-main.729.pdf

- Liang et al. (2023) Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. 2023. Code as Policies: Language Model Programs for Embodied Control. In 2023 IEEE International Conference on Robotics and Automation (ICRA). 9493–9500. doi:10.1109/ICRA48891.2023.10160591

- Liu et al. (2018) Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. 2018. Reinforcement Learning on Web Interfaces using Workflow-Guided Exploration. In International Conference on Learning Representations (ICLR). https://arxiv.org/abs/1802.08802

- Liu et al. (2024) Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics 12 (2024), 157–173. doi:10.1162/tacl_a_00638

- Liu et al. (2025) Shunyu Liu, Minghao Liu, Huichi Zhou, Zhenyu Cui, Yang Zhou, Yuhao Zhou, Wendong Fan, Ge Zhang, Jiajun Shi, Weihao Xuan, et al. 2025. Verigui: Verifiable long-chain gui dataset. arXiv preprint arXiv:2508.04026v1 (2025).

- Liu et al. (2026) Shunyu Liu, Minghao Liu, Huichi Zhou, Zhenyu Cui, Yang Zhou, Yuhao Zhou, Jialiang Gao, Heng Zhou, Yunhao Yang, Wendong Fan, puzhen zhang, Ge Zhang, Jiajun Shi, Weihao Xuan, Jiaxing Huang, Shuang Luo, Fang Wu, Heli Qi, Qingcheng Zeng, Junjie Wang, Aosong Feng, Jindi Lv, Sicong Jiang, Ziqi Ren, Wangchunshu Zhou, Zhenfei Yin, Wenlong Zhang, Guohao Li, Wenhao Yu, Lei Ma, Lei Bai, Qunshu Lin, Mingli Song, and Dacheng Tao. 2026. VeriWeb: Verifiable Long-Chain Web Benchmark for Agentic Information-Seeking. arXiv preprint arXiv:2508.04026v2 (2026).

- Mozannar et al. (2025) Hussein Mozannar, Gagan Bansal, Cheng Tan, Adam Fourney, Victor Dibia, Jingya Chen, Jack Gerrits, Tyler Payne, Matheus Kunzler Maldaner, Madeleine Grunde-McLaughlin, et al. 2025. Magentic-UI: Towards Human-in-the-loop Agentic Systems. arXiv preprint arXiv:2507.22358 (2025).

- Nguyen et al. (2025) Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, Xintong Li, Jing Shi, Hongjie Chen, Viet Dac Lai, Zhouhang Xie, Sungchul Kim, Ruiyi Zhang, Tong Yu, Mehrab Tanjim, Nesreen K. Ahmed, Puneet Mathur, Seunghyun Yoon, Lina Yao, Branislav Kveton, Jihyung Kil, Thien Huu Nguyen, Trung Bui, Tianyi Zhou, Ryan A. Rossi, and Franck Dernoncourt. 2025. GUI Agents: A Survey. arXiv:2412.13501 [cs.AI] https://arxiv.org/abs/2412.13501

- Qin et al. (2025) Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. 2025. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326 (2025).

- Rawles et al. (2024) Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. 2024. AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents. arXiv:2405.14573 [cs.AI] https://arxiv.org/abs/2405.14573

- Schroeder et al. (2025) Philip Schroeder, Nathaniel W Morgan, Hongyin Luo, and James Glass. 2025. Thread: Thinking deeper with recursive spawning. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 8418–8442.

- Shen et al. (2025) Leming Shen, Qiang Yang, Yuanqing Zheng, and Mo Li. 2025. AutoIOT: LLM-Driven Automated Natural Language Programming for AIoT Applications. In Proceedings of the 31st Annual International Conference on Mobile Computing and Networking (Kerry Hotel, Hong Kong, Hong Kong, China) (ACM MOBICOM ’25). Association for Computing Machinery, New York, NY, USA, 468–482. doi:10.1145/3680207.3723486

- Shinn et al. (2023) Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal Reinforcement Learning.

- Sondik (1978) Edward J. Sondik. 1978. The Optimal Control of Partially Observable Markov Processes Over the Infinite Horizon: Discounted Costs. Operations Research 26, 2 (1978), 282–304. https://doi.org/10.1287/opre.26.2.282

- Song et al. (2025a) Linxin Song, Yutong Dai, Viraj Prabhu, Jieyu Zhang, Taiwei Shi, Li Li, Junnan Li, Silvio Savarese, Zeyuan Chen, Jieyu Zhao, et al. 2025a. Coact-1: Computer-using agents with coding as actions. arXiv preprint arXiv:2508.03923 (2025).

- Song et al. (2025b) Yuanyi Song, Heyuan Huang, Qiqiang Lin, Yin Zhao, Xiangmou Qu, Jun Wang, Xingyu Lou, Weiwen Liu, Zhuosheng Zhang, Yong Yu, Weinan Zhang, and Zhaoxiang Wang. 2025b. ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks. arXiv preprint arXiv:2510.14621 (2025).

- Sun et al. (2023) Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. 2023. AdaPlanner: Adaptive Planning from Feedback with Language Models. arXiv preprint arXiv:2305.16653 (2023).

- Sun et al. (2022) Liangtai Sun, Xingyu Chen, Lu Chen, Tianle Dai, Zichen Zhu, and Kai Yu. 2022. META-GUI: Towards Multi-modal Conversational Agents on Mobile GUI. arXiv preprint arXiv:2205.11029 (2022).

- Tan et al. (2024) Weihao Tan, Wentao Zhang, Xinrun Xu, Haochong Xia, Ziluo Ding, Boyu Li, Bohan Zhou, Junpeng Yue, Jiechuan Jiang, Yewen Li, Ruyi An, Molei Qin, Chuqiao Zong, Longtao Zheng, Yujie Wu, Xiaoqiang Chai, Yifei Bi, Tianbao Xie, Pengjie Gu, Xiyun Li, Ceyao Zhang, Long Tian, Chaojie Wang, Xinrun Wang, Börje F. Karlsson, Bo An, Shuicheng Yan, and Zongqing Lu. 2024. Cradle: Empowering Foundation Agents Towards General Computer Control. arXiv:2403.03186 [cs.AI] https://arxiv.org/abs/2403.03186

- Wang et al. (2025b) Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, Bo Li, Chen Dun, Chong Liu, Fuxing Leng, Han rui Wang, Hao Yu, Haobin Chen, Hongyi Guo, Jing Su, Jingjia Huang, Kai Shen, Kaiyu Shi, Lin Yan, Pei-Xiong Zhao, Pengfei Liu, Qinghao Ye, Renjie Zheng, Wayne Xin Zhao, Wen Heng, Wenhao Huang, Wenqian Wang, Xiao-Jun Qin, Yi Lin, Youbing Wu, Zehui Chen, Zihao Wang, Baoquan Zhong, Xinchun Zhang, Xujing Li, YuanFang Li, Zhongkai Zhao, Chengquan Jiang, Faming Wu, Hao Zhou, Jinlin Pang, Li Han, Qianli Ma, Siyao Liu, Songhua Cai, Wenqi Fu, Xin Liu, Zhi Zhang, Bo Zhou, Guoliang Li, Jiajun Shi, Jiale Yang, Jie Tang, Li Li, Taoran Lu, Woyu Lin, Xiao Tong, Xinyao Li, Yichi Zhang, Yu Miao, Zheng-Wang Jiang, Zili Li, Zi-Hao Zhao, Chenxi Li, Dehua Ma, Feng Lin, Ge Zhang, Haihua Yang, Hangyu Guo, Hongda Zhu, Jiaheng Liu, Jun-Yan Du, Kai Cai, Kuanye Li, Lichen Yuan, Mei Han, Minchao Wang, Shuyu Guo, Tianhao Cheng, Xiaobo Ma, Xiao Xiao, Xiaolong Huang, Xinjie Chen, Yi-Zhen Du, Yilin Chen, Yiwen Wang, Zhaojian Li, Zhen Yang, Zhiyuan Zeng, Chaolin Jin, Chen Li, Haolin Chen, Haolin Chen, Jian Chen, Qinghao Zhao, and Guang Shi. 2025b. UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning. arXiv preprint arXiv:2509.02544 (2025).

- Wang et al. (2024c) Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024c. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158 (2024).

- Wang et al. (2024b) Luyuan Wang, Yongyu Deng, Yiwei Zha, Guodong Mao, Qinmin Wang, Tianchen Min, Wei Chen, and Shoufa Chen. 2024b. MobileAgentBench: An Efficient and User-Friendly Benchmark for Mobile LLM Agents. arXiv preprint arXiv:2406.08184 (2024).

- Wang et al. (2024a) Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. 2024a. Executable Code Actions Elicit Better LLM Agents. In ICML. arXiv:2402.01030

- Wang et al. (2025a) Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, and Daniel Fried. 2025a. Inducing Programmatic Skills for Agentic Tasks. In Second Conference on Language Modeling. https://openreview.net/forum?id=lsAY6fWsog

- Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc., 24824–24837. https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf

- Wen et al. (2024) Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. 2024. AutoDroid: LLM-powered Task Automation in Android. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking (Washington D.C., DC, USA) (ACM MobiCom ’24). Association for Computing Machinery, New York, NY, USA, 543–557. doi:10.1145/3636534.3649379

- Wen et al. (2025) Hao Wen, Shizuo Tian, Borislav Pavlov, Wenjie Du, Yixuan Li, Ge Chang, Shanhui Zhao, Jiacheng Liu, Yunxin Liu, Ya-Qin Zhang, and Yuanchun Li. 2025. AutoDroid-V2: Boosting SLM-based GUI Agents via Code Generation. Association for Computing Machinery, New York, NY, USA, 223–235. https://doi.org/10.1145/3711875.3729134

- Wu et al. (2024a) Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. 2024a. Os-copilot: Towards generalist computer agents with self-improvement. arXiv preprint arXiv:2402.07456 (2024).

- Wu et al. (2024b) Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. 2024b. OS-ATLAS: A Foundation Action Model for Generalist GUI Agents. arXiv preprint arXiv:2410.23218 (2024).

- Xu et al. (2024a) Jiajun Xu, Zhiyuan Li, Wei Chen, Qun Wang, Xin Gao, Qi Cai, and Ziyuan Ling. 2024a. On-device language models: A comprehensive review. arXiv preprint arXiv:2409.00088 (2024).

- Xu et al. (2025) Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, and Yuxiao Dong. 2025. MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents. arXiv preprint arXiv:2509.18119 (2025).

- Xu et al. (2024b) Yifan Xu, Xiao Liu, Xueqiao Sun, Siyi Cheng, Hao Yu, Hanyu Lai, Shudan Zhang, Dan Zhang, Jie Tang, and Yuxiao Dong. 2024b. AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents. arXiv:2410.24024 [cs.AI] https://arxiv.org/abs/2410.24024

- Yan et al. (2023) An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. 2023. GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation. arXiv preprint arXiv:2311.07562 (2023).

- Yao et al. (2023) Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv:2305.10601 [cs.CL]

- Ye et al. (2025b) Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. 2025b. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144 (2025).

- Ye et al. (2025a) Suyu Ye, Haojun Shi, Darren Shih, Hyokun Yun, Tanya Roosta, and Tianmin Shu. 2025a. RealWebAssist: A Benchmark for Long-Horizon Web Assistance with Real-World Users. arXiv preprint arXiv:2504.10445 (2025).

- Zhang et al. (2024) Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. 2024. UFO: A UI-Focused Agent for Windows OS Interaction. arXiv preprint arXiv:2402.07939 (2024).

- Zhang et al. (2023) Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2023. AppAgent: Multimodal Agents as Smartphone Users. arXiv:2312.13771 [cs.CV]

- Zhang et al. (2025) Zhenyu Zhang, Tianyi Chen, Weiran Xu, Alex Pentland, and Jiaxin Pei. 2025. ReCAP: Recursive Context-Aware Reasoning and Planning for Large Language Model Agents. In Conference on Neural Information Processing Systems (NeurIPS).

- Zhang and Zhang (2024) Zhuosheng Zhang and Aston Zhang. 2024. You Only Look at Screens: Multimodal Chain-of-Action Agents. In Findings of the Association for Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 3132–3149. doi:10.18653/v1/2024.findings-acl.186

- Zheng et al. (2024) Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. GPT-4V(ision) is a Generalist Web Agent, if Grounded. In Forty-first International Conference on Machine Learning (ICML’24). https://openreview.net/forum?id=piecKJ2DlB

- Zhou et al. (2025) Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. 2025. MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents. https://arxiv.org/abs/2506.15841

## Appendix A Syntax of Semantic Task Program

The syntax of Semantic Task Program (STP) is designed to resolve the conflict between the need for structural rigor in workflows and the inherent ambiguity of agent tasks. While Semantic Task Program adopts a natural-language-style appearance to ensure expressiveness, it is fundamentally a structured language. It imposes a rigid skeleton of control flow (loops, conditionals, functions) to maintain logical consistency, while allowing the atomic instructions within that skeleton to remain "soft" and flexible.

This design philosophy differs from traditional coding, which demands precision everywhere, and prompt engineering, which lacks structural constraints. Semantic Task Program allows developers to be precise about procedure (e.g., "iterate exactly 10 times") while being fuzzy about implementation (e.g., "extract the sentiment"). This section outlines how Semantic Task Program implements this hybrid approach.

### A.1. Basic Statements and Comments

Semantic Task Program adopts an intent-based programming style. Instructions are written as imperative or declarative sentences that prioritize conveying the user’s goal over adhering to strict syntactic rules.

⬇ 1 # This is a comment, used for human-readable annotations. 2 tell user "Hello, world!" # Inline comments are also supported. 3 4 # The following statements are also valid: 5 send "Hello, world!" to the user 6 notify user with message "Hello, world!" 7 send a greeting message to user

Comments are ignored by AgentProg and serve only to document the code. Statements like tell user "..." are direct commands that are translated into actions. For familiarity, Semantic Task Program also permits Python-style syntax where appropriate (e.g., print("Hello, world!")), which the AgentProg processes as an equivalent intent rather than executing directly.

### A.2. Variables and Data Management

In agentic workflows, data is often unstructured (e.g., natural text, web content) and does not fit neatly into the rigid types of C++ or Java. Semantic Task Program addresses this by treating variables as named containers for information, explicitly demarcated by curly braces {} (e.g., {userName}). This notation serves a dual purpose: it provides a clear anchor for the AgentProg to track state within a fuzzy instruction, and it allows the language to handle unstructured data without requiring complex schema definitions upfront.

Declaration and Assignment: Variables are declared implicitly upon their first assignment. Semantic Task Program supports multiple natural phrasings for assignment, treating them as synonyms.

⬇ 1 set variable {userName} to "Alice" 2 store 100 into {initialScore} 3 set {userCount} to 0 # Python-style assignment is also valid.

Using Variables: To reference a variable’s value, simply include its braced name within a statement. The AgentProg substitutes it with its current value during execution.

⬇ 1 tell user "Welcome, {userName}! Your score is {initialScore}." 2 calculate {initialScore} + 50, record as {finalScore}

Dynamic Type System: Semantic Task Program employs a dynamic type system where users are not required to declare types explicitly. The AgentProg infers and manages types—such as Text, Number, Boolean, List, and Object—based on context and value. This approach aligns with the needs of both novice programmers and LLMs, which may struggle with strict static typing. While types are inferred, users can provide hints to guide the interpreter, for instance, when interacting with an AI model.

- •

List: An ordered collection.

⬇ 1 record list "apples", "bananas", "cherries" as {fruitBasket}

- •

Object: A key-value store, akin to a dictionary or JSON object. Properties can be accessed using dot notation ({product.price}) or possessive phrasing ({myCar’s color}).

⬇ 1 create an object {product} with "name" as "Laptop" and "price" as 1200 2 tell user "{product.name} is available for {product.price}."

- •

Table: A structured list of objects, ideal for representing tabular data.

⬇ 1 read table "sales_data.csv" as {salesReport}

Crucially, the AgentProg handles potential type mismatches intelligently. Instead of crashing, it attempts a reasonable type coercion (e.g., converting the text "123" to a number) or reports a descriptive, human-understandable issue.

### A.3. Control Flow

Control flow structures in Semantic Task Program use natural language keywords and rely on indentation to define code blocks, similar to Python. This maintains readability while avoiding complex bracketing.

Conditional Statements: if, else, else if structures evaluate conditions expressed in natural language or with standard operators.

⬇ 1 ask user "Enter your age:", get response as {ageInput} 2 convert {ageInput} to number, as {userAge} 3 if {userAge} < 18: 4 tell user "You are a minor." 5 else, if {userAge} is between 18 and 65: # Natural language condition allowed 6 tell user "You are an adult." 7 else: 8 tell user "You are a senior citizen."

Loops: Semantic Task Program supports various loop constructs, each initiated with flexible, descriptive phrasing.

- •

Fixed Repetitions:

⬇ 1 repeat 5 times: 2 # {loop.iteration} is a special variable (1-indexed) 3 tell user "This is message number {loop.iteration}"

- •

Iterating over Collections:

⬇ 1 iterate through each item in {fruitBasket} as {fruit}: 2 tell user "Found fruit: {fruit}"

- •

Conditional Loops:

⬇ 1 while {systemStatus} is "active": 2 check for new messages 3 wait 10 seconds

### A.4. Functions

To promote modularity and code reuse, Semantic Task Program allows users to define reusable blocks of logic as functions.

Defining a Function: A function is defined with a name, a list of expected inputs (parameters), and a body of instructions.

⬇ 1 define function named "calculateArea": 2 function inputs: {length} (number), {width} (number) 3 calculate {length} * {width}, record as {area} 4 function returns {area}

Executing a Function: Functions are called by name, with arguments passed using a clear key-value format.

⬇ 1 execute function "calculateArea", with {length} as 10 and {width} as 5, save result as {roomArea} 2 tell user "The room area is {roomArea} square units."

### A.5. Interactions with External Tools

A cornerstone of Semantic Task Program is its native ability to interact with the broader digital environment, including AI models and external tools, through high-level, declarative commands.

Tool Use (File System, Browser): Semantic Task Program provides a unified, high-level interface for common tool interactions. This design allows users to specify *what* they want to achieve, leaving the *how* to the AgentProg.

⬇ 1 # File System 2 read text from file "config.txt" as {configContent} 3 save {summary} to file "summary.txt" 4 5 # Web Browser Automation 6 open URL "https://en.wikipedia.org" 7 search "Hello World" 8 wait for page to load 9 get current page content as {wikiPageContent}

The AgentProg’s context-aware nature is critical here. It resolves ambiguous references like "search input" or "Search button" by analyzing the current state of the web page (its DOM structure and visual layout), making automation scripts more robust and less brittle.

## Appendix B Runtime Recovery Example of Global Belief State

Figure 9 demonstrates a runtime recovery example of Global Belief State.

Figure 9. A runtime recovery example of global belief state.

## Appendix C Tasks in AW-Extend

Table 5 shows the task descriptions of all tasks in AW-Extend.

Table 5. Task descriptions in AW-Extend. “Long” denotes tasks with $n{=}10$ subtasks and “SuperLong” denotes $n{=}20$ .

Task Name Instruction Template

ContactsAddContactAndSms Create a new contact for {name}. Their number is {number}. Then, send ‘hello, {name}’ to {number} via SMS using Simple SMS Messenger.

ContactsAddMultipleContactsAndSms For the following persons: {contacts}, add them as new contacts, and then use Simple SMS Messenger to send each of them a ‘hello, [Name]’ message.

MarkorFetchMultipleNotesAndSms Retrieve the file name they sent you via Simple SMS Messenger, open it in Markor, and then send its contents back to that number for each number in {numbers}.

MarkorFetchNoteAndSms Obtain the file name sent to you by {number} in Simple SMS Messenger, open the file in Markor, and send its contents back to {number}.

MarkorTodoList Open my Markor note {file_name} and complete all the tasks on the list.

SimpleCalendarDeleteEventsLong In Simple Calendar Pro, delete all the calendar events on {year}-{month}-{day}.

SimpleCalendarDeleteEventsOnRelativeDayLong In Simple Calendar Pro, delete all events scheduled for this {day_of_week}.

SimpleCalendarDeleteEventsSuperLong In Simple Calendar Pro, delete all the calendar events on {year}-{month}-{day}.

SimpleCalendarDeleteEventsOnRelativeDaySuperLong In Simple Calendar Pro, delete all events scheduled for this {day_of_week}.

ExpenseAddMultipleLong Add the following expenses into the Pro Expense: {expenses}.

ExpenseAddMultipleSuperLong Add the following expenses into the Pro Expense: {expenses}.

ExpenseDeleteMultipleSuperLong Delete the following expenses from Arduia Pro Expense: {expenses}.

ExpenseDeleteMultipleLong2 Delete the following expenses from Arduia Pro Expense: {expenses}.

ExpenseDeleteMultipleSuperLong2 Delete the following expenses from Arduia Pro Expense: {expenses}.

RecipeAddMultipleRecipesLong Add the following recipes into the Broccoli app: {recipes}.

RecipeAddMultipleRecipesSuperLong Add the following recipes into the Broccoli app: {recipes}.

RecipeDeleteMultipleRecipesLong Delete the following recipes from Broccoli app: {titles}.

RecipeDeleteMultipleRecipesSuperLong Delete the following recipes from Broccoli app: {titles}.

MarkorMergeNotesLong Merge the contents of Markor notes {file1_name}, {file2_name} and {file3_name} into a new note named {new_file_name}. Add a new line between each.

## Appendix D End-to-End Pipeline Walkthrough

We provide a concrete walkthrough of a representative AW-Extend task that combines explicit variables, loop control flow, cross-app execution, and persistent belief tracking. The user instruction is:

For the following persons: Name: Hana Ferreira, Number: +10662908339, Name: Sophie Martin, Number: +18723713947, Name: Olivia Alves, Number: +16278036185,add them as new contacts, and then use Simple SMS Messenger to send each of them a ’hello, [Name]’ message, where [Name] is the name of each contact.

STP Generation Phase. The high-level user request is first converted into the following STP:

⬇ 1 # Create a list of objects to store the contact information 2 Create a list of objects named {contact_list}, with the following items: 3 An object with "name" set to "Hana Ferreira" and "number" set to "+10662908339" 4 An object with "name" set to "Sophie Martin" and "number" set to "+18723713947" 5 An object with "name" set to "Olivia Alves" and "number" set to "+16278036185" 6 7 # Iterate through each contact to add them and send a message 8 Iterate through each item in {contact_list}, recorded as {current_contact}: 9 # Add the person as a new contact 10 In the ‘Contacts‘ app, add a new contact with name {current_contact.name} and phone number {current_contact.number}, and save it. 11 12 # Send a personalized SMS message 13 In the ‘Simple SMS Messenger‘ app, send a message to {current_contact.number} with the content "hello, {current_contact.name}".

STP Execution Phase. In STP Execution Phase, AgentProg alternates between two modes to executes STP.

Representative prompt instances. At execution time, the LLM receives a fixed system prefix plus a dynamic context. Below we show the dynamic prompt content for some non-trivial steps. This is the information the model uses to ground the current STP instruction into low-level actions.

Action Generation prompt instance. Here is the prompt of a single step during the execution of STP:

⬇ 1 Mode: Action
