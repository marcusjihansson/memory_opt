❯ uv run main.py --keep-cache

======================================================================
FOUR-TIER MEMORY SYSTEM DEMO
======================================================================
[CONFIG] DSPy configured successfully

[INIT] Initializing memory system...
[INIT] Connected to Redis and PostgreSQL
[META-MEMORY] Initialized new narrative state

[SETUP] Session initialized
Session ID: demo_session_001
User ID: demo_user_001
Constraints: 5

---

## PHASE 1: Building Memory Context

[USER] Turn 1: I'm learning Python for data science...

[USER] Turn 2: I prefer hands-on tutorials over reading documentation...

[USER] Turn 3: I'm working on a healthcare project that requires HIPAA comp...

[USER] Turn 4: Remember: I always want code examples with detailed explanat...

---

## PHASE 2: Short-term Memory Analysis

[SHORT-TERM] Importance scores:
msg_4: 0.83
msg_2: 0.67
msg_3: 0.53
msg_1: 0.51

---

## PHASE 3: Long-term Memory Retrieval

[RETRIEVAL] Query: "What are the user's learning preferences?"
[CONTEXT] Retrieved in 1170.8ms
[CONTEXT] Length: 876 chars

---

## PHASE 4: MetaMemory Coherence Validation

[ACTION CHECK] "Suggest switching to JavaScript for web development"
[META-MEMORY] Coherence: FAIL (score: 0.50)
[VIOLATION] Action doesn't support session goal: Help user learn Python for data science with hands-on tutorials
[VIOLATION] Suggests non-Python technology when user is focused on Python (constraint: Focus on Python and data science topics)
[REASONING] Goal hierarchy alignment: 0.24 - Conflicts with session goals
[REASONING] Life goal alignment: 0.18 - Action may hinder life-level objectives

[ACTION CHECK] "Recommend a tutorial without any code examples"
[META-MEMORY] Coherence: FAIL (score: 0.50)
[VIOLATION] Action doesn't support session goal: Help user learn Python for data science with hands-on tutorials
[VIOLATION] Violates user preference for code examples (constraint: User prefers hands-on tutorials with code examples)
[REASONING] Goal hierarchy alignment: 0.37 - Conflicts with session goals
[REASONING] Life goal alignment: 0.23 - Action may hinder life-level objectives

[ACTION CHECK] "Store patient data in an unencrypted plain text file"
[META-MEMORY] Coherence: FAIL (score: 0.50)
[VIOLATION] Action doesn't support session goal: Help user learn Python for data science with hands-on tutorials
[VIOLATION] Violates HIPAA compliance requirement for healthcare data (constraint: Project requires HIPAA compliance)
[REASONING] Goal hierarchy alignment: 0.19 - Conflicts with session goals
[REASONING] Life goal alignment: 0.15 - Action may hinder life-level objectives

[ACTION CHECK] "Suggest a Python pandas tutorial with hands-on examples"
[META-MEMORY] Coherence: PASS (score: 0.80)
[VIOLATION] Action doesn't support session goal: Help user learn Python for data science with hands-on tutorials
[REASONING] Goal hierarchy alignment: 0.61 - Conflicts with session goals
[REASONING] Life goal alignment: 0.18 - Action may hinder life-level objectives

[SUMMARY] Coherence test results:

---

    Suggest switching to JavaScript for web ... | Expected: FAIL | Got: FAIL | OK
    Recommend a tutorial without any code ex... | Expected: FAIL | Got: FAIL | OK
    Store patient data in an unencrypted pla... | Expected: FAIL | Got: FAIL | OK
    Suggest a Python pandas tutorial with ha... | Expected: PASS | Got: PASS | OK

---

## PHASE 5: RLM Multi-hop Reasoning

[USER] Turn 5: Based on everything you know about me, recommend a project...

[RLM] Question: "Given the user's interest in Python, data science, healthcare domain with HIPAA requirements, and preference for hands-on learning, what project would you recommend and why?"
2026/01/23 17:07:30 WARNING dspy.primitives.python_interpreter: Unable to find the Deno cache dir.
2026/01/23 17:09:16 WARNING dspy.predict.rlm: RLM reached max iterations, using extract to get final output
[RLM] Method: rlm
[RLM] Time: 114421.9ms
[RLM] Answer: **Project Outline:**

- **Project Title:** Healthcare Data Analysis Tool
- **Objective:** Analyze healthcare data while ensuring HIPAA compliance.
- **Key Tasks:**
  - Acquire a sample healthcare data...

---

## SESSION CLEANUP

[CONSOLIDATION] Moved 2 messages to long-term storage
[SESSION END] Session demo_session_001 finalized

---

## PERSISTENCE VERIFICATION

[REDIS] Total keys: 1
[REDIS] Sample keys matching '_demo_session_001_': ['context:demo_session_001']
[POSTGRES] pgvector installed: yes
[POSTGRES] conversations: present
[POSTGRES] user_profiles: present
[POSTGRES] session_summaries: present
[POSTGRES] consolidation_log: present
[POSTGRES] narrative_states: present
[POSTGRES] conversations rows: total=66, session=33, user=33

======================================================================
DEMO COMPLETE
======================================================================

Total turns: 5
Memories stored: 0
Coherence tests: 4 (4/4 as expected)
