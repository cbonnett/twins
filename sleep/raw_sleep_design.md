
# SIESTA-LLM — Sleep Improvement with Evidence-based Structured Therapy using a Large Language Model

## **Executive Summary:**

**Goal:** Demonstrate that a large language model (LLM) delivering digital Cognitive Behavioral Therapy for Insomnia (dCBT-I) produces clinically meaningful improvements versus an active control within 8 weeks, with durability at follow-up. Evidence base: large effects for dCBT-I in randomized trials ([Recent 2025 meta-analysis of fully automated dCBT-I](https://www.nature.com/articles/s41746-025-01514-4); [Meta-analysis showing large effects on insomnia and comorbid conditions](https://pmc.ncbi.nlm.nih.gov/articles/PMC10624170/)) and CBT-I as first-line care ([ACP guideline recommending CBT-I first-line](https://www.acpjournals.org/doi/10.7326/M15-2175)).

**Why insomnia:** The signal is big and fast. dCBT-I delivers large improvements (standardized mean difference ≈ −0.76) and typically reads out in ~6–8 weeks ([Meta-analysis showing large effects](https://pmc.ncbi.nlm.nih.gov/articles/PMC10624170/); [AASM guideline on behavioral treatments for insomnia](https://jcsm.aasm.org/doi/10.5664/jcsm.8986)). Digital CBT-I can be non-inferior to face-to-face CBT-I ([Non-inferiority trial of automated dCBT-I versus face-to-face CBT-I](https://academic.oup.com/sleep/article/44/12/zsab185/6325440)), and there is a shortage of CBT-I-trained clinicians, so automation expands access ([Global distribution of Behavioral Sleep Medicine clinicians](https://pmc.ncbi.nlm.nih.gov/articles/PMC5070478/)).

**Why twin-focused RCT:** A conventional individually randomised trial provides the primary causal estimate. With the majority of participants being twins (90%+ of the sample, approximately 50% monozygotic and 50% dizygotic), co-twin control analyses adjust for shared genetics, age, and early-life environment—substantially strengthening internal validity, boosting power, and reducing confounding. The twin-focused design also offers a compelling public-engagement angle.

**What success looks like:** A clinically meaningful reduction on the Insomnia Severity Index (ISI) at 8 weeks, meeting the minimal clinically important difference (MCID) of ~6 points ([ISI MCID commonly cited as 6 points](https://pubmed.ncbi.nlm.nih.gov/19689221/)), with confirmatory secondary improvements in sleep efficiency from wrist actigraphy per [American Academy of Sleep Medicine actigraphy guidance](https://jcsm.aasm.org/doi/10.5664/jcsm.7230); depressive symptoms on the Patient Health Questionnaire-9 (PHQ-9); anxiety on the Generalized Anxiety Disorder-7 (GAD-7); and daytime function using the Patient-Reported Outcomes Measurement Information System (PROMIS).

---

## **Study Design:**

- **Population:** Adults 50–65 (allow 30–70); clinician-confirmed insomnia by DSM-5 (Diagnostic and Statistical Manual of Mental Disorders, 5th Edition) or ICSD-3 (International Classification of Sleep Disorders, 3rd Edition); ISI ≥10. Common comorbidities (e.g., depression/anxiety) included unless unsafe.
    - **Key exclusions (safety):** Active suicidality (PHQ-9 item 9), severe cognitive impairment, untreated moderate–severe sleep apnea (high Apnea-Hypopnea Index), prior CBT-I within 6 months, inability to use a smartphone.
- **Design:** Twin-focused, individually randomized controlled trial (LLM dCBT-I vs active control) with 90%+ of participants being twins (approximately 50% monozygotic, 50% dizygotic). Primary analysis at the individual level; within-pair co-twin secondary analysis for twin pairs.
- **Intervention:** LLM-delivered, fully automated dCBT-I (weekly structured modules, nightly sleep diaries, sleep-restriction schedules, stimulus control, cognitive restructuring, empathetic coaching). Digital CBT-I is supported by RCTs and meta-analyses ([Recent 2025 meta-analysis of fully automated dCBT-I](https://www.nature.com/articles/s41746-025-01514-4); [Non-inferiority trial of automated dCBT-I versus face-to-face CBT-I](https://academic.oup.com/sleep/article/44/12/zsab185/6325440)).
- **Control:** Attention-matched sleep-hygiene education app + nightly diary; no LLM/chat and no CBT-I core elements (e.g., no sleep restriction).
- **Assessments:** Baseline and Week 8 (primary); optional Week 4 mid-course check; actigraphy window per guideline ([AASM actigraphy guideline](https://jcsm.aasm.org/doi/10.5664/jcsm.7230)).
- **Randomization & stratification:** Individual randomization; stratify by zygosity (monozygotic/dizygotic), age band, and baseline ISI.
- **Contamination controls:** Individual logins; asynchronous weekly unlocks; "do not share content" guidance; preference for twins living separately; intention-to-treat analysis with sensitivity checks.
- **Engagement supports (non-confounding):** Mandatory sleep diaries; light coordinator outreach for missed sessions; optional avatar (embodied conversational agent); brief clinician touchpoints.

---

## **Endpoints**

**What we will measure and how we'll claim success**

**Primary outcome (the one headline result):**

- **Insomnia Severity Index (ISI)** — change from baseline to Week 8. We will claim success if the LLM arm shows a clinically meaningful improvement exceeding the established MCID (~6 points) ([ISI MCID commonly cited as 6 points](https://pubmed.ncbi.nlm.nih.gov/19689221/); [Recent systematic review of MCID values in insomnia trials](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-024-02297-0)). (Eight-week insomnia endpoints are standard in trials; example: [Eight-week insomnia severity endpoint in an RCT](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2823423).)

**Key secondary outcomes (tested in order, only moving to the next if the previous shows a real effect):**

1. **Sleep efficiency (actigraphy)** — change from baseline to Week 4–8 (objective signal; recording windows per [AASM actigraphy guideline](https://jcsm.aasm.org/doi/10.5664/jcsm.7230)).
2. **PHQ-9 (depression)** — change from baseline to Week 8.
3. **GAD-7 (anxiety)** — change from baseline to Week 8.

*Why this order?* It prioritizes the objective sleep signal and protects power by limiting sequential confirmatory tests after the primary.

**Other outcomes we will track (to learn and tell the fuller story):**

- **Daytime function & QoL (Quality of Life):** PROMIS Physical Function and related quality-of-life measures.
- **Process/engagement:** Conversation depth, return visits, homework completion (mechanistic indicators of adherence and therapeutic alliance).
- **Durability:** ISI maintenance and sleep-medication use at follow-up.

**How we judge results (simple rules):**

- We test the primary (ISI) at p<0.05. If positive, we test sleep efficiency, then PHQ-9, then GAD-7 in that order at p<0.05.
- For the grouped "other outcomes," we control multiplicity with false-discovery rate (FDR).
- Alongside p-values, we always report effect sizes (points on ISI, % sleep efficiency, PHQ-9/GAD-7 points) and whether changes are clinically meaningful (e.g., ISI MCID per [ISI MCID commonly cited as 6 points](https://pubmed.ncbi.nlm.nih.gov/19689221/)).

---