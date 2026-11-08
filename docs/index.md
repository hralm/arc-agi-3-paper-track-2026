---
layout: default
---

# Certifying No Effect and Identifying the Win Condition Under Per-Action Scoring in ARC-AGI-3

*ARC Prize 2026 · Paper Track · how the work was done, by the author*

- Paper (PDF): [`paper/`](https://github.com/hralm/arc-agi-3-paper-track-2026/tree/main/paper), added at submission on 8 November 2026
- Code, bench, tests, and manifests: [this repository](https://github.com/hralm/arc-agi-3-paper-track-2026)
- Notebooks: the [eligible entry](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-duck-fork-k-playing) (Duck fork, K playing) and the four on the dfranzen base: [control](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-control), [K playing](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-k-playing), [K advising](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-k-advising), [last resort](https://www.kaggle.com/code/hralmjc/what-exploration-can-ignore-dfranzen-last-resort)

## The question I started from

ARC-AGI-3 says it evaluates goal-setting. But every game already has a win condition, and the score charges every
action you take. So what can an agent do here? It can identify the goal the game has set, and not choose one. I wanted
to know whether that step, *identifying the win condition*, can be done without spending the actions the score counts,
and whether it reaches the score at all.

To measure it I built a graph explorer (K). It issues revocable no-effect certificates for actions that change nothing,
flags hidden state when a certificate fails or a safe action kills, and ranks goal hypotheses that only order the
candidate actions. Then I put it inside the harness of an open-weight language model agent in three ways: playing,
advising, and as a last resort. The paper is the account of what that layer claimed, what use I could observe, and what
the score comparisons can and cannot establish. Most of what they establish is negative, and that is the result: in 50 of
66 local wins the explorer won without the identifier naming the goal, and no mode produced a measurable score gain.

## Time, machine, and budget

Six weeks, from 29 September to 8 November 2026. One author, with language model agents doing the executive work (code,
tests, benches, drafts) under rules I set and reviewed, as the paper's Disclosure states.

My machine is a small consumer PC without a discrete GPU. That one fact shaped the project. Everything that needed a GPU
ran inside Kaggle's weekly quota, so every phase-A round competed with every other for the same 30 hours, and the local
bench had to be something a CPU could finish overnight: 25 public games, 4 seeds, 1,500 steps, about 35 minutes per arm.
I conjecture that most leading entries adapted the model itself, with hardware beyond the Kaggle allotment. I could not
test that, and I did not try to. The question I could test was narrower: does a layer that *knows things about the game*
help the model that plays it?

## How the work was organized

The coupled agent went through two lines. The Duck line (v1 to v7) sits on Tufa Labs' harness, and its v5 is the eligible
entry. The dfranzen line (v8 to v14.1) sits on the open Milestone 2 notebook. Its last version finished the hidden-set
rerun after the deadline and is reported as a measurement only.

From 27 October on, before the first coupled version, every version followed one protocol. Invariants and claims were
written down first. One model built the version and its tests. A different model, in a fresh session, attacked the
frozen artifact against its hash manifest and returned a verdict with a reproducible proof for each finding. Every defect
came back with a regression test and a mutant, a deliberate reintroduction of the defect that the new test has to catch.
By v14.1 that was 92 tests and 85 mutants. I did not submit any version with an open blocking finding.

The measurements followed the same rule: decide before the number. The reading rule for phase A was fixed after one pilot
round and before the rest: an arm shows an effect only if its mean differs from the base's by more than twice the larger
sample SD of the pair. Whether a round would be read was also decided before its result, in both directions. A control
round that had to rerun because the notebooks' text changed was read, since the batch was still open. A last-resort round
with nine exceptions from my layer was excluded and replaced, by the rule written days earlier. Sources were read in full
before being cited, and the final text was revised block by block, fidelity first, style second.

## What went wrong

- **The first scored submission landed at 0.75.** K took most of the model's turns, and the score charges every one of
  them. I kept the number in the paper as evidence compatible with the explorer's action cost, and I built the advising
  and last-resort modes to cap the intervention in time and in actions. The cap limits the intervention, and not the loss.
- **The model server took 305 to 1,494 seconds to load,** and in four of eight runs the games were released while it was
  still loading. One control round was lost because the server never came up. It was rerun.
- **The last submission did not have its rerun budget.** v14.1 went in on 2 November at 22:30 UTC. Its nine-hour rerun
  ended after the deadline, so it could not be selected. Another team on the same base reports the same miss. The lesson is
  arithmetic: a last submission needs the full rerun time plus queue margin before the selection deadline, and I did not
  have it.
- **Kaggle mounted the notebooks' inputs under two different layouts.** Three pushes of one notebook died before any game
  ran. The public notebooks now resolve both layouts and record which one they found.
- **In the one game I diagnosed, the model never mentioned the advisor's notes** in its visible reasoning. Its costly moves
  came from a code loop it wrote itself, which bypassed the pruned action list. That trace does not let me blame the layer
  for the gap, and it does not let me clear it either.
- **A revision rule applied in bulk created a new tic.** One reviewer, following a style manual, asked to replace every
  contrastive *while* with *whereas*. Done across the paper, that put *whereas* at fifteen times its frequency in pre-2019
  papers. After that, I measured a rule against the corpus before accepting it, and the style guide the agents follow now
  records where the authorities disagree and lets the passage decide.

## What I learned

- **Three public games and three rounds detect only large effects.** On the same day, with the same unmodified notebook,
  one game scored from 21 to 100. With sample SDs above 15 the bar was about 45 points, and no arm came near it.
- **Two batches of the same base, five days apart, differed by about 23 points** (74.91 against 52.30). Phase-A levels do
  not compare across batches, so every comparison in the paper is read inside the batch it was run in.
- **Deciding before the number is what makes a negative result worth publishing.** What counts as a round, how an effect is
  read, which arm is descriptive: all of it was written before the data arrived, and that is why I can report 0.75 and
  50 of 66 without flinching.
- **Red teams run by other models found one class of defect again and again:** places where what the agent asserts, in its
  prompt or its plan, diverges from what the environment permits. The paper offers that list as a checklist for harness work.
- **The explorer wins by search, and the identifier mostly misses.** If goal-setting is to be scored as a component of
  intelligence, the benchmark would have to ask the agent to state the win condition it pursues and score that statement
  apart from action efficiency. Choosing goals, rather than identifying the one a game has set, lies further still, and I
  did not test it.

## Disclosure

Large language model agents from different providers and of different profiles, Western and Eastern, were used
deliberately under a fixed protocol. I set the questions, the hypotheses, and the working rules. The agents implemented
the code, ran the local tests, and drafted the text. I adjudicated every output, ran every Kaggle experiment, and take
responsibility for the content. No language model is an author. The full statement is in the paper.
