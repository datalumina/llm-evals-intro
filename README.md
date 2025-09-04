# Introduction to LLM Evaluations
Evaluation is the cornerstone of building reliable LLM applications. Without structured evaluation, you're essentially hoping that your system works rather than knowing it does. This video introduces a framework for LLM evaluation across three distinct levels, each serving different purposes and providing unique insights.

## The Three Levels of LLM Evaluation

A successful evaluation strategy for AI systems incorporates testing at multiple levels, with each offering different benefits:
- Level 1: Unit Tests - Fast, targeted, assertion-based tests
- Level 2: Model & Human Evaluation - Qualitative assessment through trace analysis
- Level 3: A/B Testing - Production validation with real users

The cost and complexity increases with each level, which dictates how frequently you should deploy them. Typically, you'll run Level 1 evaluations on every code change, Level 2 on a regular cadence, and Level 3 only after significant product changes.

## Resources

- [LLM Alignment Example Excel Sheet](https://docs.google.com/spreadsheets/d/1JXPRPMJDGlEsHbhuEDC1wJpVPwQwGKP-6toC5nNe2Io/edit?gid=0#gid=0)
- [Your AI Product Needs Evals by Hamal Husain](https://hamel.dev/blog/posts/evals/)