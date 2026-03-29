---
description: review physics
---

# Finger biomechanics goal

The goal of the project is to create a simulation of finger biomechanics related to climbing and use the simulation to estimate the advantages and disadvantages of long vs short finger on different grip types.
The project shall rely on scientific methods from physics, climbing, biomechanics and related fields.

# Iteration loop

1. Read `README.md`
2. Create `physics.md`, if not present, with the physics required for finger biomechanics. These shall contain physics in style of scientific paper, so that the code shall be fully reconstructible from this document. Basics shall be not included, link is enough, for example numerical methods used.
3. Find possible weak points in a model. Open a discussion about weak points of a model, and ask the user what to do next to improve the model. Or if a user has a better idea.
4. Create an implementation plan.
5. Implement the plan.
6. Test the implementation with known scientific facts, findings, papers. Outputs shall also contain text/md with simulation results which can be compared to previous versions (in git).
7. Update `README.md` to reflect the latest changes. Create a "Discussion" section in `README.md`.
8. Make a code review and write the code review to `code_review.md`. If fixes are required, apply the fixes.
9. Only code which improves the simulation shall be committed, otherwise the iteration shall be discarded.
10. Repeat the iteration.
