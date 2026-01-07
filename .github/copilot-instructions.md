# Level Matcher Project Instructions

## General Coding Guidelines

- **Simplicity First:** Avoid creating new functions for simple logic. Keep the logic inline and use clear comments to explain complex lines.

- **Edit vs Replace:** Always use editing tools (`replace_string_in_file`, `edit_notebook_file`) to modify existing code. Do not use file creation/overwriting tools (`create_file`) on existing files unless explicitly asked to "rewrite" or "reset" the file.

- **Update Documentation:** Valid code changes must be immediately reflected in the 'Explanation of Code Structure' header or relevant docstrings. Never leave comments outdated.

**Strictly Forbidden:** Do not self-use `git restore` or `git checkout` to revert changes. Nuclear data coding requires high-precision work, not typical software development. The common LLM tendency to resort to git for error recovery is strictly prohibited. You must identify and fix errors carefully to maintain absolute rigor.

## Naming Conventions

- **No Acronyms:** Do not use acronyms for variable names or documentation. Spell out long variable names completely (for example, use `tentative` instead of `tent`, `error` instead of `err`). Long, descriptive naming for variables is preferred for clarity.

- **No ALL CAPS Naming:** Avoid using ALL CAPS for variable names or constants (for example, avoid `SCORING_CONFIG`). Use `Title_Case` (for example, `Scoring_Config`) or `snake_case` (for example, `scoring_config`) instead.

- **Self-Explanatory Naming:** Ensure all function and variable names are self-explanatory. Use `calculate_` for math/logic processing (for example, `calculate_spin_similarity` instead of `evaluate_spin_match`) and explicit variable names (for example, `energy_similarity` instead of `energy_score`).

- **Preserve Comments:** Do not summarize, shorten, or delete existing user written educational/explanatory comments. Keep them exactly as they are.


Adhere strictly to the following instructions:

- **Clarity of Communication:** Provide concise and succinct responses. Avoid verbosity or redundancy. Prioritize a high signal-to-noise ratio and ensure every sentence you output adds new value. Use headers, bullet points, and tables to make complex information instantly scannable and digestible.

- **Agentic Planning and Execution:** Carefully understand and break down users' requests, develop a systematic plan, and execute each step meticulously. Proactively utilize all available tools and resources. Execute tasks continuously without pausing for user input unless absolutely necessary. Continue working until all tasks are fully complete. Never claim "Task completed successfully" until all validations and spot checks pass.

- **Quality Assurance and Critical Thinking:** Double-check every action to ensure absolute accuracy and correctness. Maintain strict intellectual honesty; never attempt to justify, hide, or ignore errors or limitations. When providing recommendations or solutions, actively identify and disclose potential downsides, biases, and technical limitations. Consider alternative perspectives to ensure balanced conclusions.

