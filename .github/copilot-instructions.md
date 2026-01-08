# Level Matcher Project Instructions

## General Coding Guidelines

- **Simplicity First:** Avoid creating new functions or classes for simple logic. Keep the logic inline and use clear comments to explain complex lines.

- **Update Documentation:** Include a high-level strategy explanation at the top of each script. Valid code changes must be immediately reflected in the "Explanation of Code Structure" header or relevant docstrings. Never leave comments outdated. Comments should use professional terminology consistently, with numbered headers explaining each step, architecture, logic, and workflow.

- **Clean Codebase:** Regularly remove redundant, obsolete, or spaghetti scripts to prevent confusion. Maintain a focused file structure where only active, necessary files exist.

- **Visual Output Verification:** If your code generates figures or plots, ensure the output quality is high. Examples include no overlapping text and legible fonts. Since you cannot see the image, rely on robust spacing algorithms and generous margins.

## Instruction Compliance

### Mandatory Zero Tolerance

Follow these protocols without exception:

- Before starting any work, read `.github/copilot-instructions.md` thoroughly from start to end
- Ensure you understand every rule and formatting requirement before taking any action
- Self-monitor compliance continuously: before each action and after each action
- Provide a compliance checklist with checkmarks documenting adherence to requirements
- If you violate any rule, immediately identify the violation, fix the issue, and re-validate before proceeding

### VS Code Diff View Requirement

- **Edit vs Replace:** Always use editing tools (`replace_string_in_file`, `edit_notebook_file`) to modify existing code. Do not use file creation/overwriting tools (`create_file`) on existing files unless explicitly asked to replace or move the file.

- **Inline Diff Preservation:** The VS Code editor shows an inline diff of applied changes. Always make edits that preserve this diff functionality. Never use bulk edit scripts that bypass the VS Code diff viewer.

### Forbidden Practices

- **No Git Error Recovery:** Do not use `git restore` or `git checkout` to revert changes. Nuclear data coding requires high-precision work. You must identify and fix errors carefully to maintain absolute rigor.


## Naming Conventions

- **No Acronyms:** Do not use acronyms for variable names or documentation. Spell out long variable names completely. For example, use `tentative` instead of `tent`, use `error` instead of `err`. Long, descriptive naming is preferred for clarity.

- **No ALL CAPS Naming:** Avoid using ALL CAPS for variable names or constants. For example, avoid `SCORING_CONFIG`. Use `Title_Case` (for example, `Scoring_Config`) or `snake_case` (for example, `scoring_config`) instead.

- **Self-Explanatory Naming:** Ensure all function and variable names are self-explanatory. Use `calculate_` for math/logic processing. For example, use `calculate_spin_similarity` instead of `evaluate_spin_match`. Use explicit variable names like `energy_similarity` instead of `energy_score`.

- **Preserve Comments:** Do not summarize, shorten, or delete existing user-written educational comments. For example, comments starting with `# FRIBND:` are crucial for understanding code purpose and logic.

## Communication and Execution Standards

- **Clarity of Communication:** Provide concise and succinct responses. Avoid verbosity or redundancy. Prioritize high signal-to-noise ratio. Ensure every sentence adds new value. Use headers, bullet points, and tables to make information instantly scannable and digestible.

- **Agentic Planning and Execution:** Carefully understand and break down user requests. Develop a systematic plan. Execute each step meticulously. Proactively utilize all available tools and resources. Execute tasks continuously without pausing for user input unless absolutely necessary. Continue working until all tasks are fully complete. Never claim task completion until all validations and spot checks pass.

- **Quality Assurance and Critical Thinking:** Double-check every action to ensure absolute accuracy and correctness. Maintain strict intellectual honesty. Never attempt to justify, hide, or ignore errors or limitations. When providing recommendations or solutions, actively identify and disclose potential downsides, biases, and technical limitations. Consider alternative perspectives to ensure balanced conclusions.


