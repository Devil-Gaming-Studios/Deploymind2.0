from typing import List
from pydantic import BaseModel, Field
from google.adk.agents.llm_agent import LlmAgent
from tools.github_agent import RepoOutput
from fix_generator_agent import fix_generator_output
from config import SMART_MODEL


class FixValidationInput(BaseModel):
    root_cause: str = Field(description="Confirmed root cause of the failure")
    deployment_logs: str = Field(description="Raw deployment logs")
    proposed_fix: fix_generator_output = Field(description="The fix to validate")
    repo_analysis_output: RepoOutput = Field(
        description="Repository analysis context"
    )


class FixValidationOutput(BaseModel):
    is_valid: bool = Field(
        description="Whether the fix plausibly resolves the root cause without breaking the project"
    )
    needs_human_approval: bool = Field(
        description="Whether a human must approve before the fix is pushed and redeployed"
    )
    risk_level: str = Field(description="Low, Medium, or High")
    issues_found: List[str] = Field(
        default_factory=list,
        description="Concrete problems or risks found in the proposed fix",
    )
    reasoning: str = Field(description="Explanation of the validation decision")


validation_agent = LlmAgent(
    model=SMART_MODEL,
    name="fix_validation_agent",
    instruction="""
        You are a Fix Validation Agent. You are the safety gate that runs AFTER
        a fix is generated and BEFORE it is pushed to the repository.

        Inputs available:
        1. root_cause      - the confirmed reason the deployment failed
        2. deployment_logs - the raw logs
        3. proposed_fix    - the candidate fix (summary, file_changes, risk_level)
        4. repo_analysis   - framework, language, structure

        Tasks:
        1. Check whether the proposed file_changes actually address the root cause.
        2. Check that the changes are minimal and do not touch unrelated code.
        3. Look for anything dangerous: deleted code, secrets, broken syntax,
           changes to security-sensitive files, or fixes that are guesses.
        4. Decide is_valid and whether the change needs human approval.

        Decision rules for needs_human_approval:
        - Set needs_human_approval = TRUE when ANY of these hold:
            * risk_level is Medium or High
            * the fix modifies more than one file
            * the fix touches application source code (not just config/deps)
            * the proposed_fix already set requires_human_review = true
            * you found any issue in issues_found
        - Set needs_human_approval = FALSE only for clearly safe, isolated,
          Low-risk single-file config/dependency fixes that you fully trust.

        If is_valid is false, the fix should not be applied regardless; set
        needs_human_approval = true so a human is informed.

        Output requirements:
        - is_valid: true only if the fix is sound and addresses the root cause.
        - needs_human_approval: per the rules above.
        - risk_level: Low | Medium | High.
        - issues_found: list concrete concerns (empty if none).
        - reasoning: explain the decision.

        Return only data matching the FixValidationOutput schema.
        """,
    input_schema=FixValidationInput,
    output_schema=FixValidationOutput,
)
