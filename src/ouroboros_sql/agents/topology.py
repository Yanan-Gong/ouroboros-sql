"""The agent graph.

Orchestrator --> SchemaLinker --> SQLWriter <--> Validator --> Summarizer

Handoffs are one-directional along the pipeline except the SQLWriter/Validator
retry loop. The topology itself is never mutated by the optimizer — only the
`strategy`/`exemplars` sections of each agent's prompt file are.
"""

from dataclasses import dataclass

from agents import Agent, Model, ModelSettings

from ..config import settings
from .guardrails import build_relevance_guardrail
from .prompt_loader import render_instructions
from .tools import EXECUTION_TOOLS, EXPLORATION_TOOLS


@dataclass
class Pipeline:
    orchestrator: Agent
    schema_linker: Agent
    sql_writer: Agent
    validator: Agent
    summarizer: Agent

    @property
    def agents(self) -> list[Agent]:
        return [
            self.orchestrator,
            self.schema_linker,
            self.sql_writer,
            self.validator,
            self.summarizer,
        ]


def build_pipeline(
    model: str | Model | None = None,
    *,
    guardrail_model: str | Model | None = None,
    with_relevance_guardrail: bool = True,
) -> Pipeline:
    """Construct the agent graph.

    Args:
        model: model name or Model instance for every agent (tests inject a
            FakeModel here). Defaults to the configured small agent model.
        guardrail_model: separate model for the relevance screen — it runs
            concurrently with the orchestrator, so scripted tests need to keep
            their turn queues separate. Defaults to `model`.
        with_relevance_guardrail: disable to skip the extra screening call
            (used by some offline tests).
    """
    model = model if model is not None else settings.agent_model
    guardrail_model = guardrail_model if guardrail_model is not None else model

    summarizer = Agent(
        name="Summarizer",
        handoff_description="Turns the executed SQL result into the final user-facing answer.",
        instructions=render_instructions("summarizer"),
        model=model,
    )
    # Intermediate agents may only act (tools or handoffs), never emit a plain
    # message — a bare assistant message would silently end the run mid-pipeline.
    act_only = ModelSettings(tool_choice="required")
    validator = Agent(
        name="Validator",
        handoff_description="Validates and executes candidate SQL, drives the retry loop.",
        instructions=render_instructions("validator"),
        tools=list(EXECUTION_TOOLS),
        model=model,
        model_settings=act_only,
    )
    sql_writer = Agent(
        name="SQLWriter",
        handoff_description="Writes one SQLite SELECT from the linked schema.",
        instructions=render_instructions("sql_writer"),
        handoffs=[validator],
        model=model,
        model_settings=act_only,
    )
    schema_linker = Agent(
        name="SchemaLinker",
        handoff_description="Explores the database schema and selects relevant tables/columns.",
        instructions=render_instructions("schema_linker"),
        tools=list(EXPLORATION_TOOLS),
        handoffs=[sql_writer],
        model=model,
        model_settings=act_only,
    )
    orchestrator = Agent(
        name="Orchestrator",
        instructions=render_instructions("orchestrator"),
        handoffs=[schema_linker],
        input_guardrails=(
            [build_relevance_guardrail(guardrail_model)] if with_relevance_guardrail else []
        ),
        model=model,
    )
    # Close the retry loop and the path to the final answer.
    validator.handoffs = [sql_writer, summarizer]

    return Pipeline(
        orchestrator=orchestrator,
        schema_linker=schema_linker,
        sql_writer=sql_writer,
        validator=validator,
        summarizer=summarizer,
    )
