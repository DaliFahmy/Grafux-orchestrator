"""app.prompts — centralised prompt management for Grafux-orchestrator.

All system prompts live in configs/Msg_config (section-based format).
The universal block JSON schema lives in configs/json_format_schema.json.

Usage:
    from app.prompts import get_system_prompt, get_json_schema

    system_msg = get_system_prompt("chat_assistant")
    schema     = get_json_schema()
    # section names: chat_assistant, create_topic, create_component,
    #   create_cmd, create_tool, create_procedure, run_search,
    #   build_app, build_app_modify, build_app_local_only,
    #   build_app_new_only, apply, voice_block_control,
    #   search_step1, link_constraint, format_step2_instruction,
    #   format_step2_topics, format_step2_commands,
    #   format_step2_components, format_step2_procedures, common
"""
from app.prompts.loader import get_json_schema, get_system_prompt, list_sections

__all__ = ["get_system_prompt", "get_json_schema", "list_sections"]
