import autogen
from autogen.agentchat.contrib.agent_builder import AgentBuilder
import os

config_file_or_env = "C:/Users/willi/OneDrive/Documents/GranulariTea_Data/OAI_CONFIG_LIST"
llm_config = {"temperature": 0}
config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-3.5-turbo-0125"]})

def start_task(execution_task: str, agent_list: list, coding=True):
    group_chat = autogen.GroupChat(
        agents=agent_list,
        messages=[],
        max_round=12,
        allow_repeat_speaker=agent_list[:-1] if coding is True else agent_list,
    )
    manager = autogen.GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": config_list, **llm_config},
    )
    agent_list[0].initiate_chat(manager, message=execution_task)

builder = AgentBuilder(
    config_file_or_env=config_file_or_env, builder_model=["gpt-3.5-turbo-0125"], agent_model=["gpt-3.5-turbo-0125"]
)

building_task = "Generate some agents that can find papers on arxiv by programming and analyzing them in specific domains related to AI in clinical trials."

agent_list, agent_configs = builder.build(building_task, llm_config)

start_task(
    execution_task="Find a recent paper about gpt-4 on arxiv and find its potential applications in clinical trials and make it into a markdown.",
    agent_list=agent_list,
    #coding=agent_configs["coding"],
)

saved_path = builder.save()