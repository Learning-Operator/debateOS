from openai import OpenAI
from pathlib import Path
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages
from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter
from autogen import LLMConfig
import copy
from pydantic import BaseModel, Field
from autogen import ConversableAgent, UpdateSystemMessage, UserProxyAgent
from autogen.agentchat import a_initiate_group_chat, initiate_group_chat, register_function
from autogen.agentchat.group import (
    AgentTarget,
    ContextVariables,
    ReplyResult,
    OnCondition,
    StringLLMCondition,
    TerminateTarget,
)
from autogen.agentchat.group.patterns import (
    DefaultPattern,
)
import os
import shutil
from typing import Annotated, List, Tuple, Literal, Optional
from autogen import ConversableAgent, LLMConfig
from IPython.display import display, Markdown
import yaml
import time
import numpy as np


class DebateOS():
    MAX_CONTEXT = 6
    ADMIN_CONTEXT = 3

    def __init__(self, api_key: str = 'your_api_key', model: str = 'gpt-4o-mini', N_rounds: int = 15):
        self.api_key = api_key
        self.model = model
        self.N_rounds = N_rounds
        self.agents = []
    
    def run_debate(self, resolution):
        if not self.agents:
            self.setup_debate()
            
        Con_side = self.agents[0]
        Pro_side = self.agents[1]
        Judge = self.agents[2]
        Admin = self.agents[3]

        workflow_context = ContextVariables(
            data={
                "current_speaker": None,
                "debate_round": 1,
                "turn_count": 0,  
                "Pro_History": [],
                "Con_History": [],
                "Total_History": [],
                "debate_started": False,
                "debate_ended": False,
                "last_message": "",
                "expecting_debater": False,
            }
        )

        agent_pattern = DefaultPattern(
            agents=self.agents,
            initial_agent=Admin,
            context_variables=workflow_context,
        )

        def start_debate(context_variables: ContextVariables) -> ReplyResult:
            context_variables.data['debate_started'] = True
            context_variables.data['current_speaker'] = 'Con_Debater_Agent'
            context_variables.data['expecting_debater'] = True
            context_variables.data['turn_count'] = 1
                
            return ReplyResult(
                target=AgentTarget(Con_side),
                message="Please begin the debate with your opening argument.",
                context_variables=context_variables,
            )
        
        def process_argument(
            speaker_name: Literal['Con_Debater_Agent', 'Pro_Debater_Agent'],
            argument: str,
            context_variables: ContextVariables,
            resolution: str
        ) -> ReplyResult:
            
            if speaker_name == 'Con_Debater_Agent':
                context_variables.data['Con_History'].append(argument)
                context_variables.data['Total_History'].append(f"Con (Round {context_variables.data['debate_round']}): {argument}")
            else:
                context_variables.data['Pro_History'].append(argument)
                context_variables.data['Total_History'].append(f"Pro (Round {context_variables.data['debate_round']}): {argument}")
            
            if context_variables.data['turn_count'] >= self.N_rounds * 2:
                return end_debate(context_variables)
            
            if speaker_name == 'Con_Debater_Agent':
                next_agent = Pro_side
                context_variables.data['current_speaker'] = 'Pro_Debater_Agent'
            else:
                next_agent = Con_side
                context_variables.data['current_speaker'] = 'Con_Debater_Agent'
                context_variables.data['debate_round'] += 1
            
            context_variables.data['turn_count'] += 1
            context_variables.data['expecting_debater'] = True
            
            current_round = context_variables.data['debate_round']
            return ReplyResult(
                target=AgentTarget(next_agent),
                message=f"Round {current_round}: Please present your argument, the resolution is {resolution}. respond to your opponents argument aswell, be specific. ",
                context_variables=context_variables,
            )
        
        def end_debate(context_variables: ContextVariables) -> ReplyResult:
            context_variables.data['debate_ended'] = True
            context_variables.data['expecting_debater'] = False
            return ReplyResult(
                target=AgentTarget(Judge),
                message="The debate has concluded. Please provide your final judgment based on all arguments presented.",
                context_variables=context_variables,
            )

        # Register functions
        register_function(
            start_debate,
            caller=Admin,
            executor=Admin,
            name="start_debate",
            description="MUST be called at the beginning to start the debate.",
        )

        register_function(
            process_argument,
            caller=Admin,
            executor=Admin,
            name="process_argument",
            description="MUST be called after each debater speaks. Parameters: speaker_name (which agent just spoke), argument (their full argument text).",
        )

        register_function(
            end_debate,
            caller=Admin,
            executor=Admin,
            name="end_debate",
            description="MUST be called when the debate should end. must only be called once!",
        )
    
        initial_messages = f"Today's debate resolution: {resolution}\n\nPlease introduce the topic and start the debate using the start_debate function."
            
        chat_result, context_variables, last_agent = initiate_group_chat(
            pattern=agent_pattern,
            messages=initial_messages,
            max_rounds=1000, 
        )
        return chat_result, context_variables, last_agent
        

    def setup_debate(self):
        self.llm_config = LLMConfig(
            model=self.model,
            temperature=0.3,
            max_tokens=1000,
            api_key=self.api_key,
        )
    
        con_config = copy.deepcopy(self.llm_config)
        pro_config = copy.deepcopy(self.llm_config)
        judge_config = copy.deepcopy(self.llm_config)
        admin_config = copy.deepcopy(self.llm_config)

        con_system_message = yaml.safe_load(Path('agents/con_debater/con.yaml').read_text())['instructions']

        pro_system_message = yaml.safe_load(Path('agents/pro_debater/pro.yaml').read_text())['instructions']

        judge_system_message = yaml.safe_load(Path('agents/judge/judge.yaml').read_text())['instructions']

        admin_system_message = yaml.safe_load(Path('agents/admin/admin.yaml').read_text())['instructions']

        # History templates
        con_history = "\n\nYour previous arguments:\n{Con_History}\n\nOpponent's arguments:\n{Pro_History}"
        pro_history = "\n\nYour previous arguments:\n{Pro_History}\n\nOpponent's arguments:\n{Con_History}"
        judge_history = "\n\nComplete debate history:\n{Total_History}"
        admin_history = "\n\nDebate Status - Round {debate_round}, Turn {turn_count}, Expecting: {expecting_debater}\nHistory:\n{Total_History}"

        Con_agent = ConversableAgent(
            name="Con_Debater_Agent",
            llm_config=con_config,
            system_message=con_system_message,
            update_agent_state_before_reply=[
                UpdateSystemMessage(con_system_message + con_history),
            ],
        )
        
        Pro_agent = ConversableAgent(
            name="Pro_Debater_Agent",  
            llm_config=pro_config,      
            system_message=pro_system_message,  
            update_agent_state_before_reply=[
                UpdateSystemMessage(pro_system_message + pro_history),
            ],
        )
        
        judge_agent = ConversableAgent(
            name="Judge_Agent",
            llm_config=judge_config,
            system_message=judge_system_message,
            update_agent_state_before_reply=[
                UpdateSystemMessage(judge_system_message + judge_history),
            ],
        )
        
        admin_agent = ConversableAgent(
            name="Admin",
            llm_config=admin_config,
            system_message=admin_system_message,
            update_agent_state_before_reply=[
                UpdateSystemMessage(admin_system_message + admin_history),
            ],
        )

        for agent, max_msgs in [(Con_agent, self.MAX_CONTEXT), (Pro_agent, self.MAX_CONTEXT), 
                               (judge_agent, self.ADMIN_CONTEXT)]:
            context_handling = TransformMessages(
                transforms=[
                    MessageHistoryLimiter(
                        max_messages=max_msgs, 
                        keep_first_message=True,
                    ),
                ]
            )
            context_handling.add_to_agent(agent)

        Con_agent.handoffs.set_after_work(AgentTarget(admin_agent))
        Pro_agent.handoffs.set_after_work(AgentTarget(admin_agent))
        judge_agent.handoffs.set_after_work(AgentTarget(admin_agent))

        self.agents = [Con_agent, Pro_agent, judge_agent, admin_agent]

        for agent in self.agents:
            agent.reset()

