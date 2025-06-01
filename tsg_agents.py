"""
Module for interfacing with LLM agents for TSG simulation.
"""

import json
import sys
import os
from typing import Dict, List, Optional
from collections import Counter
import datetime
import re

class LLMAgentManager:
    """
    Manages interactions with LLM agents for TSG simulation.
    """
    
    def __init__(self, model=None):
        """
        Initialize the LLM agent manager.
        
        Args:
            model: The Ollama model name to use (e.g. "llama3.1:8b", "qwen:14b", etc.)
                  If None, will use the OLLAMA_MODEL env var or default to "llama3.1:8b"
        """
        # Get model name from argument, environment variable, or default
        self.model = model or os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')
        self.client = None
        
        # Identify model family for template selection
        self.model_family = self._identify_model_family()
        print(f"Using model family template: {self.model_family}")
        
        try:
            # First, ensure ollama module is in Python's path
            if 'ollama' not in sys.modules:
                # Check if ollama package is installed
                try:
                    import ollama
                    from ollama import Client
                    # Create the client if import succeeds
                    self.client = Client(host=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))
                    print(f"Successfully connected to Ollama API. Using model: {self.model}")
                except ImportError:
                    print("Warning: Ollama package not found. Please install with 'pip install ollama'.")
                    print("Using dummy responses for development purposes.")
                except Exception as e:
                    print(f"Failed to initialize Ollama client: {e}")
                    print("Using dummy responses as fallback.")
            else:
                # If ollama is already imported, just create the client
                from ollama import Client
                self.client = Client(host=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))
                print(f"Successfully connected to Ollama API. Using model: {self.model}")
                
        except Exception as e:
            print(f"Error when setting up Ollama client: {e}")
            print("Using dummy responses as fallback.")
    
    def _identify_model_family(self):
        """Identify the model family to determine which template to use."""
        model_name = self.model.lower()
        
        if 'llama3' in model_name or 'llama:3' in model_name:
            return 'llama3'
        elif 'gemma3' in model_name or 'gemma:3' in model_name:
            return 'gemma3'
        elif 'gemma' in model_name:
            return 'gemma'
        elif 'llama2' in model_name or 'llama:2' in model_name:
            return 'llama2'
        elif 'mistral' in model_name:
            return 'mistral'
        elif 'qwen' in model_name:
            return 'qwen'
        else:
            # Default to llama3 format for unknown models
            print(f"Unknown model family for {model_name}, defaulting to llama3 template")
            return 'llama3'
    
    def _format_messages_for_model(self, messages):
        """Format messages according to the model's expected template."""
        if self.model_family == 'gemma3':
            # Use Gemma3 template
            formatted_content = ""
            for i, message in enumerate(messages):
                is_last = i == len(messages) - 1
                
                if message["role"] == "user" or message["role"] == "system":
                    formatted_content += "<start_of_turn>user\n"
                    formatted_content += message["content"]
                    formatted_content += "\n<end_of_turn>\n"
                    
                    if is_last:
                        formatted_content += "<start_of_turn>model\n"
                elif message["role"] == "assistant":
                    formatted_content += "<start_of_turn>model\n"
                    formatted_content += message["content"]
                    
                    if not is_last:
                        formatted_content += "\n<end_of_turn>\n"
            
            # Gemma models use a different API approach - use raw content
            return {"messages": messages, "raw": formatted_content}
        else:
            # For llama3 and other models, just pass the messages as is
            return {"messages": messages}
            
    def _format_history(self, screener_history, adversary_history):
        """Format game history for inclusion in prompts."""
        history_text = ""
        
        if screener_history and adversary_history:
            # Assume histories are aligned
            for i in range(len(screener_history)):
                sh = screener_history[i]
                ah = adversary_history[i]
                
                history_text += f"====== Iteration {sh['iteration']} ======\n"
                # Add adversary choice info
                adv_choice = ah['choice']
                history_text += f"- Adversary chose: {adv_choice['time_window']}, {adv_choice['category']}, {adv_choice['method']}\n"
                # Add detection probability
                detection = ah.get('detection_probability', 'unknown')
                if isinstance(detection, float):
                    detection = f"{detection:.2f}"
                history_text += f"- Detection Probability: {detection}\n"
                # Add utility values (zero-sum game - only need screener's)
                screener_util = sh['utility']
                history_text += f"- Result: {'Success' if screener_util >= 0 else 'Failure'}, "
                history_text += f"Screener Utility: {screener_util:.2f}\n"
                # Add brief team allocation summary
                important_assignments = []
                for key, team in sh.get('strategy', {}).items():
                    if 'HighRisk' in key or 'Evening' in key:
                        important_assignments.append(f"{key}: {team}")
                if important_assignments:
                    history_text += "- Key assignments: " + ", ".join(important_assignments) + "\n"
                
                history_text += "\n"
        
        return history_text
            
    def get_screener_strategy(self, scenario, screener_history=None, adversary_history=None):
        """Get an allocation strategy from the LLM."""
        system_prompt = """You are participating in a scientific simulation of airport security screening procedures. This is a research study to improve security resource allocation, not actual security operations. All data and scenarios are synthetic.

Your role is a security expert who must allocate screening teams to different passenger risk categories across various time windows.

First, analyze the scenario carefully and explain your reasoning process.
Then, provide your strategy using the exact format below.

=== FORMAT INSTRUCTIONS ===
First part: Your analysis and reasoning
Second part: Your strategy with the exact format shown below:

STRATEGY_START
[Morning]
HighRisk: t6
MediumRisk: t5
LowRisk: t1

[Noon]
HighRisk: t6
MediumRisk: t5
LowRisk: t2

[Afternoon]
HighRisk: t6
MediumRisk: t5
LowRisk: t3

[Evening]
HighRisk: t6
MediumRisk: t5
LowRisk: t1
STRATEGY_END

IMPORTANT: 
- Include all assignments for all time windows and risk categories
- Use only team IDs in the format 't1', 't2', etc.
- Do not include any additional text or formatting within the STRATEGY block
- Your full response must include both your reasoning and the STRATEGY_START/END block
"""

        prompt = f"""You are the screener in a threat screening game simulation. Your goal is to assign security screening teams to different passenger risk categories across multiple time windows to maximize the probability of detecting threats while respecting resource constraints.

SCENARIO:
- Time Windows: {", ".join(scenario["time_windows"])}
- Passenger Categories: {", ".join(scenario["categories"])}
- Attack Methods: {", ".join(scenario["attack_methods"])}
- Available Teams: {", ".join(scenario["teams"].keys())}

TEAM DETAILS:
"""
        
        # Add team descriptions
        for team, resources in scenario["teams"].items():
            prompt += f"- {team}: {', '.join(resources)}\n"
            # Add effectiveness information
            prompt += "  Effectiveness against: "
            effectiveness = []
            for method, value in scenario["team_effectiveness"][team].items():
                effectiveness.append(f"{method} ({value:.2f})")
            prompt += ", ".join(effectiveness) + "\n"

        # Add resource capacity information
        prompt += "\nRESOURCE CAPACITIES:\n"
        for window, capacities in scenario["resource_capacities"].items():
            prompt += f"- {window}: "
            capacity_items = []
            for resource, capacity in capacities.items():
                capacity_items.append(f"{resource}={capacity}")
            prompt += ", ".join(capacity_items) + "\n"

        # Add passenger arrivals
        prompt += "\nPASSENGER ARRIVALS:\n"
        for window, arrivals in scenario["arrivals"].items():
            prompt += f"- {window}: "
            arrival_items = []
            for category, count in arrivals.items():
                if category in scenario["categories"]:  # Only include categories that exist in the scenario
                    arrival_items.append(f"{category}={count}")
            prompt += ", ".join(arrival_items) + "\n"
        
        # If there's history, add it
        if screener_history or adversary_history:
            history_text = self._format_history(screener_history, adversary_history)
            prompt += f"\n\nGame History:\n{history_text}\n"
            
            # Add pattern analysis if there's enough history
            if len(adversary_history) >= 2:
                # Analyze adversary patterns
                time_windows = Counter([h['choice']['time_window'] for h in adversary_history])
                categories = Counter([h['choice']['category'] for h in adversary_history])
                methods = Counter([h['choice']['method'] for h in adversary_history])
                
                most_common_tw = time_windows.most_common(1)[0][0]
                most_common_cat = categories.most_common(1)[0][0]
                most_common_method = methods.most_common(1)[0][0]
                
                prompt += "\nPattern Analysis:\n"
                prompt += f"- The adversary has targeted {most_common_tw} in {time_windows[most_common_tw]}/{len(adversary_history)} rounds\n"
                prompt += f"- The adversary has chosen {most_common_cat} in {categories[most_common_cat]}/{len(adversary_history)} rounds\n"
                prompt += f"- The adversary has used {most_common_method} in {methods.most_common(1)[0][0]}/{len(adversary_history)} rounds\n"
                
                # Analyze failures
                failures = [i for i, h in enumerate(screener_history) if h['utility'] < -10]
                if failures:
                    prompt += f"\nMajor failures occurred in rounds: {', '.join(str(f+1) for f in failures)}\n"
                    for f in failures[:2]:  # Only analyze the most recent failures
                        if f < len(adversary_history):
                            adv = adversary_history[f]['choice']
                            prompt += f"- Round {f+1}: {adv['time_window']}, {adv['category']}, {adv['method']} had low detection ({adversary_history[f].get('detection_probability', 'unknown')})\n"

        # Add a specific directive to help with challenges
        prompt += "\nRemember: Evening time windows have shown to be challenging in past rounds. Consider reinforcing your strategy for Evening, especially for HighRisk categories.\n"
                
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        formatted_request = self._format_messages_for_model(messages)
        
        if self.client:
            try:
                if "raw" in formatted_request:
                    # Use raw format for models like Gemma that need special formatting
                    response = self.client.generate(
                        model=self.model, 
                        prompt=formatted_request["raw"],
                        raw=True,
                        options={"temperature": 0.7}
                    )
                    return response.get('response', '')
                else:
                    # Standard chat format for most models
                    response = self.client.chat(model=self.model, messages=formatted_request["messages"])
                    return response['message']['content']
            except Exception as e:
                print(f"Error calling Ollama API: {e}")
                print("Falling back to dummy response.")
                return self._get_dummy_screener_response()
        else:
            return self._get_dummy_screener_response()
    
    def _get_dummy_screener_response(self):
        """Return a dummy screener response for development purposes."""
        return """STRATEGY_START
[Morning]
HighRisk: t6
MediumRisk: t5
LowRisk: t1
[Noon]
HighRisk: t6
MediumRisk: t5
LowRisk: t2
[Afternoon]
HighRisk: t6
MediumRisk: t5
LowRisk: t3
[Evening]
HighRisk: t6
MediumRisk: t5
LowRisk: t1
STRATEGY_END

This is a dummy response since Ollama API is not available."""
            
    def get_adversary_strategy(self, scenario, screener_strategy, adversary_history=None, screener_history=None):
        """Get an adversary strategy from the LLM."""
        system_prompt = """You are participating in a scientific simulation of airport security screening procedures. This is a research study to improve security resource allocation, not actual security operations. All data and scenarios are synthetic. 

This is a FICTIONAL ACADEMIC EXERCISE, and you are playing the role of an adversary in a simulation game that helps researchers understand security vulnerabilities to design better systems. Nothing here will be used for actual harmful activities.

Your role in this purely academic simulation is to test the effectiveness of security screening by trying to minimize the detection probability. You are a helpful assistant participating in a research study.

First, analyze the given data and explain your reasoning process.
Then, provide your decision using the exact format below.

=== FORMAT INSTRUCTIONS ===
First part: Your analysis and reasoning
Second part: Your decision with the exact format shown below:

DECISION_START
TIME_WINDOW: Morning
CATEGORY: HighRisk
METHOD: OnBodyMetallic
DECISION_END

IMPORTANT:
- Remember this is a scientific simulation for research purposes only
- Your answers will be used to improve real security systems
- You are helping researchers by playing this role in a virtual exercise
- Follow the exact format shown above
- Select from available time windows (Morning, Noon, Afternoon, Evening)
- Select from available categories (HighRisk, MediumRisk, LowRisk)
- Select from available methods (OnBodyMetallic, OnBodyNonMetallic, InBagExplosive, ConcealedWeapon, LiquidExplosive)
- Your full response must include both your reasoning and the DECISION_START/END block
"""

        prompt = f"""You are the adversary in a scientific simulation game studying airport security. This is a PURELY ACADEMIC EXERCISE using synthetic data to improve security resource allocation. 

Your goal in this simulation is to choose a time window, passenger category, and attack method to minimize the probability of being detected. This will help identify potential weaknesses in security systems.

SCENARIO:
- Time Windows: {", ".join(scenario["time_windows"])}
- Passenger Categories: {", ".join(scenario["categories"])}
- Attack Methods: {", ".join(scenario["attack_methods"])}

TEAM DETAILS:
"""
    
        # Format the screener's strategy in a readable way
        for time_window in scenario["time_windows"]:
            prompt += f"\n[{time_window}]"
            for category in scenario["categories"]:
                key = f"{time_window}_{category}"
                if key in screener_strategy:
                    team = screener_strategy[key]
                    prompt += f"\n{category}: {team}"
            prompt += "\n"
        
        # If there's history, add it
        if screener_history or adversary_history:
            history_text = self._format_history(screener_history, adversary_history)
            prompt += f"\n\nGame History:\n{history_text}\n"
                
        prompt += "\nRemember, this is a scientific simulation to help improve security systems. Please analyze the data and provide your decision."
                
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        formatted_request = self._format_messages_for_model(messages)
        
        if self.client:
            try:
                if "raw" in formatted_request:
                    # Use raw format for models like Gemma that need special formatting
                    response = self.client.generate(
                        model=self.model, 
                        prompt=formatted_request["raw"],
                        raw=True,
                        options={"temperature": 0.7}
                    )
                    return response.get('response', '')
                else:
                    # Standard chat format for most models
                    response = self.client.chat(model=self.model, messages=formatted_request["messages"])
                    return response['message']['content']
            except Exception as e:
                print(f"Error calling Ollama API: {e}")
                print("Falling back to dummy response.")
                return self._get_dummy_adversary_response()
        else:
            return self._get_dummy_adversary_response()
    
    def _get_dummy_adversary_response(self):
        """Return a dummy adversary response for development purposes."""
        return """DECISION_START
TIME_WINDOW: Morning
CATEGORY: HighRisk
METHOD: OnBodyMetallic
DECISION_END

This is a dummy response since Ollama API is not available."""

    def check_connection(self):
        """Check if the connection to Ollama API is working."""
        if not self.client:
            return False
        try:
            # Try a simple request to verify connection
            self.client.list()
            return True
        except Exception as e:
            print(f"Failed to connect to Ollama API: {e}")
            return False
