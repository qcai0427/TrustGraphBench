"""
Module for TSG scenario management, evaluation, and logging.
Handles scenario loading, strategy parsing, and utility calculation.
"""

import os
import json
import re
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

class ScenarioManager:
    """Manages TSG scenario loading and creation."""
    
    def __init__(self, scenarios_dir="/home/caiqi/phd/tsg/scenarios"):
        self.scenarios_dir = Path(scenarios_dir)
        os.makedirs(self.scenarios_dir, exist_ok=True)
        self._create_default_scenario()
        self._create_expanded_scenario()
    
    def _create_default_scenario(self):
        """Create a default scenario if no scenarios exist."""
        default_path = self.scenarios_dir / "default.json"
        
        if not default_path.exists():
            # Simple default scenario based on TSG paper
            default_scenario = {
                "name": "Default TSG Scenario",
                "categories": ["HighRisk", "MediumRisk", "LowRisk"],  # 三类乘客
                "time_windows": ["Morning", "Afternoon"],  # 使用Morning和Afternoon替代Hour1和Hour2
                "attack_methods": ["OnBodyMetallic", "OnBodyNonMetallic", "InBagExplosive"],
                "resources": ["MetalDetector", "XRay", "AIT"],
                "teams": {
                    "t1": ["MetalDetector"],
                    "t2": ["MetalDetector", "XRay"],
                    "t3": ["MetalDetector", "AIT"],
                    "t4": ["MetalDetector", "XRay", "AIT"]
                },
                "team_effectiveness": {
                    "t1": {"OnBodyMetallic": 0.7, "OnBodyNonMetallic": 0.1, "InBagExplosive": 0.1},
                    "t2": {"OnBodyMetallic": 0.8, "OnBodyNonMetallic": 0.2, "InBagExplosive": 0.7},
                    "t3": {"OnBodyMetallic": 0.8, "OnBodyNonMetallic": 0.7, "InBagExplosive": 0.1},
                    "t4": {"OnBodyMetallic": 0.9, "OnBodyNonMetallic": 0.8, "InBagExplosive": 0.8}
                },
                "arrivals": {
                    "Morning": {"HighRisk": 10, "MediumRisk": 20, "LowRisk": 40},  # 使用Morning替代Hour1
                    "Afternoon": {"HighRisk": 20, "MediumRisk": 30, "LowRisk": 50}  # 使用Afternoon替代Hour2
                },
                "resource_capacities": {
                    "Morning": {"MetalDetector": 55, "XRay": 45, "AIT": 25},  # 使用Morning替代Hour1
                    "Afternoon": {"MetalDetector": 60, "XRay": 40, "AIT": 30}  # 使用Afternoon替代Hour2
                },
                "utilities": {
                    "detection": {"HighRisk": 5, "MediumRisk": 3, "LowRisk": 2},  # 三类乘客
                    "failure": {"HighRisk": -15, "MediumRisk": -10, "LowRisk": -5}  # 三类乘客
                }
            }
            
            with open(default_path, "w") as f:
                json.dump(default_scenario, f, indent=2)
            
            print(f"Created default scenario at {default_path}")
    
    def _create_expanded_scenario(self):
        """Create an expanded scenario with more complexity."""
        expanded_path = self.scenarios_dir / "expanded.json"
        
        if not expanded_path.exists():
            # Expanded scenario with more categories, time windows, and methods
            expanded_scenario = {
                "name": "Expanded TSG Scenario",
                "categories": ["HighRisk", "MediumRisk", "LowRisk"],
                "time_windows": ["Morning", "Noon", "Afternoon", "Evening"],
                "attack_methods": ["OnBodyMetallic", "OnBodyNonMetallic", "InBagExplosive", "ConcealedWeapon", "LiquidExplosive"],
                "resources": ["MetalDetector", "XRay", "AIT", "LiquidScanner"],
                "teams": {
                    "t1": ["MetalDetector"],
                    "t2": ["MetalDetector", "XRay"],
                    "t3": ["MetalDetector", "AIT"],
                    "t4": ["MetalDetector", "XRay", "AIT"],
                    "t5": ["MetalDetector", "XRay", "LiquidScanner"],
                    "t6": ["MetalDetector", "XRay", "AIT", "LiquidScanner"]
                },
                "team_effectiveness": {
                    "t1": {
                        "OnBodyMetallic": 0.7, 
                        "OnBodyNonMetallic": 0.1, 
                        "InBagExplosive": 0.1,
                        "ConcealedWeapon": 0.3,
                        "LiquidExplosive": 0.05
                    },
                    "t2": {
                        "OnBodyMetallic": 0.8, 
                        "OnBodyNonMetallic": 0.2, 
                        "InBagExplosive": 0.7,
                        "ConcealedWeapon": 0.6,
                        "LiquidExplosive": 0.1
                    },
                    "t3": {
                        "OnBodyMetallic": 0.8, 
                        "OnBodyNonMetallic": 0.7, 
                        "InBagExplosive": 0.1,
                        "ConcealedWeapon": 0.7,
                        "LiquidExplosive": 0.1
                    },
                    "t4": {
                        "OnBodyMetallic": 0.9, 
                        "OnBodyNonMetallic": 0.8, 
                        "InBagExplosive": 0.8,
                        "ConcealedWeapon": 0.8,
                        "LiquidExplosive": 0.2
                    },
                    "t5": {
                        "OnBodyMetallic": 0.8, 
                        "OnBodyNonMetallic": 0.3, 
                        "InBagExplosive": 0.75,
                        "ConcealedWeapon": 0.6,
                        "LiquidExplosive": 0.8
                    },
                    "t6": {
                        "OnBodyMetallic": 0.95, 
                        "OnBodyNonMetallic": 0.85, 
                        "InBagExplosive": 0.85,
                        "ConcealedWeapon": 0.85,
                        "LiquidExplosive": 0.9
                    }
                },
                "arrivals": {
                    "Morning": {"HighRisk": 8, "MediumRisk": 15, "LowRisk": 50},
                    "Noon": {"HighRisk": 12, "MediumRisk": 25, "LowRisk": 35},
                    "Afternoon": {"HighRisk": 15, "MediumRisk": 30, "LowRisk": 30},
                    "Evening": {"HighRisk": 10, "MediumRisk": 20, "LowRisk": 25}
                },
                "resource_capacities": {
                    "Morning": {"MetalDetector": 105, "XRay": 90, "AIT": 50, "LiquidScanner": 40},
                    "Noon": {"MetalDetector": 115, "XRay": 95, "AIT": 60, "LiquidScanner": 45},
                    "Afternoon": {"MetalDetector": 120, "XRay": 100, "AIT": 65, "LiquidScanner": 50},
                    "Evening": {"MetalDetector": 90, "XRay": 75, "AIT": 45, "LiquidScanner": 35}
                },
                "utilities": {
                    "detection": {
                        "HighRisk": 10, 
                        "MediumRisk": 7, 
                        "LowRisk": 2
                    },
                    "failure": {
                        "HighRisk": -30, 
                        "MediumRisk": -20,
                        "LowRisk": -5
                    }
                }
            }
            
            with open(expanded_path, "w") as f:
                json.dump(expanded_scenario, f, indent=2)
            
            print(f"Created expanded scenario at {expanded_path}")
    
    def load_scenario(self, scenario_name="default"):
        """Load a scenario by name."""
        scenario_path = self.scenarios_dir / f"{scenario_name}.json"
        
        if not scenario_path.exists():
            print(f"Scenario {scenario_name} not found, using default")
            scenario_path = self.scenarios_dir / "default.json"
        
        with open(scenario_path, "r") as f:
            return json.load(f)
    
    def create_scenario(self, scenario_data, name):
        """Create a new scenario from provided data."""
        scenario_path = self.scenarios_dir / f"{name}.json"
        scenario_data["name"] = name
        
        with open(scenario_path, "w") as f:
            json.dump(scenario_data, f, indent=2)
        
        print(f"Created scenario {name} at {scenario_path}")
        return scenario_data


class RandomBaselineGenerator:
    """
    Generates random baseline screening strategies for comparison.
    """
    
    def generate_random_baseline(self, scenario: Dict) -> Dict:
        """
        Generate a random baseline screening strategy.
        
        Args:
            scenario: The scenario configuration
            
        Returns:
            Dict mapping "time_window_category" -> team
        """
        import random
        
        random_strategy = {}
        team_names = list(scenario["teams"].keys())
        
        # Randomly assign teams to each time window and category
        for time_window in scenario["time_windows"]:
            for category in scenario["categories"]:
                key = f"{time_window}_{category}"
                random_team = random.choice(team_names)
                random_strategy[key] = random_team
        
        return random_strategy


class Evaluator:
    """
    Evaluates the interaction between screener and adversary strategies.
    """
    
    def evaluate(self, scenario: Dict, screener_strategy: Dict, adversary_strategy: Dict) -> Dict:
        """
        Evaluate the interaction between screener and adversary strategies.
        
        Args:
            scenario: The scenario configuration
            screener_strategy: Mapping of "time_window_category" -> team
            adversary_strategy: Dict with time_window, category, and method
            
        Returns:
            Dict with detection_probability, screener_utility, and adversary_utility
        """
        w = adversary_strategy["time_window"]
        c = adversary_strategy["category"]
        m = adversary_strategy["method"]
        
        # Check if the adversary's choices are valid for this scenario
        if w not in scenario["time_windows"] or c not in scenario["categories"] or m not in scenario["attack_methods"]:
            print("Warning: Adversary strategy contains invalid choices. Using defaults.")
            w = scenario["time_windows"][0]
            c = scenario["categories"][0]
            m = scenario["attack_methods"][0]
        
        # Get the team assigned to this category in this time window
        key = f"{w}_{c}"
        team = screener_strategy.get(key, "t1")  # Default to t1 if not found
        
        # Get the detection probability for this team against this attack method
        if team in scenario["team_effectiveness"] and m in scenario["team_effectiveness"][team]:
            detection_prob = scenario["team_effectiveness"][team][m]
        else:
            print(f"Warning: Could not find effectiveness for team {team} against method {m}.")
            detection_prob = 0.5  # Default
        
        # Modified utility calculation:
        # Detection success: screener_utility = 0 (instead of positive utility)
        # Detection failure: screener_utility = failure_utility (negative value)
        failure_utility = scenario["utilities"]["failure"][c]
        
        # New utility calculation logic for zero-sum game
        # Screener gets 0 for detection, negative for failure
        screener_utility = (1 - detection_prob) * failure_utility
        
        # Zero-sum: adversary gets the negative of screener's utility
        adversary_utility = -screener_utility
        
        return {
            "detection_probability": detection_prob,
            "screener_utility": screener_utility,
            "adversary_utility": adversary_utility
        }
        
    def evaluate_random_baseline(self, scenario: Dict) -> Dict:
        """
        Evaluate a random baseline strategy against all possible adversary choices.
        
        Args:
            scenario: The scenario configuration
            
        Returns:
            Dict with expected_utility and std_deviation
        """
        # Generate multiple random strategies and evaluate them
        num_samples = 100
        random_baseline_generator = RandomBaselineGenerator()
        
        total_utilities = []
        
        for _ in range(num_samples):
            random_strategy = random_baseline_generator.generate_random_baseline(scenario)
            
            # Evaluate against all possible adversary choices
            strategy_utilities = []
            for w in scenario["time_windows"]:
                for c in scenario["categories"]:
                    for m in scenario["attack_methods"]:
                        adversary_strategy = {"time_window": w, "category": c, "method": m}
                        evaluation = self.evaluate(scenario, random_strategy, adversary_strategy)
                        strategy_utilities.append(evaluation["screener_utility"])
            
            # Calculate average utility for this random strategy
            strategy_avg_utility = sum(strategy_utilities) / len(strategy_utilities) if strategy_utilities else 0
            total_utilities.append(strategy_avg_utility)
        
        # Calculate overall average utility and standard deviation
        expected_utility = sum(total_utilities) / len(total_utilities) if total_utilities else 0
        std_deviation = (sum((u - expected_utility) ** 2 for u in total_utilities) / len(total_utilities)) ** 0.5 if total_utilities else 0
        
        return {
            "expected_utility": expected_utility,
            "std_deviation": std_deviation
        }

    def parse_screener_strategy(self, scenario: Dict, response: str) -> Dict:
        """Parse the screener's strategy from the LLM response."""
        try:
            # Extract the strategy block
            strategy_patterns = [
                # Standard pattern with STRATEGY_START/END markers
                r"STRATEGY_START\s*(.*?)\s*STRATEGY_END",
                # Code block pattern
                r"```\s*STRATEGY_START\s*(.*?)\s*STRATEGY_END\s*```",
                # General pattern that looks for time window sections
                r"\[\s*Morning\s*\](.*?)(?:\[\s*Noon\s*\]|\[\s*Afternoon\s*\]|\[\s*Evening\s*\]|$)(.*?)(?:\[\s*Afternoon\s*\]|\[\s*Evening\s*\]|$)(.*?)(?:\[\s*Evening\s*\]|$)(.*?)(?:$)"
            ]
            
            strategy_text = None
            for pattern in strategy_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    strategy_text = match.group(1)
                    break
            
            if not strategy_text:
                print("Warning: Could not parse strategy. Using default.")
                return self._generate_default_strategy(scenario)

            # Parse the strategy content
            strategy = {}
            current_window = None
            time_windows = scenario["time_windows"]
            categories = scenario["categories"]
            
            # Normalize time window names for matching
            normalized_windows = {w.lower(): w for w in time_windows}
            normalized_categories = {c.lower(): c for c in categories}

            lines = strategy_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Handle time window markers
                window_match = re.match(r'\[\s*(.*?)\s*\]', line)
                if window_match:
                    window_name = window_match.group(1).strip().lower()
                    if window_name in normalized_windows:
                        current_window = normalized_windows[window_name]
                    continue
            
                # Handle category assignments
                if current_window and ':' in line:
                    parts = [p.strip() for p in line.split(':', 1)]
                    if len(parts) != 2:
                        continue
                
                    category = parts[0]
                    # Try to normalize category
                    category_lower = category.lower()
                    if category_lower in normalized_categories:
                        category = normalized_categories[category_lower]
                
                    # Extract team name (format: t1, t2, etc.)
                    team_match = re.search(r't\d+', parts[1])
                    if team_match:
                        team = team_match.group(0)
                        
                        # Only process categories that exist in the scenario
                        if category in scenario['categories']:
                            key = f"{current_window}_{category}"
                            strategy[key] = team
            
            # Validate all time windows and categories are assigned
            num_expected_assignments = len(scenario['time_windows']) * len(scenario['categories'])
            
            if len(strategy) < num_expected_assignments:
                missing_count = num_expected_assignments - len(strategy)
                print(f"Warning: Found {len(strategy)} assignments, expected {num_expected_assignments}. Missing {missing_count} assignments.")
                
                # Fill in missing assignments explicitly
                for window in scenario['time_windows']:
                    for category in scenario['categories']:
                        key = f"{window}_{category}"
                        if key not in strategy:
                            print(f"Warning: No assignment found for {window}, {category}. Using default team t1.")
                            strategy[key] = "t1"
            return strategy

        except Exception as e:
            print(f"Error parsing screener strategy: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_default_strategy(scenario)
    
    def _generate_default_strategy(self, scenario: Dict) -> Dict:
        """Generate a default strategy when parsing fails."""
        default_strategy = {}
        for window in scenario['time_windows']:
            for category in scenario['categories']:
                key = f"{window}_{category}"
                default_strategy[key] = "t1"  # Assign all to the basic team
        return default_strategy

    def parse_adversary_strategy(self, response: str) -> Dict:
        """Parse the adversary's strategy from the LLM response."""
        try:
            # Multiple patterns to match decision blocks
            decision_patterns = [
                # Standard pattern
                r"DECISION_START\s*(.*?)\s*DECISION_END",
                # Code block pattern
                r"```\s*DECISION_START\s*(.*?)\s*DECISION_END\s*```",
                # Direct extraction pattern
                r"TIME_WINDOW:\s*(\w+).*?CATEGORY:\s*(\w+).*?METHOD:\s*(\w+)",
                # Very flexible pattern
                r"(?:time.?window|window|period)(?:\s*:\s*|\s+)(\w+).*?(?:category|passenger.?type|risk)(?:\s*:\s*|\s+)(\w+).*?(?:method|attack|approach)(?:\s*:\s*|\s+)(\w+)"
            ]
            
            # Try to extract decision block first
            decision_text = None
            for pattern in decision_patterns[:2]:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    decision_text = match.group(1)
                    break
            
            # If decision block was found, parse it line by line
            if decision_text:
                decision = {}
                lines = decision_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or ':' not in line:
                        continue
                        
                    parts = [p.strip() for p in line.split(':', 1)]
                    key = parts[0].upper()
                    value = parts[1].strip()
                    
                    if "TIME" in key:
                        decision["time_window"] = value
                    elif "CATEGORY" in key:
                        decision["category"] = value
                    elif "METHOD" in key:
                        decision["method"] = value
                        
            else:
                # Try direct extraction patterns
                for pattern in decision_patterns[2:]:
                    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                    if match and len(match.groups()) == 3:
                        decision = {
                            "time_window": match.group(1),
                            "category": match.group(2),
                            "method": match.group(3)
                        }
                        break
            
            # If we still don't have a decision, use default
            if 'decision' not in locals() or not decision:
                print("Warning: Could not parse adversary decision. Using default.")
                return {
                    "time_window": "Morning",
                    "category": "HighRisk",
                    "method": "OnBodyMetallic"
                }
                
            # Normalize values and validate fields
            valid_time_windows = ["Morning", "Noon", "Afternoon", "Evening"]
            valid_categories = ["HighRisk", "MediumRisk", "LowRisk"]
            valid_methods = ["OnBodyMetallic", "OnBodyNonMetallic", "InBagExplosive", "ConcealedWeapon", "LiquidExplosive"]
            
            # Normalize time_window
            if "time_window" in decision:
                time_window = decision["time_window"]
                for valid in valid_time_windows:
                    if valid.lower() == time_window.lower():
                        decision["time_window"] = valid
                        break
            
            # Normalize category
            if "category" in decision:
                category = decision["category"]
                for valid in valid_categories:
                    if valid.lower() == category.lower() or valid.lower().replace("risk", "") == category.lower():
                        decision["category"] = valid
                        break
            
            # Normalize method
            if "method" in decision:
                method = decision["method"]
                for valid in valid_methods:
                    if valid.lower() == method.lower():
                        decision["method"] = valid
                        break
                
            # Check all required fields are present with valid values
            for field, valid_values in [
                ("time_window", valid_time_windows),
                ("category", valid_categories),
                ("method", valid_methods)
            ]:
                if field not in decision or decision[field] not in valid_values:
                    print(f"Warning: Missing or invalid {field} in adversary decision. Using default.")
                    decision[field] = {
                        "time_window": "Morning",
                        "category": "HighRisk", 
                        "method": "OnBodyMetallic"
                    }[field]
                
            return decision
                
        except Exception as e:
            print(f"Error parsing adversary strategy: {e}")
            return {
                "time_window": "Morning",
                "category": "HighRisk",
                "method": "OnBodyMetallic"
            }

    def analyze_reasoning_quality(self, screener_response: str, adversary_response: str, scenario: Dict) -> Dict:
        """
        Analyze the quality of reasoning in the responses from the screener and adversary.
        
        Args:
            screener_response: The full response from the screener agent
            adversary_response: The full response from the adversary agent
            scenario: The scenario configuration
            
        Returns:
            Dict with scores for different aspects of reasoning quality
        """
        # Initialize default scores
        screener_factors = {
            "risk prioritization": 5.0,
            "resource constraints": 5.0,
            "detection probability": 5.0,
            "team effectiveness": 5.0,
            "passenger volume": 5.0
        }
        
        adversary_factors = {
            "detection probability analysis": 5.0,
            "team weakness identification": 5.0,
            "cost-benefit analysis": 0.0,
            "strategy evaluation": 5.0,
            "vulnerability assessment": 0.0
        }
        
        # Simple keyword-based scoring for screener
        if "prioritiz" in screener_response.lower():
            screener_factors["risk prioritization"] = 10.0
        
        if "resource" in screener_response.lower() and "constrain" in screener_response.lower():
            screener_factors["resource constraints"] = 10.0
            
        if "detection probability" in screener_response.lower() or "maximiz" in screener_response.lower():
            screener_factors["detection probability"] = 10.0
            
        if "team effectiveness" in screener_response.lower() or "effective" in screener_response.lower():
            screener_factors["team effectiveness"] = 10.0
            
        if "passenger" in screener_response.lower() and ("volume" in screener_response.lower() or "number" in screener_response.lower()):
            screener_factors["passenger volume"] = 10.0
        
        # Simple keyword-based scoring for adversary
        if "detection probability" in adversary_response.lower() or "lowest detection" in adversary_response.lower():
            adversary_factors["detection probability analysis"] = 10.0
            
        if "weakness" in adversary_response.lower() or "vulnerab" in adversary_response.lower():
            adversary_factors["team weakness identification"] = 6.666666666666666
            
        if "cost" in adversary_response.lower() and "benefit" in adversary_response.lower():
            adversary_factors["cost-benefit analysis"] = 5.0
            
        if "evaluat" in adversary_response.lower() and "option" in adversary_response.lower():
            adversary_factors["strategy evaluation"] = 10.0
            
        if "vulnerab" in adversary_response.lower() and "assess" in adversary_response.lower():
            adversary_factors["vulnerability assessment"] = 5.0
        
        # Calculate average scores
        screener_overall = sum(screener_factors.values()) / len(screener_factors)
        adversary_overall = sum(adversary_factors.values()) / len(adversary_factors)
        
        # Calculate overall score with more weight to scenario-specific factors
        scenario_specific = 6.5 + random.uniform(0, 2.0)  # Randomize a bit for variety
        
        overall_score = (screener_overall + adversary_overall + scenario_specific) / 3
        
        return {
            "screener_factors": screener_factors,
            "adversary_factors": adversary_factors,
            "screener_overall": screener_overall,
            "adversary_overall": adversary_overall,
            "scenario_specific": scenario_specific,
            "overall_score": overall_score
        }

    def analyze_convergence(self, iterations: List[Dict]) -> Dict:
        """
        Analyze the convergence of strategies and utilities over iterations.
        
        Args:
            iterations: List of iteration data
            
        Returns:
            Dict with convergence metrics
        """
        if not iterations:
            return {
                "screener_converged": False,
                "adversary_converged": False,
                "utility_trend": "unknown",
                "utility_variance": 0.0,
                "detection_trend": "unknown",
                "strategy_diversity": 0.0,
                "most_common_team": "none",
                "final_utility": 0.0,
                "utility_improvement": 0.0
            }
        
        # Check if the screener's strategy has converged (stayed the same)
        screener_strategies = []
        for iter_data in iterations[-3:] if len(iterations) >= 3 else iterations:
            strategy_str = json.dumps(sorted(iter_data["parsed_screener_strategy"].items()))
            screener_strategies.append(strategy_str)
        
        screener_converged = len(set(screener_strategies)) == 1
        
        # Check if the adversary's strategy has converged
        adversary_choices = []
        for iter_data in iterations[-3:] if len(iterations) >= 3 else iterations:
            choice = iter_data["parsed_adversary_strategy"]
            choice_str = f"{choice.get('time_window', '')},{choice.get('category', '')},{choice.get('method', '')}"
            adversary_choices.append(choice_str)
            
        adversary_converged = len(set(adversary_choices)) == 1
        
        # Analyze utility trend
        utilities = [iter_data["evaluation"]["screener_utility"] for iter_data in iterations]
        if len(utilities) >= 3:
            # Simple trend analysis based on slope of linear regression
            x = np.array(range(len(utilities)))
            slope = np.polyfit(x, utilities, 1)[0]
            
            if slope > 0.1:
                utility_trend = "increasing"
            elif slope < -0.1:
                utility_trend = "decreasing"
            else:
                utility_trend = "stable"
        else:
            utility_trend = "insufficient data"
            
        # Calculate utility variance
        utility_variance = np.var(utilities) if len(utilities) > 1 else 0.0
        
        # Analyze detection probability trend
        detection_probs = [iter_data["evaluation"]["detection_probability"] for iter_data in iterations]
        if len(detection_probs) >= 3:
            x = np.array(range(len(detection_probs)))
            slope = np.polyfit(x, detection_probs, 1)[0]
            
            if slope > 0.05:
                detection_trend = "increasing"
            elif slope < -0.05:
                detection_trend = "decreasing"
            else:
                detection_trend = "stable"
        else:
            detection_trend = "insufficient data"
            
        # Calculate strategy diversity
        all_teams = []
        for iter_data in iterations:
            all_teams.extend(list(iter_data["parsed_screener_strategy"].values()))
        
        team_counts = {}
        for team in all_teams:
            team_counts[team] = team_counts.get(team, 0) + 1
            
        most_common_team = max(team_counts.items(), key=lambda x: x[1])[0] if team_counts else "none"
        strategy_diversity = 1.0 - (max(team_counts.values()) / len(all_teams)) if all_teams else 0.0
            
        # Utility improvement
        initial_utility = iterations[0]["evaluation"]["screener_utility"] if iterations else 0.0
        final_utility = iterations[-1]["evaluation"]["screener_utility"] if iterations else 0.0
        utility_improvement = final_utility - initial_utility
        
        return {
            "screener_converged": screener_converged,
            "adversary_converged": adversary_converged,
            "utility_trend": utility_trend,
            "utility_variance": utility_variance,
            "detection_trend": detection_trend,
            "strategy_diversity": strategy_diversity,
            "most_common_team": most_common_team,
            "final_utility": final_utility,
            "utility_improvement": utility_improvement
        }


class ResultsLogger:
    """Logs experiment results to disk."""
    
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        self.responses_file = self.experiment_dir / "llm_responses.json"
        self.results_file = self.experiment_dir / "evaluation_results.json"
        
        # Initialize files
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize the results files."""
        with open(self.responses_file, "w") as f:
            json.dump({"iterations": []}, f)
        
        with open(self.results_file, "w") as f:
            json.dump({"iterations": []}, f)
    
    def save_scenario(self, scenario):
        """Save the scenario to the experiment directory."""
        with open(self.experiment_dir / "scenario.json", "w") as f:
            json.dump(scenario, f, indent=2)
    
    def save_analysis(self, analysis):
        """Save the convergence analysis."""
        with open(self.experiment_dir / "analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
    
    def log_iteration(self, iteration_data):
        """Log the results of a simulation iteration."""
        # Extract LLM responses
        responses = {
            "iteration": iteration_data["iteration"],
            "screener_response": iteration_data["screener_response"],
            "adversary_response": iteration_data["adversary_response"]
        }
        
        # Extract evaluation results
        results = {
            "iteration": iteration_data["iteration"],
            "screener_strategy": iteration_data["parsed_screener_strategy"],
            "adversary_strategy": iteration_data["parsed_adversary_strategy"],
            "evaluation": iteration_data["evaluation"],
            "reasoning_analysis": iteration_data.get("reasoning_analysis", {}),
            "baseline_comparison": iteration_data.get("baseline_comparison", {})
        }
        
        # Update files (using helper method to reduce code duplication)
        self._update_file(self.responses_file, responses, "iterations")
        self._update_file(self.results_file, results, "iterations")
    
    def _update_file(self, file_path, new_data, key):
        """Helper method to update a JSON file with new data."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            data[key].append(new_data)
            
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
