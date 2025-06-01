"""
Main script for the Threat Screening Games (TSG) simulation using LLM agents.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter

from tsg_agents import LLMAgentManager
from tsg_evaluation import ScenarioManager, Evaluator, ResultsLogger
from tsg_visualization import visualize_results

def setup_experiment_directory(base_dir="/home/caiqi/phd/tsg/experiments"):
    """Create a timestamped directory for the experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / timestamp
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def run_single_simulation(scenario, llm_agent_manager, evaluator, logger, num_iterations=1, history_window=3):
    """Run a single TSG simulation with the given scenario."""
    print(f"Running simulation with scenario: {scenario['name']}")
    
    results = {
        "scenario": scenario,
        "iterations": []
    }
    
    # 只保留随机基线评估
    random_baseline_evaluation = evaluator.evaluate_random_baseline(scenario)
    
    print(f"Random baseline expected utility: {random_baseline_evaluation['expected_utility']:.4f} ± {random_baseline_evaluation['std_deviation']:.4f}")
    
    # Store history for both agents
    screener_history = []
    adversary_history = []
    
    for i in range(num_iterations):
        print(f"\nIteration {i+1}/{num_iterations}")
        
        # Step 1: Get and evaluate screener strategy
        screener_response = llm_agent_manager.get_screener_strategy(
            scenario, 
            screener_history[-history_window:] if i > 0 else None,
            adversary_history[-history_window:] if i > 0 else None
        )
        screener_strategy = evaluator.parse_screener_strategy(scenario, screener_response)
        
        # Step 2: Get and evaluate adversary strategy
        adversary_response = llm_agent_manager.get_adversary_strategy(
            scenario, 
            screener_strategy,
            adversary_history[-history_window:] if i > 0 else None,
            screener_history[-history_window:] if i > 0 else None
        )
        adversary_strategy = evaluator.parse_adversary_strategy(adversary_response)
        
        # Step 3: Evaluate interaction and analyze reasoning
        evaluation_results = evaluator.evaluate(scenario, screener_strategy, adversary_strategy)
        
        # Analyze reasoning quality if method exists
        reasoning_scores = {}
        try:
            reasoning_scores = evaluator.analyze_reasoning_quality(
                screener_response, adversary_response, scenario
            )
        except (AttributeError, Exception) as e:
            print(f"Warning: Could not analyze reasoning quality: {e}")
            reasoning_scores = {
                "overall_score": 5.0,
                "screener_overall": 5.0,
                "adversary_overall": 5.0,
                "scenario_specific": 5.0,
            }
        
        # Log iteration data, 移除与启发式基线的比较
        iteration_data = {
            "iteration": i+1,
            "screener_response": screener_response,
            "parsed_screener_strategy": screener_strategy,
            "adversary_response": adversary_response,
            "parsed_adversary_strategy": adversary_strategy,
            "evaluation": evaluation_results,
            "reasoning_analysis": reasoning_scores
        }
        
        # Update history
        screener_history.append({
            "iteration": i+1,
            "strategy": screener_strategy,
            "utility": evaluation_results["screener_utility"]
        })
        
        adversary_history.append({
            "iteration": i+1,
            "choice": adversary_strategy,
            "utility": evaluation_results["adversary_utility"],
            "detection_probability": evaluation_results["detection_probability"]
        })
        
        results["iterations"].append(iteration_data)
        logger.log_iteration(iteration_data)
        
        # Display summary
        print(f"Screener strategy: {len(screener_strategy)} assignments")
        print(f"Adversary choice: {adversary_strategy['time_window']}, {adversary_strategy['category']}, {adversary_strategy['method']}")
        print(f"Detection: {evaluation_results['detection_probability']:.2f} | Screener: {evaluation_results['screener_utility']:.2f} | Adversary: {evaluation_results['adversary_utility']:.2f}")
        
        # 添加明确的胜负判定结果
        outcome = "Screener Success" if evaluation_results['screener_utility'] >= 0 else "Adversary Success"
        print(f"Outcome: {outcome}")
    
    # Add convergence analysis
    try:
        results["analysis"] = evaluator.analyze_convergence(results["iterations"])
    except (AttributeError, Exception) as e:
        print(f"Warning: Could not analyze convergence: {e}")
        results["analysis"] = {
            "screener_converged": False,
            "adversary_converged": False,
            "utility_trend": "unknown",
            "strategy_diversity": 0.0
        }
    
    # 只保留随机基线
    results["baseline_comparison"] = {
        "random_baseline": random_baseline_evaluation
    }
    
    logger.save_analysis(results["analysis"])
    
    return results

def main():
    parser = argparse.ArgumentParser(description="TSG Simulation with LLM Agents")
    parser.add_argument("--scenario", type=str, default="default", help="Scenario to run")
    parser.add_argument("--iterations", type=int, default=8, help="Number of iterations")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="LLM model")
    parser.add_argument("--history-window", type=int, default=3, help="Past iterations in history")
    parser.add_argument("--ollama-host", type=str, help="Ollama API host URL (e.g., http://localhost:11434)")
    
    # 删除 provider 参数，我们只用 Ollama
    args = parser.parse_args()
    
    # Set Ollama host if provided
    if args.ollama_host:
        os.environ['OLLAMA_HOST'] = args.ollama_host
    
    # Setup experiment directory and components
    exp_dir = setup_experiment_directory()
    print(f"Experiment results will be saved to: {exp_dir}")
    
    scenario_manager = ScenarioManager()
    scenario = scenario_manager.load_scenario(args.scenario)
    
    # 修正这里: 去掉 provider 参数
    llm_agent_manager = LLMAgentManager(model=args.model)
    
    # Save model information to the experiment directory
    with open(exp_dir / "model_info.json", "w") as f:
        json.dump({
            "model": args.model,
            "model_family": llm_agent_manager.model_family,
            "history_window": args.history_window
        }, f, indent=2)
    
    # Verify LLM connection before proceeding
    if not llm_agent_manager.check_connection():
        print("\nWARNING: Cannot connect to Ollama API. Consider the following:")
        print("1. Make sure Ollama is installed and running")
        print("2. Check if the Ollama Python client is installed: pip install ollama")
        print("3. Verify the Ollama server is accessible (default: http://localhost:11434)")
        print("4. You can specify a custom host with --ollama-host parameter\n")
        user_input = input("Continue with dummy responses? (y/n): ")
        if user_input.lower() != 'y':
            print("Exiting.")
            return
    
    evaluator = Evaluator()
    logger = ResultsLogger(exp_dir)
    
    logger.save_scenario(scenario)
    
    # Run simulation
    results = run_single_simulation(
        scenario=scenario, llm_agent_manager=llm_agent_manager,
        evaluator=evaluator, logger=logger,
        num_iterations=args.iterations, history_window=args.history_window
    )
    
    # Save results and generate visualization
    with open(exp_dir / "final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    visualize_results(exp_dir)
    
    # Print summary
    print("\n=== Experiment Summary ===")
    
    if results["iterations"]:
        # Count attack patterns
        attack_methods = Counter(item["parsed_adversary_strategy"]["method"] for item in results["iterations"])
        targets = Counter((item["parsed_adversary_strategy"]["time_window"], item["parsed_adversary_strategy"]["category"]) 
                         for item in results["iterations"])
        
        # Calculate win rates
        screener_wins = sum(1 for item in results["iterations"] if item["evaluation"]["screener_utility"] >= 0)
        
        print(f"Iterations: {args.iterations} | Screener wins: {screener_wins}/{args.iterations}")
        print(f"Most common target: {targets.most_common(1)[0][0]}")
        print(f"Most common attack: {attack_methods.most_common(1)[0][0]}")
        print(f"Final detection: {results['iterations'][-1]['evaluation']['detection_probability']:.2f}")
        print(f"Utility trend: {results['analysis']['utility_trend']}")
    
    print(f"Results saved to: {exp_dir}/tsg_results.png")

if __name__ == "__main__":
    main()
