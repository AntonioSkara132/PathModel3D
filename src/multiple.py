from evaluate_model import plot_path, encode_text, load_model, autoregressive_generate
from run_and_export import calc_point_vels, export_to_csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from nn import LangPathModel 

def main():
	ap = argparse.ArgumentParser("Generate and plot a chained trajectory from multiple prompts")
	ap.add_argument("--model_path", required=True, help="Path to model checkpoint (.pth)")
	ap.add_argument("--prompts", nargs='+', required=True, help="List of text prompts")
	ap.add_argument("--max_steps", type=int, default=200)
	ap.add_argument("--device", default="cpu")
	ap.add_argument("--x0", type=float, default=0.5)
	ap.add_argument("--y0", type=float, default=0.5)
	ap.add_argument("--target_velocity", type=float, default=1.0)
	ap.add_argument("--csv_out", default="trajectory.csv")
	args = ap.parse_args()

	device = torch.device(args.device)
	print(f"Using device → {device}")

	model = load_model(args.model_path, device)

	all_positions = []
	all_actions = []

	x0, y0 = args.x0, args.y0

	for i, prompt in enumerate(args.prompts):
		print(f"\nGenerating trajectory for prompt {i+1}: \"{prompt}\"")

		txt, txt_mask = encode_text(prompt, device)

		positions, actions = autoregressive_generate(
			model=model, txt=txt, txt_mask=txt_mask,
			x0=x0, y0=y0,
			max_steps=args.max_steps,
			device=device
		)

		# Store trajectory
		all_positions.append(positions)
		all_actions.append(actions)

		# Update x0, y0 for next prompt
		last_point = positions[-1].astype(np.float32)
		x0, y0 = float(last_point[0]), float(last_point[1])
		print(x0)
		print(y0)

	# Concatenate all
	all_positions = np.vstack(all_positions)
	all_actions = np.concatenate(all_actions)
	velocities = calc_point_vels(all_positions, target_velocity=args.target_velocity)

	# Save and plot
	export_to_csv(all_positions, velocities, all_actions, filename=args.csv_out)
	plot_path(all_positions, all_actions, title=" → ".join(args.prompts))

main()
