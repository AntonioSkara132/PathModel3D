import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from nn import LangPathModel 
from evaluate_model import plot_path, encode_text, load_model, autoregressive_generate


# --- Velocity Computation ---
def calc_point_vels(path: np.ndarray, target_velocity=1.0, ramp_up_time=0, ramp_down_time=0):
	if len(path) < 2:
		raise ValueError("Path must have at least two points")

	segs = path[1:] - path[:-1]
	seg_lens = np.linalg.norm(segs, axis=1)
	seg_dirs = (segs.T / seg_lens).T

	time_s = np.zeros(len(path))
	time_s[1:] = np.cumsum(seg_lens)
	total_time = time_s[-1]

	speeds = np.ones(len(path)) * target_velocity
	if ramp_up_time > 0:
		mask_up = time_s < ramp_up_time
		speeds[mask_up] = target_velocity * (time_s[mask_up] / ramp_up_time)
	if ramp_down_time > 0:
		mask_down = time_s > (total_time - ramp_down_time)
		speeds[mask_down] = target_velocity * ((total_time - time_s[mask_down]) / ramp_down_time)

	dirs = np.zeros_like(path)
	dirs[1:-1] = (seg_dirs[:-1] + seg_dirs[1:]) / 2
	dirs[1:-1] /= np.linalg.norm(dirs[1:-1], axis=1, keepdims=True)
	dirs[0] = np.zeros(2)
	dirs[-1] = np.zeros(2)

	vels = dirs * speeds[:, None]
	return vels

# --- Save CSV ---
def export_to_csv(positions, velocities, actions, filename="trajectory.csv"):
	data = np.hstack([positions, velocities, actions[:, None]])
	header = ["x", "y", "vx", "vy", "action"]
	np.savetxt(filename, data, delimiter=",", header=",".join(header), comments='')

# --- Main ---
def main():
	ap = argparse.ArgumentParser("Generate and plot a trajectory")
	ap.add_argument("--model_path", required=True, help="Path to model checkpoint (.pth)")
	ap.add_argument("--prompt", required=True, help="Text prompt, e.g. 'draw circle in the middle'")
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
	txt, txt_mask = encode_text(args.prompt, device)

	positions, actions = autoregressive_generate(
		model = model, txt = txt, txt_mask = txt_mask,
		x0 = args.x0, y0 = args.y0,
		max_steps=args.max_steps,
		device=device
	)

	velocities = calc_point_vels(positions, target_velocity=args.target_velocity)
	export_to_csv(positions, velocities, actions, filename=args.csv_out)
	plot_path(positions, actions, title=args.prompt)

# Entry point
if __name__ == "__main__":
	main()

