import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.nn import PathModel3D

"""
Generate a robotic drawing trajectory from a trained model
and plot the (x, y) path with binned actions (0 = blue, 1 = red).
"""

# helper: load / build your model from config
def load_model(model_path: str, device: torch.device):
	"""
	Load a trained model. Supports:
	- PyTorch checkpoint with config dict (`torch.save({'model_state_dict': ..., 'config': ...})`)
	Returns a model moved to the target device.
	"""

	# Standard torch.save(dict) format
	checkpoint = torch.load(model_path, map_location=device)
	config = checkpoint.get("config", {
		"d_model": 128,
		"num_heads": 8,
		"num_decoder_layers": 5,
		"dropout": 0.2
	})
	model = PathModel3D(
		d_model=config["d_model"],
		num_heads_decoder=config["num_heads"],
		num_decoder_layers=config["num_decoder_layers"],
		dropout=config["dropout"]
	)
	model.load_state_dict(checkpoint["model_state_dict"])

	model.eval()
	return model.to(device)

# ---------------------------------------------------------------------
# helper: tokenize text prompt
# ---------------------------------------------------------------------
def encode_text(prompt: str, device: torch.device):
	"""
	Returns token_ids (Tensor[1, L])  and  mask (Bool[1, L])
	Adapt this to your tokenizer.
	"""
	from transformers import AutoTokenizer
	tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
	enc = tokenizer(
		prompt,
		return_tensors="pt",
		padding=True,
		truncation=True
	)
	txt_ids = enc["input_ids"].to(device)
	txt_mask = (enc["attention_mask"] == 0).to(device)  # True = pad
	return txt_ids, txt_mask

# ---------------------------------------------------------------------
# generation loop
# ---------------------------------------------------------------------
def autoregressive_generate(x0, y0, model, txt, txt_mask,
							max_steps=200,
							device=torch.device("cpu")):
	"""
	Generates up to max_steps points or stops early
	if the model sets stop_flag > 0.5.
	Returns:
	positions : ndarray [N, 2]
	actions   : ndarray [N]   (0/1 after binning on threshold 0.5)
	"""
	# masks / tensors
	path_mask = torch.Tensor([[0, 0]]).to(device)
	start = torch.tensor([[[x0, y0, 0.0, 0.0]]], device=device)  # [1, 1, 4]
	tgt = start.clone()

	positions = [[x0, y0]]
	actions = [start[0, 0, 2].item()]  # initial action (0)
	print(txt)

	for _ in range(max_steps):
		with torch.no_grad():
			pred = model(text=txt,
						 tgt=tgt,
						 text_mask=txt_mask,
						 path_mask=path_mask
						 )  # [1, T, 4]

		next_pt = pred[:, -1, :]  # [1, 4]
		positions.append(next_pt[0, :2].cpu().numpy())
		actions.append(next_pt[0, 2].item())

		# cat for next step
		tgt = torch.cat([tgt, next_pt.unsqueeze(1)], dim=1)
		path_mask = torch.cat([path_mask, torch.zeros((1, 1), device=device)], dim=1)

		# stop flag
		if next_pt[0, 3] > 0.5:
			break

	positions = np.array(positions)
	actions = np.array(actions)
	binned = (actions >= 0.5).astype(int)  # 0/1
	return positions, binned

# ---------------------------------------------------------------------
# plotting helper
# ---------------------------------------------------------------------
def plot_path(positions, actions, title="Generated Path"):
	plt.figure(figsize=(8, 6))
	for (x, y), a in zip(positions, actions):
		color = "red" if a == 1 else "blue"
		label = "Action = 1" if (a == 1 and np.sum(actions == 1) == 1) else \
			"Action = 0" if (a == 0 and np.sum(actions == 0) == 1) else None
		plt.scatter(x, y, c=color, s=50, label=label)

	plt.title(title + "  (0=blue, 1=red)")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.xlim(-0.1, 1.1)
	plt.ylim(-0.1, 1.1)
	plt.gca().set_aspect('equal')
	plt.grid(True)
	plt.legend()
	plt.savefig('fig.png')

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
	ap = argparse.ArgumentParser("Generate and plot a trajectory")
	ap.add_argument("--model_path", required=True, help="Path to model checkpoint (.pth)")
	ap.add_argument("--prompt", required=True, help="Text prompt, e.g. 'draw circle in the middle'")
	ap.add_argument("--max_steps", type=int, default=200, help="Maximum autoregressive steps")
	ap.add_argument("--device", default="cpu")
	ap.add_argument("--x0", type=float, default=0.5)
	ap.add_argument("--y0", type=float, default=0.5)
	args = ap.parse_args()
	x0 = args.x0
	y0 = args.y0

	device = torch.device(args.device)
	print(f"Using device → {device}")

	model = load_model(args.model_path, device)
	txt, txt_mask = encode_text(args.prompt, device)
	print(txt)
	positions, actions = autoregressive_generate(
		x0 = x0, y0 = 0,
		model = model, txt = txt, txt_mask = txt_mask,
		max_steps=args.max_steps,
		device=device
	)

	plot_path(positions, actions, title=args.prompt)
	print("Created fig.png file") 

# ---------------------------------------------------------------------
if __name__ == "__main__":
	main()


