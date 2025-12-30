import argparse
import torch
from torch.utils.data import DataLoader
from nn import PathModel3D
import sys
sys.path.append("/content/LangPathModel")
from data.data_utils import ShapeDataset, collate_fn
from train import train

"""
If you want to train your LangPath model you call this function.
It has 12 arguments explained in code below
Every 10 epochs it stores your model in LangPathModel/checkpoints folder
"""


def get_args():
	"""Parse command‑line arguments."""
	parser = argparse.ArgumentParser(description="Train TrajectoryModel with configurable hyper‑parameters")

	# Training hyper‑parameters
	parser.add_argument("--niter", type=int, default=50, help="Number of training iterations/epochs")
	parser.add_argument("--start_lr", type=float, default=1e-3, help="Initial learning rate")
	parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularisation)")
	parser.add_argument("--lr_step", type=int, default=15, help="Step size for LR scheduler (if any)")
	parser.add_argument("--gamma", type=float, default=0.1, help="Scheduler factor")


	# Model architecture
	parser.add_argument("--d_model", type=int, default=128, help="Transformer hidden size (d_model)")
	parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads in the decoder")
	parser.add_argument("--num_decoder_layers", type=int, default=5, help="Number of decoder layers")
	parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

	# Dataloader / misc
	parser.add_argument("--dataset_path", type=str, default="cats_and_dogs.pth", help="Where is data")
	parser.add_argument("--batch_size", type=int, default=500, help="Mini‑batch size")
	parser.add_argument("--output_path", type=str, default="cats_and_dogs.pth", help="Where to save the trained model")
	parser.add_argument("--ckpt_path", type=str, default="cats_and_dogs.pth", help="Where to store checkpoints")

	return parser.parse_args()


def main():
	args = get_args()

	dataset = PathDataset(args.dataset_path)
	dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

	# Instantiate model with cmd‑line hyper‑parameters
	model = LangPathModel(
	d_model=args.d_model,
	num_heads_decoder=args.num_heads,
	num_decoder_layers=args.num_decoder_layers,
	dropout=args.dropout,
	)

	#  Move model to device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	train(
		model=model,
		niter=args.niter,
		dataloader=dataloader,
		device=device,
		start_lr=args.start_lr,
		gamma = args.gamma,
		weight_decay=args.weight_decay,
		step=args.lr_step,
		ckpt_dir = args.ckpt_path
		)

    #Save the model
	model.to("cpu")
	save_dict = {
		"model_state_dict": model.state_dict(),
		"config": {
			"d_model": args.d_model,
			"num_heads": args.num_heads,
			"num_decoder_layers": args.num_decoder_layers,
			"dropout": args.dropout,
			"niter": args.niter,
			"start_lr": args.start_lr,
			"weight_decay": args.weight_decay,
			"lr_step": args.lr_step,
			"gamma": args.gamma,
			"batch_size": args.batch_size,
		}	
	}
	torch.save(save_dict, args.output_path)
	print(f"\nModel and config saved to {args.output_path}")


if __name__ == "__main__":
    main()


