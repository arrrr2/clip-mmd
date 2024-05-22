import torch
import argparse

from clip_mmd.logic import CMMD  

def main():
    parser = argparse.ArgumentParser(description="Command Line Interface for CMMD calculations and feature extraction.")
    parser.add_argument('data_path_1', type=str, help='Path to the first data folder or file')
    parser.add_argument('data_path_2', type=str, nargs='?', default=None, help='Path to the second data folder or file, or output path of pre-extracted features')
    parser.add_argument('--no-cuda', action='store_false', help='only use cpu')
    parser.add_argument('--gpus', type=str, default='', help='Comma-separated list of GPUs to use (e.g., 0,1,2,3). Use it if you want do on multi gpus.')
    parser.add_argument('--no-mem-save', action='store_false', help='Flag to disable memory-saving features')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--calculator-bs', type=int, default=128, help='Batch size for the CMMD calculator')
    parser.add_argument('--model', type=str, default='openai/clip-vit-large-patch14-336', help='Model to use for feature extraction. Default: openai/clip-vit-large-patch14-336,')
    parser.add_argument('--num-workers', type=int, default=8, help='Numbers of dataloader workers')
    parser.add_argument('--extract-mode', action='store_true', help='If enabled, only extract reatures from data_path_1, and save to data_path_2.')
    parser.add_argument('--size', type=int, default=336, help='Image patch size for model input')
    parser.add_argument('--interpolation', type=str, default='bicubic',choices=['nearest', 'bilinear', 'bicubic', 'lanczos', 'hamming','box'], help='Interpolation algorithm for resampling an image. Default: bicubic.')

    args = parser.parse_args()

    # Initialize the CMMD processor
    if args.gpus != '': gpus = [int(i) for i in args.gpus.split(',')]
    else: gpus = None



    processor = CMMD(True, extract_model=args.model, 
                     img_size=(args.size,) * 2, device="cuda" if args.no_cuda else 'cpu', 
                     data_parallel= (gpus is not None), device_ids=gpus, interpolation=args.interpolation, 
                     feat_bs=args.batch_size, num_workers=args.num_workers, compute_bs=args.calculator_bs, 
                     low_mem=args.no_mem_save)

    if args.extract_mode:
        if args.data_path_2:
            # Output path provided, extract features to specified path
            x, x_computed = processor.prepare_input(args.data_path_1)
            features = processor.calculate_statics(x)
            torch.save(features, args.data_path_2)
        else:
            # No output path provided, default action needed
            print("Error: Output path for extracted features not provided.")
            exit(1)
    elif args.data_path_2:
        # Calculate CMMD between two datasets
        cmmd_value = processor.execute(args.data_path_1, args.data_path_2)
        print(f"CMMD Value: {cmmd_value}")
    else:
        # Only one data path provided, assume generation of pre-extracted features without explicit extraction mode
        print("Error: Second data path or extraction mode flag required.")
        exit(1)

if __name__ == "__main__":
    main()
