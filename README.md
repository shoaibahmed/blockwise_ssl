# Blockwise SSL

A PyTorch implementation for the paper [***Blockwise Self-Supervised Learning at Scale***](https://arxiv.org/abs/2302.01647).

## Execution

To train the model, use `train.sh` script.
The default hyperparameters refers to our best setting, including noise injection.

For testing based on the linear evaluation protocol, use `eval.sh`.
The script trains 4 different linear evaluation heads, one for each of the different blocks of the model.

## Pretrained Checkpoint

Our main model (***Simultaneous Blockwise Training (1x1 CbE + GSP)***, without noise addition) is available to be downloaded from [***Google Drive***](https://drive.google.com/drive/folders/1HFgSjJT0LW5H2E5m4v0ThCzGSXa25zxt).

The model achieves a final accuracy of ***70.15%*** (from the output of block-4).

## Citation

```
@article{siddiqui2022localssl,
  title={Blockwise Self-Supervised Learning at Scale},
  author={Siddiqui, Shoaib Ahmed and Krueger, David and LeCun, Yann and Deny, Stéphane},
  journal={arXiv preprint},
  year={2022},
  url={https://arxiv.org/abs/2302.01647}
}
```

## Credits

This code is mainly adapted from the original Barlow Twins codebase:
`https://github.com/facebookresearch/barlowtwins`

## Issues/Feedback

In case of any issues, feel free to drop me an email or open an issue on the repository.

Email: **msas3@cam.ac.uk**

## License

MIT
