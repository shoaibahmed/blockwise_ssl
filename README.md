# Blockwise SSL

A PyTorch implementation for the paper [***Blockwise Self-Supervised Learning at Scale***](https://arxiv.org/abs/2302.01647).

## Execution

To train the model, use `train.sh` script.
The default hyperparameters refers to our best setting, including noise injection.

For testing based on the linear evaluation protocol, use `eval.sh`.
The script trains 4 different linear evaluation heads, one for each of the different blocks of the model.

## Pretrained Checkpoint

Our main model i.e., ***Simultaneous Blockwise Training (1x1 CbE + GSP)***, without noise addition, is available to be downloaded from [***here***](https://drive.google.com/drive/folders/1o65G5_fDG4nsbu5TMXxwn-53RETrOonC).

The model achieves a final accuracy of ***70.15%*** (from the output of block-4).

The end-to-end trained Barlow Twins model w/ GSP (300 epochs) which served as the target for our experiments can be downloaded from [***here***](https://drive.google.com/drive/folders/1NtoDYU7ZiFe2LQTt8z33vFoUclMBC3Nm).

## Citation

```
@article{
  siddiqui2024blockwise,
  title={Blockwise Self-Supervised Learning at Scale},
  author={Shoaib Siddiqui and David Krueger and Yann LeCun and Stephane Deny},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=M2m618iIPk},
  note={}
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
