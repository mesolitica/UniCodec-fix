import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from pytorch_lightning.cli import LightningCLI, ArgsType


def cli_main(args: ArgsType = None):
    cli = LightningCLI(args=args)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
