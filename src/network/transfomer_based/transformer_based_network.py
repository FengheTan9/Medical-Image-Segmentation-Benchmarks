from .transUnet.transunet import TransUnet
from .swinUnet.vision_transformer import SwinUnet
from .swinUnet.config import get_config
from .medicalT.axialnet import MedT


def get_transformer_based_model(parser, model_name: str, img_size: int, num_classes: int, in_ch: int):
    if model_name == "MedT":
        model = MedT(img_size=img_size, imgchan=in_ch, num_classes=num_classes)
    elif model_name == "SwinUnet":
        parser.add_argument('--zip', action='store_true',
                            help='use zipped dataset instead of folder dataset')
        parser.add_argument(
            '--cfg', type=str, default="./src/network/transfomer_based/swinUnet/swin_tiny_patch4_window7_224_lite.yaml",
            help='path to config file', )
        parser.add_argument(
            "--opts",
            help="Modify config options by adding 'KEY VALUE' pairs. ",
            default=None,
            nargs='+',
        )
        parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                            help='no: no cache, '
                                 'full: cache all data, '
                                 'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
        parser.add_argument('--resume', help='resume from checkpoint')
        parser.add_argument('--accumulation-steps', type=int,
                            help="gradient accumulation steps")
        parser.add_argument('--use-checkpoint', action='store_true',
                            help="whether to use gradient checkpointing to save memory")
        parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                            help='mixed precision opt level, if O0, no amp is used')
        parser.add_argument('--tag', help='tag of experiment')
        parser.add_argument('--eval', action='store_true',
                            help='Perform evaluation only')
        parser.add_argument('--throughput', action='store_true',
                            help='Test throughput only')
        config = get_config(parser.parse_args())
        model = SwinUnet(config, img_size=224, num_classes=num_classes)
    elif model_name == "TransUnet":
        model = TransUnet(img_ch=in_ch, output_ch=num_classes)
    else:
        model = None
        print("model err")
        exit(0)
    return model
