from options.base_option import BaseOptions

class EvalT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default="latest", help='Checkpoint you want to use, {latest, net_best_fid, etc}')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

        self.parser.add_argument('--ext', type=str, default='text2motion', help='Extension of the result file or folder')
        self.parser.add_argument("--num_batch", default=2, type=int,
                                 help="Number of batch for generation")
        self.parser.add_argument("--repeat_times", default=1, type=int,
                                 help="Number of repetitions, per sample text prompt")
        self.parser.add_argument("--cond_scale", default=4, type=float,
                                 help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
        self.parser.add_argument("--temperature", default=1., type=float,
                                 help="Sampling Temperature.")
        self.parser.add_argument("--topkr", default=0.9, type=float,
                                 help="Filter out percentil low prop entries.")
        self.parser.add_argument("--time_steps", default=18, type=int,
                                 help="Mask Generate steps.")
        self.parser.add_argument("--seed", default=10107, type=int)

        self.parser.add_argument('--gumbel_sample', action="store_true", help='True: gumbel sampling, False: categorical sampling.')
        self.parser.add_argument('--use_res_model', action="store_true", help='Whether to use residual transformer.')
        # self.parser.add_argument('--est_length', action="store_true", help='Training iterations')

        self.parser.add_argument('--res_name', type=str, default='tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw', help='Model name of residual transformer')
        self.parser.add_argument('--text_path', type=str, default="", help='Text prompt file')


        self.parser.add_argument('-msec', '--mask_edit_section', nargs='*', type=str, help='Indicate sections for editing, use comma to separate the start and end of a section'
                                 'type int will specify the token frame, type float will specify the ratio of seq_len')
        self.parser.add_argument('--text_prompt', default='', type=str, help="A text prompt to be generated. If empty, will take text prompts from dataset.")
        self.parser.add_argument('--source_motion', default='example_data/000612.npy', type=str, help="Source motion path for editing. (new_joint_vecs format .npy file)")
        self.parser.add_argument("--motion_length", default=0, type=int,
                                 help="Motion length for generation, only applicable with single text prompt.")
        self.is_train = False


class EvalG1Options(EvalT2MOptions):
    """Generation options for native Unitree G1 control trajectories."""

    def initialize(self):
        super().initialize()
        self.parser.set_defaults(which_epoch='net_best_val')
        self.parser.add_argument('--g1_data_root', type=str, default='./dataset/BONES-G1',
                                 help='Converted native G1 dataset directory.')
        self.parser.add_argument('--model_fps', type=float, default=20.0,
                                 help='Frame rate used while training the G1 model.')
        self.parser.add_argument('--output_fps', type=float, default=120.0,
                                 help='Frame rate written to the generated G1 CSV.')
        self.parser.add_argument('--root_position_scale', type=float, default=0.01,
                                 help='Meters-to-CSV root translation scale; BONES-SEED uses centimeters.')
        self.parser.add_argument('--disable_data_range_clip', action='store_true',
                                 help='Do not clip generated features to observed training-data ranges.')
