import models.model_factory as factory
import models.gm_utils as gm_utils
from options import TrainOptions, Options
from constants import OUT_DIR
from show.view_utils import view
from custom_types import *
from process_data.mesh_loader import get_loader, AnotherLoaderWrap
from process_data.files_utils import collect, init_folders
import matplotlib.pyplot as plt


class ViewMem:

    @staticmethod
    def get_palette(num_colors: int) -> list:
        if num_colors == 1:
            return [.45]
        if num_colors not in ViewMem.colors:
            ViewMem.colors[num_colors] = [ViewMem.color_map(float(idx) / (num_colors - 1)) for idx in range(num_colors)]
        return ViewMem.colors[num_colors]

    colors = {}
    color_map =  plt.cm.get_cmap('Spectral')
    device = CPU
    ds_size = 1
    max_items = 8
    points_in_sample = 2048
    loader = None
    memory = None
    save_separate = {'sample', 'byid'}
    last_idx = None


def create_palettes(splits: list) -> list:
    palette = []
    for split in splits:
        num_colors = len(split) - 1
        palette.append(ViewMem.get_palette(num_colors))
    return palette


def save_pil_image(image, path, prefix: str, start_counts:int, _) -> int:
    image.save(f'{path}{prefix}_{start_counts:03d}.png')
    return 1


def save_np_points(points_group, path: str, prefix: str, start_counts:int, trace: ViewMem) -> int :
    for i, group in enumerate(zip(*points_group)):
        if prefix in trace.save_separate:
            saving_path = f'{path}{prefix}_{start_counts + i:03d}.npz'
        else:
            saving_path = f'{path}{prefix}_{start_counts:03d}_{i:02d}.npz'
        np.savez_compressed(saving_path, points=group[0], splits=group[1], palette=group[2])
    if prefix in trace.save_separate:
        return len(points_group[0])
    return 1


def saving_handler(args: Options, trace: ViewMem):

    def init_prefix(prefix: str, saving_index: int, saving_folder: str):
        nonlocal saving_dict, suffix
        same_type_file = collect(saving_folder, suffix[saving_index], prefix=prefix)
        if len(same_type_file) == 0:
            saving_dict[saving_index][prefix] = 0
        else:
            last_number_file_name = same_type_file[-1][1]
            saving_dict[saving_index][prefix] = int(last_number_file_name.split('_')[1]) + 1

    def get_path(prefix: str, saving_index:int) -> str:
        nonlocal saving_dict
        saving_folder = f'{saving_folders[saving_index]}{prefix}/'
        if prefix not in saving_dict[saving_index]:
            init_folders(f'{saving_folders[saving_index]}/{prefix}/')
            init_prefix(prefix, saving_index, saving_folder)
        # saving_dict[saving_index][prefix] += 1
        return saving_folder


    def handle(prefix: str, *items):
        nonlocal saving_dict
        msg = '0: continue | 1: save image | 2: save points | 3: save both '
        to_do = get_integer((0, 4), msg)
        if to_do > 0:
            to_do = to_do - 1
            for i in range(len(saving_f)):
                if to_do == i or to_do == len(saving_f):
                    path = get_path(prefix, i)
                    saving_dict[i][prefix] += saving_f[i](items[i], path, prefix, saving_dict[i][prefix], trace)

    saving_dict = [dict(), dict()]
    saving_folders = [f'{OUT_DIR}/{args.info}/eval_images/', f'{OUT_DIR}/{args.info}/eval_points/']
    saving_f = [save_pil_image, save_np_points]
    suffix = ['.png', '.npz']
    return handle


def init_loader(args, trace: ViewMem):
    if trace.loader is None:
        trace.loader = AnotherLoaderWrap(get_loader(args), trace.max_items)


def get_z_by_id(encoder, args: Options, num_items: int, idx, trace: ViewMem):
    init_loader(args, trace)
    if idx is None or trace.last_idx is None:
        inds, data = trace.loader.get_random_batch()
        trace.last_idx = [int(index) for index in inds[:num_items]]
    else:
        data = trace.loader.get_by_ids(*idx)
    input_points = data[:num_items].to(trace.device)
    z, _, _ = encoder(input_points)
    return z


def get_integer(allowed_range: tuple, msg: str='') -> int:
    if msg == '':
        msg = f'\tPlease choose number of objects to show from {allowed_range[0]} to {allowed_range[1] -1}\n\t'
    while (True):
        try:
            integer = int(input(msg))
            if allowed_range[0] <= integer < allowed_range[1]:
                break
            else:
                raise ValueError
        except ValueError:
            print('Unexpected argument, please try again')
    return integer


def sample(_, decoder, args: Options, trace: ViewMem):
    if trace.memory is None:
        num_items = get_integer((1, 8))
    else:
        num_items = trace.memory[1]
    z = torch.randn(num_items, args.dim_z).to(trace.device)
    gms = decoder(z)
    vs, splits = gm_utils.hierarchical_gm_sample(gms, trace.points_in_sample, False)
    vs = vs.cpu().numpy()
    splits = [s for s in splits]
    palette = create_palettes(splits)
    im, points = view([vs_ for vs_ in vs], splits, palette)
    return True, (sample, num_items), im, (points, splits, palette)


def hgmms(encoder, decoder, args: Options, trace: ViewMem):
    z = get_z_by_id(encoder, args, 1, None, trace)
    gms = decoder(z)
    num_gms = len(gms)
    vs = []
    splits = []
    for i in range(num_gms):
        gms_ = [gms[i] for i in range(i+1)]
        vs_, splits_ = gm_utils.hierarchical_gm_sample(gms_, trace.points_in_sample)
        vs.append(vs_.squeeze(0).cpu().numpy())
        splits.append(splits_.squeeze(0).cpu().numpy())
    palette = create_palettes(splits)
    im, points = view(vs, splits, palette)
    return True, (hgmms, ), im, (points, splits, palette)


def interpolate(encoder, decoder, args: Options, trace: ViewMem):
    if trace.memory is None:
        msg = '\tPlease choose number of interpolation: from 8 to 20\n\t'
        num_interpulate = get_integer((7, 21), msg)
    else:
        num_interpulate = trace.memory[1]
    z = get_z_by_id(encoder, args, 2, trace.last_idx, trace)
    gms = decoder.interpulate(z, num_interpulate)
    vs, splits = gm_utils.hierarchical_gm_sample(gms, trace.points_in_sample)
    spread = [vs[i].cpu().numpy() for i in range(vs.shape[0])]
    splits = [s for s in splits]
    palette = create_palettes(splits)
    im, points = view(spread, splits, palette) #, save_path=f'{cp_folder}/interpulate_{idx[0].item()}_{idx[1].item()}.png')
    return True, (interpolate, (num_interpulate,)), im, (points, splits, palette)


def evaluate(args: Options, trace: ViewMem):

    def last_try(encdoer, decoder, args, trace):
        if trace.memory is None:
            print("Don't know what to do")
            return False, None
        return trace.memory[0](encdoer, decoder, args, trace)

    def to_exit(_, __, ___, ____):
        print(':-o Goodbye')
        return False, None, None, None

    encoder, _ = factory.model_lc(args.encoder, args, device=trace.device)
    decoder, _ = factory.model_lc(args.decoder, args, device=trace.device)
    encoder.eval(), decoder.eval()

    choices = {0: to_exit, 1: sample, 2: interpolate, 3: hgmms, 4: last_try}
    menu = ' | '.join(sorted(list(f"{key}: {str(item).split()[1].split('.')[-1]}" for key, item in choices.items())))
    eval_choice = 1
    allow_saving = saving_handler(args, trace)
    while eval_choice:
        eval_choice = get_integer((0, len(choices)), menu + '\n')
        with torch.no_grad():
            if eval_choice != len(choices) - 1:
                trace.memory = None
            check, trace.memory, image, points_group = choices[eval_choice](encoder, decoder, args, trace)
            if check:
                function_name = str(trace.memory[0]).split()[1].split('.')[-1]
                allow_saving(function_name, image, points_group)
                print(f"{function_name} done")


if __name__ == '__main__':
    cls = 'table'
    evaluate(TrainOptions(tag=cls).load(), ViewMem())