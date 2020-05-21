from options import RegOptions
from registration.registration_algorithm import RegistrationAlgorithm
from registration.registration_hgm import RegistrationHgmDual
from process_data.files_utils import collect
from tqdm import tqdm
import constants as const
from process_data.points_utils import apply_affine
from custom_types import *


def mse_error(source, target_, rotations, translations):
    target = np.einsum('dc,nd->nc',rotations[0], source) - translations[0][np.newaxis, :]
    target = np.einsum('rd,nd->nr', rotations[1], target + translations[1][np.newaxis, :])
    error = (target - target_)**2
    error = error.sum() / (source.shape[0] * source.shape[1])
    return error


def registration_test(register: RegistrationAlgorithm, test_name: str, export_number=(0, 0)):
    registration_test_folder = f'{const.REGISTRATION}/{test_name}/'
    files = collect(registration_test_folder, '.npz')
    progress = tqdm(total=len(files),
                    desc=f'Registration | {test_name} | {register.name}')
    total_error = 0
    total_check = 0
    for i, file in enumerate(files):
        data = np.load(''.join(file))
        source, target = data['test_vs'][0], data['test_vs'][1]
        rot, trnl = data['rotations'], data['translations']
        rot_, trnl_ = register(source, target)
        target_ = apply_affine(source, rot_, trnl_)
        error = mse_error(source, target_, rot, trnl)
        if np.isnan(error):
            print('error')
        total_error += min(error,1)
        total_check+= 1
        progress.update()
        progress.set_postfix(error= error, average= total_error / max(1, total_check))
    print(f'testeted: {total_check} / {len(files)}')
    print(f'error: {total_error / max(1, total_check)}')
    progress.set_postfix({'error': total_error / max(1, total_check)})
    progress.close()


if __name__ == '__main__':
    device = CUDA(2)
    sub_test_ = 'noise'
    cls = 'airplane'
    opt = RegOptions(task='reg', tag=cls).load()
    alg = RegistrationHgmDual(opt)
    for max_angle in [30, 180]:
        for cover_range in [(0.5, 0.8), (0.3, 0.5)]:
            max_angle_tag = max_angle if max_angle > 0 else 180
            test_name_ = f'{sub_test_}/{cls}_{cover_range[0]}_{cover_range[1]}_{max_angle_tag}'
            registration_test(alg, test_name_)
