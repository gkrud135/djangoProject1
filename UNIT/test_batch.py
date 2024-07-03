from __future__ import print_function
from UNIT.utils import get_config, pytorch03_to_pytorch04
from UNIT.trainer import MUNIT_Trainer, UNIT_Trainer
from subprocess import call
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils
import torch
import os
import sys
from time import time
import io

def d2n(image):
    # 현재 파일의 절대 경로
    current_file_path = os.path.abspath(__file__)

    # 현재 파일의 디렉토리
    current_dir = os.path.dirname(current_file_path)

    # 프로젝트 최상위 폴더 경로
    project_root = os.path.dirname(current_dir)  # 필요에 따라 반복적으로 적용
    # 변수 정의
    checkdir = "UNIT/checkpoints/512_vgg"
    checkpoints = os.path.join(checkdir, "gen_00473000.pt")
    configs = "UNIT/configs/unit_day2night_512_vgg.yaml"
    indir = "disaster_inpainting/disaster.png"

    outdir = "static/disaster"
    gpu = 0  # 기본 GPU 장치 번호를 0으로 설정
    d2n = 1  # 기본 변환 방향을 Day -> Night로 설정

    # 기존의 argparse 사용 부분을 직접 변수 설정으로 대체
    class Args:
        config = configs
        input_folder = indir
        output_folder = outdir
        checkpoint = checkpoints
        a2b = d2n
        recon = True
        seed = 1
        num_style = 10
        synchronized = False
        output_only = False
        output_path = '.'
        trainer = 'UNIT'
        device = [str(gpu)]

    opts = Args()

    # Choose GPU device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opts.device)

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    if not opts.output_only and not os.path.exists(os.path.join(opts.output_folder, 'input')):
        os.makedirs(os.path.join(opts.output_folder, 'input'))

    # Print System Info
    print('CUDA Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print(f'Available devices {torch.cuda.device_count()}: {", ".join(str(x) for x in opts.device)}')
    print('Active CUDA Device: GPU', torch.cuda.current_device())

    print('Load experiment setting')
    config = get_config(opts.config)
    input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

    print('Setup model and data loader')
    if 'new_size' in config:
        new_size = config['new_size']
    else:
        if opts.a2b == 1:
            new_size = config['new_size_a']
        else:
            new_size = config['new_size_b']

    # 단일 이미지 로드 및 전처리
    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 이미지 데이터 타입 확인 및 변환
    if isinstance(image, io.BytesIO):
        image.seek(0)  # 스트림의 시작 부분으로 포인터 이동
        image = Image.open(image).convert('RGB')  # BytesIO를 PIL Image 객체로 변환 및 RGB로 변환
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')  # 파일 경로에서 이미지 로드 및 RGB로 변환
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')  # PIL Image 객체를 RGB로 변환
    image = transform(image).unsqueeze(0)  # 배치 차원 추가

    config['vgg_model_path'] = opts.output_path
    config['resnet_model_path'] = opts.output_path
    if opts.trainer == 'MUNIT':
        style_dim = config['gen']['style_dim']
        trainer = MUNIT_Trainer(config)
    elif opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")

    try:
        state_dict = torch.load(opts.checkpoint)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint))
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])

    trainer.cuda()
    trainer.eval()

    encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode  # encode function
    decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode  # decode function
    if opts.recon:
        decode_r = trainer.gen_b.decode if not opts.a2b else trainer.gen_a.decode  # decode for reconstruction function

    with torch.no_grad():
        t_start = time()
        if opts.trainer == 'MUNIT':

            print('Start testing')
            style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
            images = Variable(image.cuda())
            code, _ = encode(images)
            style = style_fixed if opts.synchronized else Variable(
                torch.randn(opts.num_style, style_dim, 1, 1).cuda())
            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(code, s)
                outputs = (outputs + 1) / 2.
                basename = os.path.basename(opts.input_folder)
                path = os.path.join(opts.output_folder + "_%02d" % j, basename)
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if not opts.output_only:
                # also save input images
                vutils.save_image(images.data, os.path.join(opts.output_folder, 'input/', basename),
                                  padding=0, normalize=True)
            print('Testing Complete!')
        elif opts.trainer == 'UNIT':

            print('Start testing')
            images = Variable(image.cuda())
            code, _ = encode(images)
            if opts.recon:
                reconstructed = decode_r(code)
                reconstructed = (reconstructed + 1) / 2.

            outputs = decode(code)
            outputs = (outputs + 1) / 2.

            basename = os.path.basename(opts.input_folder)
            path = os.path.join(opts.output_folder, basename)
            path_dir = os.path.dirname(path)
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if not opts.output_only:
                if not os.path.exists(os.path.join(path_dir, 'input/')):
                    os.makedirs(os.path.join(path_dir, 'input/'))
                vutils.save_image(images.data, os.path.join(path_dir, 'input/', basename), padding=0,
                                  normalize=True)
            if opts.recon:
                if not os.path.exists(os.path.join(path_dir, 'recon/')):
                    os.makedirs(os.path.join(path_dir, 'recon/'))
                vutils.save_image(reconstructed.data, os.path.join(opts.output_folder, 'recon/', basename),
                                  padding=0, normalize=True)
            print(f'{opts.input_folder} --> {opts.output_folder}')

            print('Testing Complete')
        else:
            pass
        t_fin = time() - t_start
        print(f'Time: {int(t_fin // 60)}m {int(t_fin % 60)}s | {int(t_fin)}s')



