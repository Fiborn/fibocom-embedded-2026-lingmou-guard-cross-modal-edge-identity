# _*_ coding:utf-8 _*_
import os
import random
import argparse
import sys


class GeneratePairs:
    """
    Generate the pairs.txt file for applying "validate on LFW" on your own datasets.
    """

    # 写成命令行格式就用args解析参数
    # def __init__(self, args):
    #     """
    #     Parameter data_dir, is your data directory.
    #     Parameter pairs_filepath, where is the pairs.txt that belongs to.
    #     Parameter img_ext, is the image data extension for all of your image data.
    #     """
    #     self.data_dir = args.data_dir
    #     self.data_dir =self.data_dir + "/"
    #     self.pairs_filepath = args.saved_dir + "/" + 'pairs.txt'
    #     self.repeat_times = int(args.repeat_times)
    #     self.img_ext = '.png'
    # 在pycharm上直接运行，就用这种直接修改参数的方法比较方便，自己选择
    def __init__(self):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir_nir = 'D:/AAAAJK/facenet-pytorch-main/datasets/vis_nir/nir/'
        self.data_dir_vis = 'D:/AAAAJK/facenet-pytorch-main/datasets/vis_nir/vis/'
        # self.data_dir = self.data_dir + "\\"  # 验证集路径
        self.pairs_filepath = 'D:/AAAAJK/facenet-pytorch-main/my_pairs.txt'   # pairs.txt存放路径
        self.repeat_times = int(300)
        # self.img_ext = '.png'  #因为我自己验证集png和jpg格式都有，所以不固定图片格式后缀

    def generate(self):
        # The repeate times. You can edit this number by yourself
        folder_number = self.get_folder_numbers()
        print('folder_number--', folder_number)
        # This step will generate the hearder for pair_list.txt, which contains
        # the number of classes and the repeate times of generate the pair
        # 如果存在旧的pairs先删除
        if os.path.exists(self.pairs_filepath):
            os.remove(self.pairs_filepath)
        # 删完重开一个pair.txt
        if not os.path.exists(self.pairs_filepath):
            with open(self.pairs_filepath, "x") as f:
                f.write(str(self.repeat_times) + "\t" + str(folder_number) + "\n")
        self.nir_dict, self.nir_dict_keys, self.vis_dict, self.vis_dict_keys = self._generate_dicts_pairs()
        for nuum_reapeat in range(10):
            for i in range(self.repeat_times):
                print('第 %d 次：' % int(i)+'same')
                self.get_image_pair('same')
            for i in range(self.repeat_times):
                print('第 %d 次：' % int(i)+'differ')
                self.get_image_pair('differ')
                # self._generate_mismatches_pairs()

    def get_folder_numbers(self):
        count = 0
        for folder in os.listdir(self.data_dir_vis):
            if os.path.isdir(self.data_dir_vis + folder):
                count += 1
        return count

    def _generate_dicts_pairs(self):
        """
        Generate all matches pairs
        """
        vis_dict={}
        nir_dict={}
        ## vis图像遍历产生字典文件用于配对
        for person_name in os.listdir(self.data_dir_vis):
            a = []
            for person_name_img_id in os.listdir(self.data_dir_vis + person_name):
                a.append(person_name_img_id.split('.')[0])
            a.sort(key=lambda x: int(x))
            vis_dict[person_name] = a
        ## nir图像遍历产生字典文件用于配对
        for person_name in os.listdir(self.data_dir_nir):
            b = []
            for person_name_img_id in os.listdir(self.data_dir_nir+ person_name):
                b.append(person_name_img_id.split('.')[0])
            b.sort(key=lambda x: int(x))
            nir_dict[person_name] = b
        vis_dict_keys = list(vis_dict.keys())
        nir_dict_keys = list(nir_dict.keys())

        return nir_dict, nir_dict_keys, vis_dict, vis_dict_keys

    def get_image_pair(self, mode='same'):
        if mode=='same':
            person_name_nir = self.nir_dict_keys[random.randint(0,len(self.nir_dict_keys)-1)]
            person_name_vis = person_name_nir[:-4]+'_vis'
            nir_person_image_id = self.nir_dict[person_name_nir][random.randint(0,len(self.nir_dict[person_name_nir])-1)]
            vis_person_image_id = self.vis_dict[person_name_vis][random.randint(0,len(self.vis_dict[person_name_vis])-1)]
            with open(self.pairs_filepath, "a") as f:
                f.write('0\t'+person_name_nir+'\t'+nir_person_image_id+'\t'+person_name_vis+'\t'+vis_person_image_id+'\n')
        elif mode=='differ':
            while True:
                person_name_nir = self.nir_dict_keys[random.randint(0, len(self.nir_dict_keys)-1)]
                person_name_vis = self.vis_dict_keys[random.randint(0, len(self.vis_dict_keys)-1)]
                if person_name_nir[:-4] != person_name_vis[:-4]:
                    break
            nir_person_image_id = self.nir_dict[person_name_nir][random.randint(0, len(self.nir_dict[person_name_nir])-1)]
            vis_person_image_id = self.vis_dict[person_name_vis][random.randint(0, len(self.vis_dict[person_name_vis])-1)]
            with open(self.pairs_filepath, "a") as f:
                f.write('1\t'+person_name_nir+'\t'+nir_person_image_id+'\t'+person_name_vis+'\t'+vis_person_image_id+'\n')


if __name__ == '__main__':
    gen = GeneratePairs()
    gen.generate()