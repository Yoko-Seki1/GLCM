# -*- coding: utf-8 -*-

"""
2021 9 27 YOKOSEKI
=========================================
Execute this program by following command.
$ python GLCM.py
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm


def img_preprocess(bgr_img):
    # 取りたい領域を指定する時
    # y,x,c = bgr_img.shape
    # bgr_img = bgr_img[round(y/2):,round(x/2):,:]
    b = bgr_img[:,:,0] ; g = bgr_img[:,:,1] ; r = bgr_img[:,:,2]
    gray_img = b*0.114 + g*0.587 + r*0.299
    gray_img = gray_img.astype(np.uint8)
    return gray_img



def show_image(heatmap, measure_name):
    """ plot heatmap """
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(heatmap,cmap="gray")
    plt.title(measure_name, fontsize=15, color="black")
    plt.show()



def save_image(heatmap, save_dir, f_name):
    """ save heatmap """
    save_path = save_dir + "/" + f_name
    cv2.imwrite(save_path, heatmap[:,:,[2,1,0]])


class glcm:
    def __init__(self, img, kernel_size=5, levels=8, angle=0):
        self.angle = angle ; self.kernel_size = kernel_size ; self.levels = levels
        self.img_size_y , self.img_size_x = img.shape
        # self.glcm = self.make_glcm(img)
        self.make_glcm(img)


    def make_glcm(self, img):
        """ GLCMを作成 """
        # GLCM の設定 (転置行列を加える + 正規化するか)
        symmetric = True ; normed = True
        # binarize
        img_bin = img//(256//self.levels) # [0:255]->[0:7]
        # make glcm
        h,w = img.shape
        glcm = np.zeros((h, w, self.levels, self.levels), dtype=np.uint8)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        if self.angle == 45 :
            img_bin_r = np.append(img_bin[1:,:], img_bin[-1,:].reshape((1,-1)), axis=0)
            img_bin_r = np.append(img_bin_r[:,1:], img_bin_r[:,-1:], axis=1)
        elif self.angle == 90 :
            img_bin_r = np.append(img_bin[1:,:], img_bin[-1,:].reshape((1,-1)), axis=0)
        elif self.angle == 135 :
            img_bin_r = np.append(img_bin[:, 0].reshape(-1,1), img_bin[:, :-1], axis=1)
            img_bin_r = np.append(img_bin_r[0, :].reshape((1,-1)), img_bin_r[:-1, :], axis=0)
        else : # include (angle == 0)
            img_bin_r = np.append(img_bin[:,1:], img_bin[:,-1:], axis=1)
        
        for i in range(self.levels):
            for j in range(self.levels):
                mask = (img_bin==i) & (img_bin_r==j)
                mask = mask.astype(np.uint8)
                glcm[:,:,i,j] = cv2.filter2D(mask, -1, kernel)                
        glcm = glcm.astype(np.float32)

        if symmetric:
            #glcm += glcm[:,:,::-1, ::-1]
            glcm += np.transpose(glcm,(0,1,3,2))
        if normed:
            glcm = glcm/glcm.sum(axis=(2,3), keepdims=True)
        # return glcm
        self.glcm = glcm


    def normalization(self, measure_map):
        # 正規化 0 〜 255
        measure_map = (measure_map - measure_map.min()) / (measure_map.max() - measure_map.min()) * 255
        heatmap = measure_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap


    def contrast(self):
        # martrix axis
        axis = np.arange(self.levels, dtype=np.float32)+1
        w = axis.reshape(1,1,-1,1)
        x = np.repeat(axis.reshape(1,-1), self.levels, axis=0)
        y = np.repeat(axis.reshape(-1,1), self.levels, axis=1)
        # GLCM contrast                                   
        glcm_contrast = np.sum(self.glcm*(x-y)**2, axis=(2,3))
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_contrast)
        return heatmap


    def dissimilarity(self):
        # martrix axis
        axis = np.arange(self.levels, dtype=np.float32)+1
        w = axis.reshape(1,1,-1,1)
        x = np.repeat(axis.reshape(1,-1), self.levels, axis=0)
        y = np.repeat(axis.reshape(-1,1), self.levels, axis=1)
        # GLCM dissimilarity                                                        
        glcm_dissimilarity = np.sum(self.glcm*np.abs(x-y), axis=(2,3))
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_dissimilarity)
        return heatmap


    def homogeneity(self):
        # martrix axis
        axis = np.arange(self.levels, dtype=np.float32)+1
        w = axis.reshape(1,1,-1,1)
        x = np.repeat(axis.reshape(1,-1), self.levels, axis=0)
        y = np.repeat(axis.reshape(-1,1), self.levels, axis=1)
        # GLCM homogeneity
        glcm_homogeneity = np.sum(self.glcm/(1.0+(x-y)**2), axis=(2,3))
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_homogeneity)
        return heatmap


    def asm(self):
        # GLCM ASM
        glcm_asm = np.sum(self.glcm**2, axis=(2,3))
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_asm)
        return heatmap


    def energy(self):
        # GLCM energy
        glcm_asm = np.sum(self.glcm**2, axis=(2,3))
        glcm_energy = np.sqrt(glcm_asm)
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_energy)
        return heatmap
    

    def entropy(self):
        # GLCM entropy
        ks = self.kernel_size # kernel_size
        pnorm = self.glcm / np.sum(self.glcm, axis=(2,3), keepdims=True) + 1./ks**2
        glcm_entropy = np.sum(-pnorm * np.log(pnorm), axis=(2,3))
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_entropy)
        return heatmap


    def mean_glcm(self):
        # martrix axis
        axis = np.arange(self.levels, dtype=np.float32)+1
        w = axis.reshape(1,1,-1,1)
        # GLCM mean                                
        glcm_mean = np.mean(self.glcm*w, axis=(2,3))
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_mean)
        return heatmap


    def std_glcm(self):
        # martrix axis
        axis = np.arange(self.levels, dtype=np.float32)+1
        w = axis.reshape(1,1,-1,1)
        # GLCM std                                          
        glcm_std = np.std(self.glcm*w, axis=(2,3))
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_std)
        return heatmap


    def max_glcm(self):
        # GLCM max                                                                  
        glcm_max = np.max(self.glcm, axis=(2,3))
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_max)
        return heatmap






    def correlation(self):
        # martrix axis
        axis = np.arange(self.levels, dtype=np.float32)+1
        x = np.repeat(axis.reshape(1,-1), self.levels, axis=0)
        y = np.repeat(axis.reshape(-1,1), self.levels, axis=1)
        # GLCM correlation (自作)
        # ===========================
        wy = axis.reshape(1,1,-1,1)
        wx = axis.reshape(1,1,1,-1)
        mean_x = np.mean(self.glcm*wx, axis=(2,3))
        mean_y = np.mean(self.glcm*wy, axis=(2,3))

        mean_tempx = np.tile(mean_x.reshape((mean_x.shape[0], mean_x.shape[1], 1)),(1,1,self.levels))
        mean_tempy = np.tile(mean_y.reshape((mean_y.shape[0], mean_y.shape[1], 1)),(1,1,self.levels))
        axis_mat = np.tile(axis, (self.img_size_y, self.img_size_x, 1))

        sigma_x = np.sum((axis_mat-mean_tempx)*np.sum(self.glcm,axis=2), axis=2)
        sigma_x = np.sqrt(sigma_x)
        sigma_y = np.sum((axis_mat-mean_tempy)*np.sum(self.glcm,axis=3), axis=2)
        sigma_y = np.sqrt(sigma_y)
        temp = np.ones((self.img_size_y, self.img_size_x, self.levels, self.levels))
        mean_tempx = mean_x.reshape((mean_x.shape[0], mean_x.shape[1], 1, 1))
        mean_tempy = mean_y.reshape((mean_y.shape[0], mean_y.shape[1], 1, 1))
        temp_x = (temp*x)*mean_tempx
        temp_y = (temp*y)*mean_tempy

        glcm_correlation = np.sum((temp_x*temp_y)*self.glcm, axis=(2,3)) / (sigma_x*sigma_y)
        # ===== 正規化 0 〜 255=====
        heatmap = self.normalization(glcm_correlation)
        return heatmap



if __name__ =="__main__":
    
    # 2021_05_31/img_818.jpg == sample.jpg
    img = cv2.imread("sample.jpg")

    # グレースケール化(RGB to gray scale)
    img = img_preprocess(img)
    # GLCM(のインスタンス)を生成
    GLCM = glcm(img)
    
    heatmap = GLCM.correlation()
    show_image(heatmap,"correlation")

    heatmap = GLCM.contrast()
    show_image(heatmap,"contrast")
    
    heatmap = GLCM.dissimilarity()
    show_image(heatmap,"dissimilarity")
    
    heatmap = GLCM.homogeneity()
    show_image(heatmap,"homogeneity")
    
    heatmap = GLCM.asm()
    show_image(heatmap,"asm")
    
    heatmap = GLCM.energy()
    show_image(heatmap,"energy")
    
    heatmap = GLCM.entropy()
    show_image(heatmap,"entropy")

    heatmap = GLCM.mean_glcm()
    show_image(heatmap,"mean")
    
    heatmap = GLCM.std_glcm()
    show_image(heatmap,"std")
    
    heatmap = GLCM.max_glcm()
    show_image(heatmap,"max")
