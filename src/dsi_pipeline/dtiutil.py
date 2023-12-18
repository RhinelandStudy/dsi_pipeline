"""
Copyright 2023 Population Health Sciences, German Center for Neurodegenerative Diseases (DZNE)
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0 
Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""

from __future__ import division
import os
import os.path as op
import json
import numpy as np

from nipype.interfaces.fsl.base import (CommandLine, CommandLineInputSpec,
                                        FSLCommand, FSLCommandInputSpec)

from nipype.interfaces.base import (traits, TraitedSpec, File, isdefined)
import warnings

from nipype.interfaces.freesurfer.base import (FSCommand, FSTraitedSpec, Info)
from nipype import LooseVersion
from nipype.utils.filemanip import fname_presuffix

def clip_max(in_file):
    import os
    import numpy as np
    import nibabel as nb

    #load file
    img=nb.load(in_file)

    #get data
    aff=img.affine
    data=img.get_data()
    header=img.header

    max_kurtosis=10.0
    datanew=data.clip(max=max_kurtosis)

    outnii=nb.Nifti1Image(datanew,aff, header=header)
    out_file = os.path.abspath(os.path.join(os.getcwd(),'DK_RK.nii.gz'))

    nb.save(outnii,filename=out_file)
    return os.path.abspath(out_file)

def crop_xy(in_file):
    import os
    import numpy as np
    import nibabel as nb

    #load file
    img=nb.load(in_file)

    #get data
    aff=img.affine
    data=img.get_data()
    header=img.header
    #Ref:https://note.nkmk.me/en/python-numpy-delete/
    #x indices: 1st and last slice
    #y indices: 1st and last slice
    x_ind=[0, data.shape[0]-1 ]
    y_ind=[0, data.shape[1]-1 ]

    datatemp=np.delete(data, x_ind, axis=0)
    datanew=np.delete(datatemp, y_ind, axis=1)

    outnii=nb.Nifti1Image(datanew,aff, header=header)
    out_file=os.path.basename(in_file)

    nb.save(outnii,filename=out_file)
    return os.path.abspath(out_file)


def copy_pad_slices(in_file):
    import os
    import numpy as np
    import nibabel as nb

    #load file
    img=nb.load(in_file)

    #get data
    aff=img.affine
    data=img.get_data()
    header=img.header

    #pad 0s 1 slice on top and 2 slices on bottom of 3rd dimension
    zpad = ((0, 0), (0, 0), (1, 2),(0,0))
    datanew=np.pad(data,pad_width=zpad, mode='constant', constant_values=0)
    #copy nighboring slices
    datanew[:,:,0,:] = datanew[:,:,1,:]
    datanew[:,:,94,:]= datanew[:,:,93,:]
    datanew[:,:,95,:]= datanew[:,:,93,:]

    outnii=nb.Nifti1Image(datanew,aff,header)
    out_file=os.path.basename(in_file)

    nb.save(outnii,filename=out_file)

    return os.path.abspath(out_file)


def pad_slices(in_file):
    import os
    import numpy as np
    import nibabel as nb

    #load file
    img=nb.load(in_file)

    #get data
    aff=img.affine
    data=img.get_data()
    header=img.header

    #pad 0s 1 slice on top and 2 slices on bottom of 3rd dimension
    zpad = ((0, 0), (0, 0), (1, 2),(0,0))

    datanew=np.pad(data,pad_width=zpad, mode='constant', constant_values=0)

    outnii=nb.Nifti1Image(datanew,aff,header)
    out_file=os.path.basename(in_file)

    nb.save(outnii,filename=out_file)

    return os.path.abspath(out_file)

def trim_slices(in_file,sl_idx):
    import os
    import numpy as np
    import nibabel as nb
    #from shutil import copyfile

    #load file
    img=nb.load(in_file)

    #get data
    aff=img.affine
    data=img.get_data()
    header=img.header
    #Ref:https://note.nkmk.me/en/python-numpy-delete/
    #delete the 1st, and last 2 slices from the 3rd dim: ([0,94,95] for tc/tcbm, [34] for tcfc
    datanew=np.delete(data,sl_idx,axis=2)
    #modify the qoffset_z from 96 to 93 if input is tc fieldcoef
    if '_fieldcoef' in in_file:
        header['qoffset_z']=93.0
        sform_new=header.get_sform()
        sform_new[0:3,3]=[140.,140.,93.]
        header.set_sform(sform_new)
        header['sform_code']=1
        outnii=nb.Nifti1Image(datanew,None,header=header)
    else:
        outnii=nb.Nifti1Image(datanew,aff,header=header)

    #replace in_file in its source dir if fieldcoef
    if 'topup' in in_file:
        #in_file_copy='orig_' + in_file
        #copyfile(in_file, in_file_copy)
        out_file=in_file
    else:
        out_file=os.path.basename(in_file)

    nb.save(outnii,filename=out_file)
    return os.path.abspath(out_file)


def load_data(data_path):
    import numpy as np
    import nibabel as nib

    img = nib.load(data_path)
    data = img.get_data()
    data = np.array(data)
    affine = img.affine
    return data, affine



def b_threshold(bvals_file,bvecs_file,dsi_file):
    import os
    import numpy as np
    import nibabel as nib
    from dipy.io import read_bvals_bvecs


    b_thresh=1600
    img=nib.load(dsi_file)
    data=img.get_data()

    reduced_dsi = os.path.abspath(os.path.join(os.getcwd(),'dsi_b1600.nii.gz'))
    reduced_bvals = os.path.abspath(os.path.join(os.getcwd(),'dsi.bvals'))
    reduced_bvecs = os.path.abspath(os.path.join(os.getcwd(),'dsi.bvecs'))

    bvals,bvecs=read_bvals_bvecs(bvals_file,bvecs_file)

    bval_new = np.empty([1,bvals[(bvals<b_thresh)].shape[0]])
    bval_new[0,...] = bvals[(bvals<b_thresh)].astype(np.int)

    np.savetxt(reduced_bvals, bval_new, fmt=str("%i"),delimiter=' ')
    np.savetxt(reduced_bvecs, np.transpose(bvecs[(bvals<b_thresh),...]), fmt=str("%.14g"), delimiter=' ')

    nib.save(nib.Nifti1Image(data[... ,(bvals<b_thresh)].astype(np.float32), img.get_affine()), reduced_dsi)

    return reduced_bvals, reduced_bvecs, reduced_dsi

def create_mask(aseg, dnum, enum):
    from skimage.morphology import binary_dilation, binary_erosion
    from skimage.measure import label
    import numpy as np

    data = aseg.get_data()
    # reduce to binary
    datab = (data>0)
    # dilate and erode
    for x in range(dnum):
        datab = binary_dilation(datab,np.ones((3,3,3)))
    for x in range(enum):
        datab = binary_erosion(datab,np.ones((3,3,3)))
    # extract largest component
    labels = label(datab)
    assert( labels.max() != 0 ) # assume at least 1 real connected component
    if (labels.max() > 1):
        datab = (labels == np.argmax(np.bincount(labels.flat)[1:])+1)
    # set mask
    data[~datab] = 0
    data[datab] = 1
    return aseg

def aseg_mask(in_file):
    import os.path as op
    import numpy as np
    import nibabel as nib
    import copy
    from dsi_pipeline.dtiutil import create_mask

    #file_name
    fname, ext = op.splitext(op.basename(in_file))
    if ext == ".gz":
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext

    #load aseg
    aseg = nib.load(in_file)
    aseg.set_data_dtype(np.uint8)

    #generate brainmask from aseg
    aseg_bm = create_mask(copy.deepcopy(aseg),5,4)
    aseg_bm.set_data_dtype(np.uint8)
    out_file = fname + '_mask.nii.gz'
    nib.save(aseg_bm, out_file)

    # mask aseg
    data = aseg.get_data()
    data[aseg_bm.get_data()==0] = 0
    aseg_masked_file = fname + '_masked.nii.gz'
    nib.save(aseg, aseg_masked_file)

    return op.abspath(out_file)

def get_wm_mask(in_brain_mask,in_aseg,in_FA,in_MD):
    import numpy as np
    import nibabel as nib
    from dsi_pipeline.dtiutil import load_data
    import os
    from scipy.stats import iqr

    mask_brain_data_data,aff=load_data(in_brain_mask)
    mask_brain=mask_brain_data_data>0
    mask_aseg_data,mask_aseg_aff=load_data(in_aseg)
    mask_aseg=mask_aseg_data.astype(int)
    FA_data,FA_aff=load_data(in_FA)
    MD_data,MD_aff=load_data(in_MD)

    #7,46 for cerebellum; include CC for complete mask, but not for dMRI analysis, 16... brainstem, ventralDC, 10...deep GM
    mask_aseg_labels=[2,41,77,30,62,251,252,253,254,255,7,46,  16,28,60,  10,11,12,13,17,18,26,  49,50,51,52,53,54,58]

    #ASEG WM mask
    mask=np.zeros_like(mask_brain)
    for i in range(0,len(mask_aseg_labels)):
        if np.sum(mask_aseg==mask_aseg_labels[i]):
            mask=mask+(mask_aseg==mask_aseg_labels[i])
    data_wm_mask_aseg=(mask_brain*mask)>0

    #FA thresholded ASEG WM mask
    fa_tresh=0.3
    mask_FA= np.squeeze(FA_data)>=fa_tresh
    mask=(data_wm_mask_aseg*mask_FA)>0

    #Removal of MD contamination inside FA thresholded ASEG WM mask
    data_md= np.squeeze(MD_data)
    mask_MD= data_md<(np.median(data_md[mask>0.0])+1.5*iqr(data_md[mask>0.0]))
    data_wm_mask_aseg_FAMD=(mask*mask_MD)>0

    wm_mask_aseg = os.path.abspath(os.path.join(os.getcwd(),'wm_mask_aseg.nii.gz'))
    wm_mask_aseg_FAMD = os.path.abspath(os.path.join(os.getcwd(),'wm_mask_aseg_FAMD.nii.gz'))

    nib.save(nib.Nifti1Image(data_wm_mask_aseg.astype(np.uint8), aff), wm_mask_aseg)
    nib.save(nib.Nifti1Image(data_wm_mask_aseg_FAMD.astype(np.uint8), aff), wm_mask_aseg_FAMD)

    return os.path.abspath(wm_mask_aseg),os.path.abspath(wm_mask_aseg_FAMD)


def sci_format(x,prec,sci):
    import numpy as np

    if sci:
        return np.format_float_scientific(x, unique=False, precision=prec)
    else:
        return round(x,prec)

def NMSE(GT, estim):
    import numpy as np
    return np.sum((GT-estim)**2)/np.sum(GT**2)


def get_csrecon_error(filename_bval,filename_meas,filename_csrecon,filename_csrecon_idx,mask):
    import numpy as np
    import nibabel as nib
    import os
    from dipy.io import read_bvals_bvecs
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    idx_recon=np.loadtxt(filename_csrecon_idx)[1:] >0
    half_cart=int(len(idx_recon)/2)
    idx_recon_meas=idx_recon[:half_cart]
    len_meas=int(np.sum(idx_recon)/2)
    ind=np.arange(0,len_meas)

    #bvals measurements
    bvals, bvecs = read_bvals_bvecs(filename_bval,None)
    bvals=bvals.astype(int)
    numb0=np.sum(bvals==0)
    bvals_meas=((bvals[numb0:])[idx_recon_meas])
    bvals_round=(np.round(bvals_meas/10.0)*10.0).astype(int)

    #load data
    data_meas=nib.load(filename_meas).get_data()[...,numb0:]
    data_csrecon=nib.load(filename_csrecon).get_data()[...,numb0:]
    data_csrecon_meas=data_csrecon[...,idx_recon_meas]

    #remove negative or nan voxel from mask
    #print np.sum(mask)
    for i in range(0,len_meas):
        mask[np.isnan(data_meas[...,i])]=0
        mask[np.isnan(data_csrecon_meas[...,i])]=0
        mask[data_meas[...,i]<=0.0]=0
        mask[data_csrecon_meas[...,i]<=0.0]=0

    #ERROR: MEASUREMENTS - CS RECON OF MEASUREMENTS
    nmse_meas_csrecon=np.zeros(len_meas)

    nmse_stats = {}

    for i in range(0,len_meas):
        nmse_meas_csrecon[i]= NMSE(data_meas[mask,i],data_csrecon_meas[mask,i])
        nmse_stats.update({"b"+str(bvals_round[i]) : sci_format(float( nmse_meas_csrecon[i]),6,False)})

    error_mean=np.mean(nmse_meas_csrecon)
    error_std= np.std(nmse_meas_csrecon)
    nmse_stats.update({"mean": sci_format(float( error_mean),6,False)})
    nmse_stats.update({"std": sci_format(float( error_std),6,False)})

    #plot NMSE
    #matplotlib settings
    plt.style.use('ggplot')
    figw=6.92
    figh=3.2
    NIsize=10
    dpi_fig=500
    font = {'weight' : 'normal',
            'size'   : NIsize}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['xtick.major.pad']='0'
    matplotlib.rcParams['ytick.major.pad']='0'

    #settings figure
    fig, ax = plt.subplots(1,figsize=(figw, figh)) #*3
    fig.subplots_adjust(left=0.11,right=0.96,top=0.94,bottom=0.15)
    col=('coral','r','m','lightgreen','g','b')
    ax.set_xlim(-1,len(ind))
    ax.set_xlabel('Diffusion weighting '+r'$[s/mm^2]$',fontsize=NIsize)
    ax.set_ylabel('NMSE: Measurements - CSrecon',fontsize=NIsize)
    #xaxis label

    step=10
    off=1
    ax.set_xticks(ind[off::step])
    bvals_10=bvals_round[off::step]
    def format_func(value, tick_number):
        return bvals_10[tick_number]
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    #plot nmse
    ax.plot(ind,nmse_meas_csrecon,color=col[0])
    #save figure
    qc_nmse = os.path.abspath(os.path.join(os.getcwd(),'QC_csrecon_nmse.png'))
    plt.savefig(qc_nmse, dpi=dpi_fig, facecolor=fig.get_facecolor())

    return nmse_stats,os.path.abspath(qc_nmse)

def plot_motion_abs_rel_trans_rot(bvals_meas,motion_rot_trans,motion_abs,motion_rel):
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    figw=6.92
    figh=3.2
    NIsize=10
    dpi_fig=500
    font = {'weight' : 'normal',
            'size'   : NIsize}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['xtick.major.pad']='0'
    matplotlib.rcParams['ytick.major.pad']='0'

    ind=np.arange(0,len(bvals_meas))
    b0_idx=np.where(bvals_meas==0)[0]
    dwi_idx=np.where(bvals_meas>0)[0]

    #absolute and relative motion
    fig, ax_all = plt.subplots(3,figsize=(figw, figh*2.5)) #*3
    fig.subplots_adjust(left=0.11,right=0.96,top=0.96,bottom=0.07,hspace=0.2)
    ax=ax_all[0]
    col=('coral','r','cornflowerblue','b')
    lab=('Absolute','Relative')
    ax.set_xlim(-1,len(ind))
    ax.set_ylim(np.amin([-0.2,np.amin(motion_abs),np.amin(motion_rel)]),np.amax([5.2,np.amax(motion_abs),np.amax(motion_rel)]))
    ax.set_ylabel('RMS displacement (mm)',fontsize=NIsize)

    step=9
    off=0
    ax.set_xticks(ind[off::step])
    bvals_10=((np.round(bvals_meas/10.0)*10.0).astype(int))[off::step]
    def format_func(value, tick_number):
        return bvals_10[tick_number]
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.plot(ind,motion_abs,color=col[0],label=lab[0])
    ax.plot(ind[b0_idx],motion_abs[b0_idx],'o',markersize=4,color=col[1],label=lab[0]+' b0')
    ax.plot(ind,motion_rel,color=col[2],label=lab[1])
    ax.plot(ind[b0_idx],motion_rel[b0_idx],'o',markersize=4,color=col[3],label=lab[1]+' b0')
    ax.legend(loc='upper left',ncol=2, frameon=False,fontsize=NIsize)

    #x,y,z translation
    ax=ax_all[1]
    col=('r','g','b','m','lightgreen','c')
    lab=('x','y','z','x','y','z')
    ax.set_xlim(-1,len(ind))
    ax.set_ylim(np.amin([-5.2,np.amin(motion_rot_trans[:,:3].flatten())]),np.amax([5.2,np.amax(motion_rot_trans[:,:3].flatten())]))
    ax.set_ylabel('Translation (mm)',fontsize=NIsize)
    ax.set_xticks(ind[off::step])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    for i in range(0,3):
        ax.plot(ind,motion_rot_trans[:,i],color=col[i],label=lab[i])
        ax.plot(ind[b0_idx],motion_rot_trans[b0_idx,i],'o',markersize=4,color=col[i],label=lab[i]+' b0')
    ax.legend(loc='upper left',ncol=3, frameon=False,fontsize=NIsize)

    #x,y,z rotation
    ax=ax_all[2]
    ax.set_xlim(-1,len(ind))
    ax.set_ylim(np.amin([-5.2,np.amin(motion_rot_trans[:,3:6].flatten())]),np.amax([5.2,np.amax(motion_rot_trans[:,3:6].flatten())]))
    ax.set_xlabel('Diffusion weighting '+r'$[s/mm^2]$',fontsize=NIsize)
    ax.set_ylabel('Rotation (deg)',fontsize=NIsize)
    ax.set_xticks(ind[off::step])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    for i in range(3,6):
        ax.plot(ind,motion_rot_trans[:,i],color=col[i],label=lab[i])
        ax.plot(ind[b0_idx],motion_rot_trans[b0_idx,i],'o',markersize=4,color=col[i],label=lab[i]+' b0')
    ax.legend(loc='upper left',ncol=3, frameon=False,fontsize=NIsize)

    #save figure
    qc_abs_rel_trans_rot = os.path.abspath(os.path.join(os.getcwd(),'QC_motion_abs_rel_trans_rot.png'))
    plt.savefig(qc_abs_rel_trans_rot, dpi=dpi_fig, facecolor=fig.get_facecolor())

    return os.path.abspath(qc_abs_rel_trans_rot)


def plot_motion_abs_before_after(data_eddy_corrected_meas,data_scan):
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    figw=6.92
    figh=3.2
    NIsize=10
    dpi_fig=500
    font = {'weight' : 'normal',
            'size'   : NIsize}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['xtick.major.pad']='0'
    matplotlib.rcParams['ytick.major.pad']='0'

    sl_sag=[45,57,69,81,93] #slice 46,58,70,82,94
    sl_cor=[45,57,69,81,93] #slice 46,58,70,82,94
    sl_ax=[25,37,49,61,73]  #slice 26,38,50,62,74

    #settings figure images
    f, ax_all = plt.subplots(2*len(sl_sag),6,figsize=(figw, figh*3))
    f.subplots_adjust(left=0.02, right=0.96, wspace=0,hspace=0.0,top=1.2,bottom=0.0)
    f.patch.set_facecolor('k')

    #data start, abs, rel1, rel2
    idx_plot=[0,119]
    data_ax_scan= data_scan[...,idx_plot]
    data_ax= data_eddy_corrected_meas[...,idx_plot]

    #differnt intensity range for old and new RLS DSI sequence
    lim_max_scan=np.amax(data_ax_scan[...,0])
    lim_max=np.amax(data_ax[...,0])
    if lim_max_scan>10000.:
        lim_max=20000
    else:
        lim_max=2000

    for sl_id in range(0,len(sl_sag)):
        for j in range(0,3):
            for i in range(0, len(idx_plot)):

                ax = ax_all[sl_id*2,j*2+i]

                if j==2:
                    cb1=ax.imshow(np.rot90(data_ax_scan[sl_sag[sl_id],:,:,i]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                    yl=48
                if j==0:
                    cb1=ax.imshow(np.rot90(data_ax_scan[:,:,sl_ax[sl_id],i]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                    yl=70
                if j==1:
                    cb1=ax.imshow(np.rot90(data_ax_scan[:,sl_cor[sl_id],:,i]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                    yl=48

                if sl_id==0:
                    if j:
                        ax.set_title('Volume '+str(idx_plot[i]+1),fontsize=NIsize,color='white')
                        ax.title.set_position([0.5,1.24])
                    else:
                        ax.set_title('Volume '+str(idx_plot[i]+1),fontsize=NIsize,color='white')

                if j+i==0:
                    ax.set_ylabel('Before',fontsize=NIsize,labelpad=-4,zorder=100)
                    ax.yaxis.label.set_color('white')

                ax.axvline(x=70,ls='--',color='red',linewidth=0.5)
                ax.axhline(y=yl,ls='--',color='red',linewidth=0.5)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.patch.set_visible(False)

        for j in range(0,3):
            for i in range(0, len(idx_plot)):

                ax = ax_all[sl_id*2+1,j*2+i]

                if j==2:
                    cb1=ax.imshow(np.rot90(data_ax[sl_sag[sl_id],:,:,i]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                    yl=48
                if j==0:
                    cb1=ax.imshow(np.rot90(data_ax[:,:,sl_ax[sl_id],i]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                    yl=70
                if j==1:
                    cb1=ax.imshow(np.rot90(data_ax[:,sl_cor[sl_id],:,i]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                    yl=48

                if i:
                    if j==0:
                        ax.set_title('Slice '+str(sl_ax[sl_id]+1),fontsize=NIsize,color='white')
                        ax.title.set_position([0.0,0.9])
                    if j==1:
                        ax.set_title('Slice '+str(sl_cor[sl_id]+1),fontsize=NIsize,color='white')
                        ax.title.set_position([0.0,1.1])
                    if j==2:
                        ax.set_title('Slice '+str(sl_sag[sl_id]+1),fontsize=NIsize,color='white')
                        ax.title.set_position([0.0,1.1])

                if j+i==0:
                    ax.set_ylabel('After',fontsize=NIsize,labelpad=-4,zorder=100)
                    ax.yaxis.label.set_color('white')

                ax.axvline(x=70,ls='--',color='red',linewidth=0.5)
                ax.axhline(y=yl,ls='--',color='red',linewidth=0.5)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.patch.set_visible(False)

    #save figure
    qc_abs_before_after = os.path.abspath(os.path.join(os.getcwd(),'QC_motion_abs_before_after.png'))
    plt.savefig(qc_abs_before_after, dpi=dpi_fig, facecolor=f.get_facecolor(),bbox_inches = 'tight')
    return os.path.abspath(qc_abs_before_after)

def plot_motion_outlier(bvals_meas,motion_outl_mat,data_eddy_corrected_meas,data_scan):
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    figw=6.92
    figh=3.2
    NIsize=10
    NIsizeSub=8
    dpi_fig=500
    font = {'weight' : 'normal',
            'size'   : NIsize}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['xtick.major.pad']='0'
    matplotlib.rcParams['ytick.major.pad']='0'

    #load bvals once
    ind=np.arange(0,len(bvals_meas))
    b0_idx=np.where(bvals_meas==0)[0]
    dwi_idx=np.where(bvals_meas>0)[0]

    outl_perc_subj=np.zeros(motion_outl_mat.shape[1])
    for i in range(0,motion_outl_mat.shape[1]):
        outl_perc_subj[i]=100*np.sum(np.abs(motion_outl_mat[:,i])>0)/motion_outl_mat.shape[0]
    xlim_max=np.amax([10.,np.amax(outl_perc_subj)])+5.

    #outlier percentage
    fig, ax_all = plt.subplots(2,figsize=(figw, figh*2))
    fig.subplots_adjust(left=0.11,right=0.96,top=0.96,bottom=0.08,hspace=0.2)
    ax=ax_all[0]
    col=('coral','r','cornflowerblue','b','m','lightgreen','g','b')
    ax.set_xlim(-1,len(ind))
    ax.set_ylim(-1.0,xlim_max)
    ax.set_ylabel('Outlier (%)',fontsize=NIsize)

    step=9
    off=0
    ax.set_xticks(ind[off::step])
    bvals_10=((np.round(bvals_meas/10.0)*10.0).astype(int))[off::step]
    def format_func(value, tick_number):
        return bvals_10[tick_number]
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.plot(ind,outl_perc_subj,color=col[0])
    ax.plot(ind[b0_idx],outl_perc_subj[b0_idx],'o',markersize=4,color=col[1])
    asp= ax.get_aspect()

    #outlier std matrix
    ax=ax_all[1]
    ax.set_xlabel('Diffusion weighting '+r'$[s/mm^2]$',fontsize=NIsize)
    ax.set_ylabel('Slice',fontsize=NIsize)
    ax.set_title('# of standard deviations away from mean slice-difference',fontsize=NIsize)
    ax.set_xticks(ind[off::step])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    clim_min=np.amin([-15.,np.amin(motion_outl_mat)])
    h=ax.imshow(motion_outl_mat,interpolation='none',origin='lower',cmap=plt.cm.gist_heat, clim=(clim_min, 0.0),aspect=asp)
    ax.tick_params(top=False,right=False)
    cax = fig.add_axes([0.27, 0.48, 0.5, 0.01])
    cbar = fig.colorbar(h,cax=cax,orientation="horizontal")
    cbar.ax.tick_params(labelsize=NIsizeSub)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, pos: '{}'.format((y+1).astype(int))))

    qc_outlier_perc_mat = os.path.abspath(os.path.join(os.getcwd(),'QC_motion_outlier_perc_mat.png'))
    plt.savefig(qc_outlier_perc_mat, dpi=dpi_fig, facecolor=fig.get_facecolor())


    #plot motion outlier images
    perc_thresh=5.0
    idx_outl=np.where(outl_perc_subj>=perc_thresh)[0]

    #no or just one outlier: plot two highest error
    if idx_outl.size<=1:
        idx_outl=np.sort(np.argsort(outl_perc_subj)[-2:])

    #too many outliers for plot: limit to 10 highest errors
    outl_max=10
    if idx_outl.size>outl_max:
        sort_perc_id=np.argsort(outl_perc_subj[idx_outl])[::-1]
        idx_outl=idx_outl[np.sort(sort_perc_id[:outl_max])]

    sl_sag=60
    sl_cor=70
    sl_ax=np.argmin(motion_outl_mat[:,idx_outl],axis=0)
    sl_ax_std=np.min(motion_outl_mat[:,idx_outl],axis=0)

    data_ax_scan= data_scan[...,idx_outl]
    data_ax= data_eddy_corrected_meas[...,idx_outl]

    figh_1row=(figh*3.)/10.
    f, ax_all = plt.subplots(len(idx_outl),6,figsize=(figw, figh_1row*len(idx_outl)))
    f.subplots_adjust(left=0.02, right=0.96, wspace=0,hspace=0.0,top=1.2,bottom=0.0)
    f.patch.set_facecolor('k')

    for sl_id in range(0,len(idx_outl)):
        for j in range(0,3):
            for i in range(0, 2):

                ax = ax_all[sl_id,j*2+i]

                if j==2:
                    if i:
                        cb1=ax.imshow(np.rot90(data_ax[sl_sag,:,:,sl_id]), cmap='gray',interpolation='None')#,clim=(0.0, lim_max))
                    else:
                        cb1=ax.imshow(np.rot90(data_ax_scan[sl_sag,:,:,sl_id]), cmap='gray',interpolation='None')#,clim=(0.0, lim_max))
                    yl=48
                if j==0:
                    if i:
                        cb1=ax.imshow(np.rot90(data_ax[:,:,sl_ax[sl_id],sl_id]), cmap='gray',interpolation='None')#,clim=(0.0, lim_max))
                        str_xlab='Slice {:d}, {:.2f} stds away'.format(sl_ax[sl_id]+1,sl_ax_std[sl_id])
                        ax.text(-120,145,str_xlab,fontsize=NIsize,color='white')
                    else:
                        cb1=ax.imshow(np.rot90(data_ax_scan[:,:,sl_ax[sl_id],sl_id]), cmap='gray',interpolation='None')#,clim=(0.0, lim_max))
                    yl=70
                if j==1:
                    if i:
                        cb1=ax.imshow(np.rot90(data_ax[:,sl_cor,:,sl_id]), cmap='gray',interpolation='None')#,clim=(0.0, lim_max))
                    else:
                        cb1=ax.imshow(np.rot90(data_ax_scan[:,sl_cor,:,sl_id]), cmap='gray',interpolation='None')#,clim=(0.0, lim_max))
                    yl=48

                if sl_id==0:
                    if j:
                        if i:
                            ax.set_title('After',fontsize=NIsize,color='white')
                        else:
                            ax.set_title('Before',fontsize=NIsize,color='white')
                        ax.title.set_position([0.5,1.24])
                    else:
                        if i:
                            ax.set_title('After',fontsize=NIsize,color='white')
                        else:
                            ax.set_title('Before',fontsize=NIsize,color='white')

                if j+i==0:
                    ax.set_ylabel('Volume '+str(idx_outl[sl_id]+1),fontsize=NIsize,labelpad=-4,zorder=100)
                    ax.yaxis.label.set_color('white')

                ax.axvline(x=70,ls='--',color='red',linewidth=0.5)
                ax.axhline(y=yl,ls='--',color='red',linewidth=0.5)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.patch.set_visible(False)

    qc_outlier= os.path.abspath(os.path.join(os.getcwd(),'QC_motion_outlier.png'))
    plt.savefig(qc_outlier, dpi=dpi_fig, facecolor=f.get_facecolor(),bbox_inches = 'tight')

    return os.path.abspath(qc_outlier_perc_mat),os.path.abspath(qc_outlier)


def plot_labels_over_FA(data_FA,mask_brain,mask_WM,mask_skelet,mask_roi):
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    figw=6.92
    figh=3.2
    NIsize=10
    dpi_fig=500
    font = {'weight' : 'normal',
            'size'   : NIsize}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['xtick.major.pad']='0'
    matplotlib.rcParams['ytick.major.pad']='0'

    sl_step=5
    sl_ax_step=5
    sl_sag= np.arange(0,data_FA.shape[0])[49:95:sl_step]
    sl_cor=np.arange(0,data_FA.shape[1])[49:95:sl_step]
    sl_ax=np.arange(0,data_FA.shape[2])[29:75:sl_ax_step]

    #settings figure images
    f, ax_all = plt.subplots(len(sl_sag),6,figsize=(figw, figh*3)) #*3
    f.subplots_adjust(left=0.02, right=0.96, wspace=0,hspace=0.0,top=1.2,bottom=0.0)
    f.patch.set_facecolor('k')

    data_ax_FA=np.squeeze(data_FA)*mask_brain
    data_ax=np.squeeze(data_FA)*mask_WM
    data_roi_masked=np.ma.masked_where((mask_roi*data_ax) == 0, mask_roi)
    data_skelet_masked=np.ma.masked_where((mask_skelet*data_ax) == 0, mask_skelet)
    lim_max=0.9

    cmap = plt.cm.plasma  # define the colormap
    # extract all colors from the .jet map
    cmap_num=np.amax(mask_roi.flatten()).astype(int)
    cmaplist = [cmap(i) for i in range(1,cmap_num*5+1,5)]
    cmap_roi = matplotlib.colors.LinearSegmentedColormap.from_list('roi_cmap', cmaplist, cmap_num)
    bounds = np.linspace(1, cmap_num, cmap_num)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap_num)

    cmap_skelet='winter'

    box=np.empty((3,4))
    for i in range(0,3):
        im=np.sum(np.abs(data_ax),axis=i)
        box[i,2]=np.argmax(np.sum(im,axis=0)>0)
        box[i,3]=len(np.sum(im,axis=0))-np.argmax((np.sum(im,axis=0)>0)[::-1])
        box[i,0]=np.argmax(np.sum(im,axis=1)>0)
        box[i,1]=len(np.sum(im,axis=1))-np.argmax((np.sum(im,axis=1)>0)[::-1])
    box=box.astype(int)

    for sl_id in range(0,len(sl_sag)):
        for j in range(0,3):
            for i in range(0, 2):

                ax = ax_all[sl_id,j*2+i]

                if j==2:
                    b=0
                    if i:
                        cb1=ax.imshow(np.rot90(data_ax[sl_sag[sl_id],box[b,0]:box[b,1],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                        ax.imshow(np.rot90(data_roi_masked[sl_sag[sl_id],box[b,0]:box[b,1],box[b,2]:box[b,3]]), cmap=cmap_roi, norm=norm,interpolation='None',alpha=0.8)#,clim=(0.0, lim_max))
                        ax.imshow(np.rot90(data_skelet_masked[sl_sag[sl_id],box[b,0]:box[b,1],box[b,2]:box[b,3]]), cmap=cmap_skelet,interpolation='None',clim=(0.0, 1.0),alpha=0.8)
                        pos = ax.get_position()
                        f.text(pos.x0,pos.y0,'Slice '+str((sl_sag[sl_id])+1),fontsize=NIsize,color='white',ha='center')
                    else:
                        cb1=ax.imshow(np.rot90(data_ax_FA[sl_sag[sl_id],box[b,0]:box[b,1],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                if j==0:
                    if i:
                        cb1=ax.imshow(np.rot90(data_ax[box[2,0]:box[2,1],box[2,2]:box[2,3],sl_ax[sl_id]]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                        ax.imshow(np.rot90(data_roi_masked[box[2,0]:box[2,1],box[2,2]:box[2,3],sl_ax[sl_id]]), cmap=cmap_roi, norm=norm,interpolation='None',alpha=0.8)#,clim=(0.0, lim_max))
                        ax.imshow(np.rot90(data_skelet_masked[box[2,0]:box[2,1],box[2,2]:box[2,3],sl_ax[sl_id]]), cmap=cmap_skelet,interpolation='None',clim=(0.0, 1.0),alpha=0.8)
                        pos = ax.get_position()
                        f.text(pos.x0,pos.y0,'Slice '+str((sl_ax[sl_id])+1),fontsize=NIsize,color='white',ha='center')
                    else:
                        cb1=ax.imshow(np.rot90(data_ax_FA[box[2,0]:box[2,1],box[2,2]:box[2,3],sl_ax[sl_id]]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                if j==1:
                    b=1
                    if i:
                        cb1=ax.imshow(np.rot90(data_ax[box[b,0]:box[b,1],sl_cor[sl_id],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, lim_max))
                        ax.imshow(np.rot90(data_roi_masked[box[b,0]:box[b,1],sl_cor[sl_id],box[b,2]:box[b,3]]), cmap=cmap_roi, norm=norm,interpolation='None',alpha=0.8)#,clim=(0.0, lim_max))
                        ax.imshow(np.rot90(data_skelet_masked[box[b,0]:box[b,1],sl_cor[sl_id],box[b,2]:box[b,3]]), cmap=cmap_skelet,interpolation='None',clim=(0.0, 1.0),alpha=0.8)
                        pos = ax.get_position()
                        f.text(pos.x0,pos.y0,'Slice '+str((sl_cor[sl_id])+1),fontsize=NIsize,color='white',ha='center')
                    else:
                        cb1=ax.imshow(np.rot90(data_ax_FA[box[b,0]:box[b,1],sl_cor[sl_id],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, lim_max))

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.patch.set_visible(False)

    qc_FA_labels= os.path.abspath(os.path.join(os.getcwd(),'QC_modelfit_FA_labels.png'))
    plt.savefig(qc_FA_labels, dpi=dpi_fig, facecolor=f.get_facecolor(),bbox_inches = 'tight')

    return os.path.abspath(qc_FA_labels)


def plot_diffmaps(data_map1,data_map2,data_map3,mask_brain,clim,str_maps):
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    figw=6.92
    figh=3.2
    NIsize=10
    dpi_fig=500
    font = {'weight' : 'normal',
            'size'   : NIsize}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['xtick.major.pad']='0'
    matplotlib.rcParams['ytick.major.pad']='0'

    sl_step=5
    sl_ax_step=5
    sl_sag= np.arange(0,data_map1.shape[0])[49:95:sl_step]
    sl_cor=np.arange(0,data_map1.shape[1])[49:95:sl_step]
    sl_ax=np.arange(0,data_map1.shape[2])[29:75:sl_ax_step]

    #settings figure images
    f, ax_all = plt.subplots(len(sl_sag),9,figsize=(figw*1.5, figh*3))
    f.subplots_adjust(left=0.02, right=0.96, wspace=0,hspace=0.0,top=1.2,bottom=0.0)
    f.patch.set_facecolor('k')

    data_ax1=np.squeeze(data_map1)*mask_brain
    data_ax2=np.squeeze(data_map2)*mask_brain
    data_ax3=np.squeeze(data_map3)*mask_brain

    box=np.empty((3,4))
    for i in range(0,3):
        im=np.sum(np.abs(mask_brain),axis=i)
        box[i,2]=np.argmax(np.sum(im,axis=0)>0)
        box[i,3]=len(np.sum(im,axis=0))-np.argmax((np.sum(im,axis=0)>0)[::-1])
        box[i,0]=np.argmax(np.sum(im,axis=1)>0)
        box[i,1]=len(np.sum(im,axis=1))-np.argmax((np.sum(im,axis=1)>0)[::-1])
    box=box.astype(int)

    num_i=3
    for sl_id in range(0,len(sl_sag)):
        for j in range(0,3):
            for i in range(0, num_i):

                ax = ax_all[sl_id,j*num_i+i]

                if j==2:
                    b=0
                    if i:
                        if i==1:
                            cb1=ax.imshow(np.rot90(data_ax2[sl_sag[sl_id],box[b,0]:box[b,1],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                            pos = ax.get_position()
                            f.text(pos.x0,pos.y0,'Slice '+str((sl_sag[sl_id])+1),fontsize=NIsize,color='white',ha='center')
                        else:
                            cb1=ax.imshow(np.rot90(data_ax3[sl_sag[sl_id],box[b,0]:box[b,1],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                    else:
                        cb1=ax.imshow(np.rot90(data_ax1[sl_sag[sl_id],box[b,0]:box[b,1],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                if j==0:
                    if i:
                        if i==1:
                            cb1=ax.imshow(np.rot90(data_ax2[box[2,0]:box[2,1],box[2,2]:box[2,3],sl_ax[sl_id]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                            pos = ax.get_position()
                            f.text(pos.x0,pos.y0,'Slice '+str((sl_ax[sl_id])+1),fontsize=NIsize,color='white',ha='center')
                        else:
                            cb1=ax.imshow(np.rot90(data_ax3[box[2,0]:box[2,1],box[2,2]:box[2,3],sl_ax[sl_id]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                    else:
                        cb1=ax.imshow(np.rot90(data_ax1[box[2,0]:box[2,1],box[2,2]:box[2,3],sl_ax[sl_id]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                if j==1:
                    b=1
                    if i:
                        if i==1:
                            cb1=ax.imshow(np.rot90(data_ax2[box[b,0]:box[b,1],sl_cor[sl_id],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                            pos = ax.get_position()
                            f.text(pos.x0,pos.y0,'Slice '+str((sl_cor[sl_id])+1),fontsize=NIsize,color='white',ha='center')
                            if sl_id==0: f.text(pos.x0+(pos.x1-pos.x0)/2.0,pos.y1,str_maps[i],fontsize=NIsize,color='white',ha='center')
                        else:
                            cb1=ax.imshow(np.rot90(data_ax3[box[b,0]:box[b,1],sl_cor[sl_id],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                            pos = ax.get_position()
                            if sl_id==0: f.text(pos.x0+(pos.x1-pos.x0)/2.0,pos.y1,str_maps[i],fontsize=NIsize,color='white',ha='center')
                    else:
                        cb1=ax.imshow(np.rot90(data_ax1[box[b,0]:box[b,1],sl_cor[sl_id],box[b,2]:box[b,3]]), cmap='gray',interpolation='None',clim=(0.0, clim[i]))
                        pos = ax.get_position()
                        if sl_id==0: f.text(pos.x0+(pos.x1-pos.x0)/2.0,pos.y1,str_maps[i],fontsize=NIsize,color='white',ha='center')

                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.patch.set_visible(False)

    qc_diffmaps=os.path.abspath(os.path.join(os.getcwd(),'QC_modelfit_{}_{}_{}.png'.format(str_maps[0],str_maps[1],str_maps[2])))
    plt.savefig(qc_diffmaps, dpi=dpi_fig, facecolor=f.get_facecolor(),bbox_inches = 'tight')

    return os.path.abspath(qc_diffmaps)


def get_motion_error(filename_rms,filename_eddy_bvals_shells,filename_motion_params,filename_motion_outl,filename_motion_outl_rep,mask,filename_scan,filename_eddy_corrected_meas):
    import numpy as np
    import nibabel as nib
    import os
    from dipy.io import read_bvals_bvecs

    motion_stats = {}

    rms_dsi_scan=np.genfromtxt(filename_rms)
    bvals_meas = np.genfromtxt(filename_eddy_bvals_shells, dtype=float)
    bvals_meas_round=(np.round(bvals_meas/10.0)*10.0).astype(int)
    #required for (3) motion outlier
    bvals_unique, counts = np.unique(bvals_meas_round[bvals_meas_round>0], return_counts=True)
    numSlices= mask.shape[2]
    num_b0=np.sum(bvals_meas == 0)
    num_dwis=np.sum(bvals_meas > 0)


    #(1) absolute/relative motion (mm)
    abs_mo_bvals=rms_dsi_scan[:,0]
    rel_mo_bvals=rms_dsi_scan[:,1]
    abs_mo_mean=np.mean(abs_mo_bvals)
    rel_mo_mean=np.mean(rel_mo_bvals)
    abs_mo_std=np.std(abs_mo_bvals)
    rel_mo_std=np.std(rel_mo_bvals)

    prec=2
    motion_stats.update({"abs_mean": sci_format(float(abs_mo_mean),prec,False)})
    motion_stats.update({"abs_std": sci_format(float(abs_mo_std),prec,False)})
    motion_stats.update({"rel_mean": sci_format(float(rel_mo_mean),prec,False)})
    motion_stats.update({"rel_std": sci_format(float(rel_mo_std),prec,False)})
    for i in range(0,bvals_meas.size):
        motion_stats.update({"abs_scan"+str(i)+"_b"+str(bvals_meas_round[i]): sci_format(float(abs_mo_bvals[i]),prec,False)})
        motion_stats.update({"rel_scan"+str(i)+"_b"+str(bvals_meas_round[i]): sci_format(float(rel_mo_bvals[i]),prec,False)})

    #return [abs_mo_mean,rel_mo_mean,abs_mo_std,rel_mo_std],abs_mo_bvals,rel_mo_bvals

    #(2) x,y,z translations, rotations (mm)
    mo_params=np.genfromtxt(filename_motion_params)
    #AVERAGE x,y,z translation (mm) (indices 0,1,2), rotations (mm) (indices 3,4,5), Eddy currents (EC) linear terms (indices 6,7,8)
    rot_trans_mean = np.mean(mo_params[:,0:9], axis=0)
    rot_trans_std = np.std(mo_params[:,0:9], axis=0)
    #rotations in degrees
    rot_trans_mean[3:6]=np.rad2deg(rot_trans_mean[3:6])
    rot_trans_std[3:6]=np.rad2deg(rot_trans_std[3:6])
    mo_params[:,3:6]=np.rad2deg(mo_params[:,3:6])

    motion_stats.update({"trans_x_mean": sci_format(float(rot_trans_mean[0]),prec,False)})
    motion_stats.update({"trans_x_std": sci_format(float(rot_trans_std[0]),prec,False)})
    motion_stats.update({"trans_y_mean": sci_format(float(rot_trans_mean[1]),prec,False)})
    motion_stats.update({"trans_y_std": sci_format(float(rot_trans_std[1]),prec,False)})
    motion_stats.update({"trans_z_mean": sci_format(float(rot_trans_mean[2]),prec,False)})
    motion_stats.update({"trans_z_std": sci_format(float(rot_trans_std[2]),prec,False)})
    motion_stats.update({"rot_x_mean": sci_format(float(rot_trans_mean[3]),prec,False)})
    motion_stats.update({"rot_x_std": sci_format(float(rot_trans_std[3]),prec,False)})
    motion_stats.update({"rot_y_mean": sci_format(float(rot_trans_mean[4]),prec,False)})
    motion_stats.update({"rot_y_std": sci_format(float(rot_trans_std[4]),prec,False)})
    motion_stats.update({"rot_z_mean": sci_format(float(rot_trans_mean[5]),prec,False)})
    motion_stats.update({"rot_z_std": sci_format(float(rot_trans_std[5]),prec,False)})

    for i in range(0,bvals_meas.size):
        motion_stats.update({"trans_x_scan"+str(i)+"_b"+str(bvals_meas_round[i]): sci_format(float(mo_params[i,0]),prec,False)})
        motion_stats.update({"trans_y_scan"+str(i)+"_b"+str(bvals_meas_round[i]): sci_format(float(mo_params[i,1]),prec,False)})
        motion_stats.update({"trans_z_scan"+str(i)+"_b"+str(bvals_meas_round[i]): sci_format(float(mo_params[i,2]),prec,False)})
        motion_stats.update({"rot_x_scan"+str(i)+"_b"+str(bvals_meas_round[i]): sci_format(float(mo_params[i,3]),prec,False)})
        motion_stats.update({"rot_y_scan"+str(i)+"_b"+str(bvals_meas_round[i]): sci_format(float(mo_params[i,4]),prec,False)})
        motion_stats.update({"rot_z_scan"+str(i)+"_b"+str(bvals_meas_round[i]): sci_format(float(mo_params[i,5]),prec,False)})

    motion_rot_trans=mo_params[:,0:9]
    #return rot_trans_mean,rot_trans_std,mo_params[:,0:9]


    #(3) motion outliers
    outl_map = np.genfromtxt(filename_motion_outl,dtype=None, delimiter=" ", skip_header=1)
    outl_rep = np.genfromtxt(filename_motion_outl_rep,dtype=np.string_, delimiter=" ")

    rep_idx=[1,4,10,17]
    slice_vol_stds=outl_rep[:,rep_idx].astype(float)

    #std matrix volumes vs sclices
    stds_mat=np.zeros((numSlices,bvals_meas.size))
    for i in range(0,slice_vol_stds.shape[0]):
        stds_mat[int(slice_vol_stds[i,0]),int(slice_vol_stds[i,1])]=slice_vol_stds[i,2]

    tot_ol = 100*np.count_nonzero(outl_map)/(num_dwis*numSlices)
    b_ol=np.zeros(bvals_unique.size)
    mo_params_outl = np.zeros(bvals_unique.size+1)
    mo_std = np.zeros(bvals_unique.size+1)
    std_bvals=np.zeros(bvals_unique.size)

    motion_stats.update({"outl_total_perc": sci_format(float(tot_ol),prec,False)})
    motion_stats.update({"outl_total_mean_stdoff": sci_format(float(np.mean(slice_vol_stds[:,2])),prec,False)})

    for i in range(0, bvals_unique.size):
        b_ol[i] = 100*np.count_nonzero(outl_map[bvals_meas == bvals_unique[i], :])/(counts[i]*numSlices)
        std_mat_uniques=stds_mat[:,bvals_meas == bvals_unique[i]]
        if np.sum(np.abs(std_mat_uniques)>0):
            std_bvals[i] = np.mean(std_mat_uniques[np.abs(std_mat_uniques)>0])
        motion_stats.update({"outl_b"+str(bvals_unique[i].astype(int))+"_perc": sci_format(float(b_ol[i]),prec,False)})
        motion_stats.update({"outl_b"+str(bvals_unique[i].astype(int))+"_mean_stdoff": sci_format(float(std_bvals[i]),prec,False)})

    mo_params_outl[0]=tot_ol
    mo_params_outl[1:]=b_ol[:]
    mo_std[0]=np.mean(slice_vol_stds[:,2])
    mo_std[1:]=std_bvals[:]
    #return mo_params_outl,mo_std,stds_mat

    #plot motion error
    data_scan=nib.load(filename_scan).get_data()
    data_eddy_corrected_meas=nib.load(filename_eddy_corrected_meas).get_data()
    #total, trans, rot
    qc_abs_rel_trans_rot = plot_motion_abs_rel_trans_rot(bvals_meas,motion_rot_trans,abs_mo_bvals,rel_mo_bvals)
    #comparison absolute motion before and after eddy correction
    qc_abs_before_after = plot_motion_abs_before_after(data_eddy_corrected_meas,data_scan)
    #motion outlier
    qc_outlier_perc_mat, qc_outlier = plot_motion_outlier(bvals_meas,stds_mat,data_eddy_corrected_meas,data_scan)

    return motion_stats, os.path.abspath(qc_abs_rel_trans_rot),os.path.abspath(qc_abs_before_after),os.path.abspath(qc_outlier_perc_mat),os.path.abspath(qc_outlier)

def initial_b0s(in_data,ref_bvecs):
    """
    reorder b0-sorted b-files and data to initial b0 ordering
    """
    import os
    import nibabel as nib
    import numpy as np
    from dipy.io import read_bvals_bvecs

    #initial b0 idx
    bvals, bvecs = read_bvals_bvecs(None, ref_bvecs)
    norm_ref_bvecs=(bvecs[:,0]**2+bvecs[:,1]**2+bvecs[:,2]**2)**0.5
    b0_indices = np.where(norm_ref_bvecs == 0)
    numb0=np.sum(norm_ref_bvecs == 0)
    b0_selection = np.where(norm_ref_bvecs == 0)[0]
    dwi_selection = np.where(norm_ref_bvecs > 0)[0]

    #load b0-sorted data
    im = nib.load(in_data)
    data = im.get_data()
    #insert b0 data at initial index
    data_initb0=np.empty_like(data)
    data_initb0[...,b0_selection]=data[...,:numb0]
    data_initb0[...,dwi_selection]=data[...,numb0:]

    #save data with initial b0 ordering
    out_data = os.path.abspath(os.path.join(os.getcwd(),'eddy_corrected_meas.nii.gz'))
    nib.save(nib.nifti1.Nifti1Image(data_initb0.astype(np.float32),im.affine),out_data)

    return os.path.abspath(out_data)


def stats_labels(diff_files,label_files,qc_files,tbss_files,filename_csrecon_idx,filename_eddy_bvals_shells,skeletMNI_PSMD,skeletMNI_PSMD_WMH):

    import numpy as np
    from dsi_pipeline.dtiutil import load_data, sci_format, NMSE, get_csrecon_error, get_motion_error, plot_labels_over_FA, plot_diffmaps
    from dsi_pipeline.dtiutil import plot_motion_abs_rel_trans_rot,plot_motion_abs_before_after,plot_motion_outlier
    from dsi_pipeline.dtiutil import NpEncoder
    from dsi_pipeline.configoptions import labelsROIsMNI,labelsROIsMNI_RL
    import os
    import json
    import copy

    #diff parameter maps
    #dti_files=diff_files

    #WM JHU atlas in MNI (load first because name from configoptions is used for other input below)
    labelsJHUROIsMNI_data, labelsJHUROIsMNI_aff = load_data(labelsROIsMNI)
    labelsJHUROIsMNI_RL_data, labelsJHUROIsMNI_RL_aff = load_data(labelsROIsMNI_RL)

    #labels in diff native space
    labelsAseg     = [x for x in label_files if 'T12b0_labels' in x][0]
    labelsWMH     = [x for x in label_files if 'WMH2b0_labels' in x][0]
    labelsROIsMNI  = [x for x in label_files if 'MNI2b0_labels_ROI' in x][0]
    labelsROIsHistMNI  = [x for x in label_files if 'MNI2b0_labels_ROI_HIST' in x][0]
    labelsTractsMNI= [x for x in label_files if 'MNI2b0_labels_tract' in x][0]
    skeletMNI      = [x for x in label_files if 'MNI2b0_skelet' in x][0]

    labelsROIsMNI_fsl  = [x for x in label_files if 'MNI2FA_labels_ROI' in x][0]
    labelsROIsHistMNI_fsl  = [x for x in label_files if 'MNI2FA_labels_ROI_Hist' in x][0]
    labelsTractsMNI_fsl= [x for x in label_files if 'MNI2FA_labels_tract' in x][0]
    skeletMNI_fsl      = [x for x in label_files if 'MNI2FA_skelet' in x][0]

    brain_mask=       [x for x in label_files if 'T12b0_mask' in x][0]
    wm_mask_aseg=  [x for x in label_files if 'wm_mask_aseg' in x][0]

    #JHU WM tracts atlas
    WMTRACTS_name='MRI_JHU_TRACT_'
    WMTRACTS_labels=['ATR_L','ATR_R','CST_L','CST_R','CCG_L','CCG_R','CH_L',
                     'CH_R','FMA','FMI','IFOF_L','IFOF_R','ILF_L','ILF_R',
                     'SLF_L','SLF_R','UF_L','UF_R','TSLF_L','TSLF_R']
    WMTRACTS_labels=[WMTRACTS_name + lab for lab in WMTRACTS_labels]
    WMTRACTS_idx=np.arange(1,len(WMTRACTS_labels)+1)

    #JHU WM ROI atlas
    WMROI_name='MRI_JHU_ROI_'
    WMROI_name_MNI='MRI_JHU_ROI_MNI_'
    WMROI_labels=['MCP','PCT','GCC','BCC','SCC','FOR','CST_R','CST_L','ML_R',
                  'ML_L','ICP_R','ICP_L','SCP_R','SCP_L','CP_R','CP_L',
                  'ALIC_R','ALIC_L','PLIC_R','PLIC_L','RIC_R','RIC_L','ACR_R',
                  'ACR_L','SCR_R','SCR_L','PCR_R','PCR_L','PTR_R','PTR_L',
                  'SS_R','SS_L','EC_R','EC_L','CCG_R','CCG_L','CH_R','CH_L',
                  'FST_R','FST_L','SLF_R','SLF_L','SFOF_R','SFOF_L','UF_R',
                  'UF_L','TAP_R','TAP_L']
    WMROI_labels_MNI=[WMROI_name_MNI + lab for lab in WMROI_labels]
    WMROI_labels=[WMROI_name + lab for lab in WMROI_labels]
    WMROI_idx=np.arange(1,len(WMROI_labels)+1)

    WMROI_labels_MNI_RL=['CST','ML',
                  'ICP','SCP','CP',
                  'ALIC','PLIC','RIC',
                  'ACR','SCR','PCR','PTR',
                  'SS','EC','CCG','CH',
                  'FST','SLF','SFOF',
                  'UF','TAP']
    WMROI_labels_MNI_RL=[WMROI_name_MNI + lab for lab in WMROI_labels_MNI_RL]
    offset_RL=6
    WMROI_idx_RL=np.arange(1+offset_RL,offset_RL+len(WMROI_labels_MNI_RL)+1)

    #JHU WM ROI histological atlas
    WMROIHIST_name='MRI_JUEL_ROI_HIST_'
    WMROIHIST_labels=['AR_R','AR_L','CC','CING_R','CING_L','CST_R','CST_L','FOR',
                      'IFOF_R','IFOF_L','LGN_R','LGN_L','MB','MGN_R','MGN_L',
                      'OR_R','OR_L','SLF_R','SLF_L','SFOF_R','SFOF_L','UF_R','UF_L']
    WMROIHIST_labels=[WMROIHIST_name + lab for lab in WMROIHIST_labels]
    WMROIHIST_idx=np.arange(93,93+len(WMROIHIST_labels))
    gm_flag_idx=[103,104,105,106,107]

    #T1 FreeSurfer segmentation
    FS_name='MRI_ASEG_'
    FS_labels=['WM_L','GM_CORTEX_L','LAT_VENTR_L','INFLAT_VENTR_L',
               'WM_CEREBELLUM_L','C_CEREBELLUM_L',
               'THALAMUS_L','CAUDATE_L','PUTAMEN_L',
               'PALLIDUM_L','VENTR3','VENTR4','BRAINSTEM','HIPPOCAMPUS_L',
               'AMYGDALA_L','CSF','ACCUMBENS_L','VENTRAL_DC_L','VESSEL_L','CHOROID_PLEXUS_L',
               'WM_R','GM_CORTEX_R','LATERAL_VENTR_R','INFLAT_VENTR_R',
               'WM_CEREBELLUM_R','C_CEREBELLUM_R',
               'THALAMUS_R','CAUDATE_R','PUTAMEN_R',
               'PALLIDUM_R','HIPPOCAMPUS_R',
               'AMYGDALA_R','ACCUMBENS_R','VENTRAL_DC_R','VESSEL_R','CHOROID_PLEXUS_R','OPTIC_CHIASM']
               #'WMH','CC_P','CC_MP','CC_C','CC_MA','CC_A']
    FS_labels=[FS_name + lab for lab in FS_labels]
    FS_idx=[2,3,4,5, 7,8, 10,11,12, 13,14,15,16,17, 18,24,26,28,30,31,
            41,42,43,44, 46,47, 49,50,51, 52,53, 54,58,60,62,63, 85 ]#77,251,252,253,254,255]
    gm_aseg_flag_idx=[3,4,5, 8, 10,11,12, 13,14,15,16,17, 18,24,26,28,30,31,
                      42,43,44, 47, 49,50,51, 52,53, 54,58,60,62,63, 85 ]

    #compute stats (mean and std per ROI/tract)
    labelsAseg_data, labelsAseg_aff = load_data(labelsAseg)
    labelsWMH_data, labelsWMH_aff = load_data(labelsWMH)

    labelsROIsMNI_data, labelsROIsMNI_aff = load_data(labelsROIsMNI)
    labelsROIsHistMNI_data, labelsROIsHistMNI_aff = load_data(labelsROIsHistMNI)
    labelsTractsMNI_data, labelsTractsMNI_aff = load_data(labelsTractsMNI)
    skeletMNI_data, skeletMNI_aff = load_data(skeletMNI)

    labelsROIsMNI_fsl_data, labelsROIsMNI_fsl_aff = load_data(labelsROIsMNI_fsl)
    labelsROIsHistMNI_fsl_data, labelsROIsHistMNI_fsl_aff = load_data(labelsROIsHistMNI_fsl)
    labelsTractsMNI_fsl_data, labelsTractsMNI_fsl_aff = load_data(labelsTractsMNI_fsl)
    skeletMNI_fsl_data, skeletMNI_fsl_aff = load_data(skeletMNI_fsl)

    #PSMD skeleton mask
    skeletMNI_PSMD_data, skeletMNI_PSMD_aff = load_data(skeletMNI_PSMD)
    skeletMNI_PSMD_data=np.squeeze(skeletMNI_PSMD_data)>0
    skeletMNI_PSMD_WMH_data, skeletMNI_PSMD_WMH_aff = load_data(skeletMNI_PSMD_WMH)
    skeletMNI_PSMD_WMH_data=np.squeeze(skeletMNI_PSMD_WMH_data)>0

    #brain mask for GM regions
    brain_mask_data,brain_mask_aff = load_data(brain_mask)
    brain_mask_data=brain_mask_data>0

    #MD contamination removed and FA thresholded ASEG WM mask
    wm_mask_aseg_data, wm_mask_aseg_aff = load_data(wm_mask_aseg)
    wm_mask_aseg_data=wm_mask_aseg_data>0

    #apply wm_mask_aseg to pure wm_labels atlases
    labelsROIsMNI_data=labelsROIsMNI_data*wm_mask_aseg_data
    labelsROIsMNI_fsl_data=labelsROIsMNI_fsl_data*wm_mask_aseg_data
    labelsTractsMNI_data=labelsTractsMNI_data*wm_mask_aseg_data
    labelsTractsMNI_fsl_data=labelsTractsMNI_fsl_data*wm_mask_aseg_data

    #apply wm_mask_aseg to wm_labels atlases, but not GM_labels
    wm_mask_aseg_data_ROIsHistMNI=wm_mask_aseg_data>0
    for i in range(0,len(gm_flag_idx)):
        wm_mask_aseg_data_ROIsHistMNI[labelsROIsHistMNI_data==gm_flag_idx[i]]=1
    labelsROIsHistMNI_data=labelsROIsHistMNI_data*wm_mask_aseg_data_ROIsHistMNI

    wm_mask_aseg_data_ROIsHistMNI_fsl=wm_mask_aseg_data>0
    for i in range(0,len(gm_flag_idx)):
        wm_mask_aseg_data_ROIsHistMNI_fsl[labelsROIsHistMNI_fsl_data==gm_flag_idx[i]]=1
    labelsROIsHistMNI_fsl_data=labelsROIsHistMNI_fsl_data*wm_mask_aseg_data_ROIsHistMNI_fsl

    wm_mask_aseg_data_labelsAseg=wm_mask_aseg_data>0
    for i in range(0,len(gm_aseg_flag_idx)):
        wm_mask_aseg_data_labelsAseg[labelsAseg_data==gm_aseg_flag_idx[i]]=1
    labelsAseg_data=labelsAseg_data*wm_mask_aseg_data_labelsAseg

    #dict for labels from both registration methods
    labelsROIsMNI_dict={'ants':labelsROIsMNI_data, 'fsl':labelsROIsMNI_fsl_data}
    labelsROIsHistMNI_dict={'ants':labelsROIsHistMNI_data, 'fsl':labelsROIsHistMNI_fsl_data}
    labelsTractsMNI_dict={'ants':labelsTractsMNI_data, 'fsl':labelsTractsMNI_fsl_data}
    skeletMNI_data_dict={'ants':skeletMNI_data, 'fsl':skeletMNI_fsl_data}
    skel_thr=[0.2]#,0.3]
    str_thr=['_skelet']#,'_skelet03']
    dict_reg=['ants','fsl']
    str_reg=['','_fsl']
    prec=15 #3

    stats = {}
    stats_fsl = {}

    diff_files_labels = ['DT_FA','DT_MD','DT_AD','DT_RD',
                         'DT_FSL_FA','DT_FSL_MD','DT_FSL_AD','DT_FSL_RD',
                         'FW_FA','FW_MD','FW_AD','FW_RD','FW_VF',
                         'DK_FA','DK_MD','DK_AD','DK_RD','DK_MK','DK_AK','DK_RK',
                         'NODDI_W_IC','NODDI_W_ISO','NODDI_W_EC','NODDI_ODI','NODDI_NDI','NODDI_VF_EC']
    diff_sci=[0,1,1,1, 0,1,1,1, 0,1,1,1,0, 0,1,1,1,0,0,0, 0,0,0,0,0,0]
    diff_files_idx=0

    for diff_file in diff_files:

        diff_data, diff_aff = load_data(diff_file)

        #apply brain mask
        diff_data=np.squeeze(diff_data)*brain_mask_data

        stats.update({diff_files_labels[diff_files_idx]: {}})
        stats_fsl.update({diff_files_labels[diff_files_idx]: {}})

        #WMTRACTS
        lab_idx=0
        for idx in WMTRACTS_labels:
            stats[diff_files_labels[diff_files_idx]].update({idx : {}})
            stats_fsl[diff_files_labels[diff_files_idx]].update({idx : {}})

            #iterate over registration methods:
            idx_r_str=0
            for idx_r in dict_reg:

                #TRACTS
                diff_mean = np.mean(diff_data[labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]])
                diff_median = np.median(diff_data[labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]])
                diff_std  = np.std(diff_data[labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]])

                diff_nan_WMH  = np.sum((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (labelsWMH_data>0))
                num_NAWM=np.sum((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (labelsWMH_data==0))
                num_WM=np.sum(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx])
                if diff_nan_WMH:
                    diff_mean_WMH = np.mean(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (labelsWMH_data>0)])
                    diff_median_WMH = np.median(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (labelsWMH_data>0)])
                    diff_std_WMH  = np.std(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (labelsWMH_data>0)])
                    if diff_nan_WMH == np.sum(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]):
                        if idx_r_str:
                            stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                            )
                        else:
                            stats[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                    else:
                        diff_mean_NAWM = np.mean(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (labelsWMH_data==0)])
                        diff_median_NAWM = np.median(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (labelsWMH_data==0)])
                        diff_std_NAWM  = np.std(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (labelsWMH_data==0)])
                        if idx_r_str:
                            stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_WM"+str_reg[idx_r_str]:num_WM.astype(int),
                                     "mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int),
                                     "mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                        else:
                            stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_WM"+str_reg[idx_r_str]:num_WM.astype(int),
                                 "mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int),
                                 "mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                            )
                else:
                    if idx_r_str:
                        stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int)}
                         )
                    else:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int)}
                         )

                #itertate over different thresholds for skeleton
                idx_t_str=0
                for idx_t in skel_thr:
                    #TRACTS within skeleton thr02
                    diff_nan_skelet  = np.sum((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t))
                    if diff_nan_skelet:
                        diff_mean_skelet = np.mean(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                        diff_median_skelet = np.median(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                        diff_std_skelet  = np.std(diff_data[(labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])

                        diff_nan_skelet_WMH  = np.sum(((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0))
                        num_skelet_NAWM= np.sum(((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0))
                        num_skelet_WM=np.sum((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t))
                        if diff_nan_skelet_WMH:
                            diff_mean_skelet_WMH = np.mean(diff_data[((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            diff_median_skelet_WMH = np.median(diff_data[((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            diff_std_skelet_WMH  = np.std(diff_data[((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            if diff_nan_skelet_WMH == np.sum((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)):
                                if idx_r_str:
                                    stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                                else:
                                    stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                            else:
                                diff_mean_skelet_NAWM = np.mean(diff_data[((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                diff_median_skelet_NAWM = np.median(diff_data[((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                diff_std_skelet_NAWM  = np.std(diff_data[((labelsTractsMNI_dict[idx_r] == WMTRACTS_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                if idx_r_str:
                                    stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                             "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                             "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                                else:
                                    stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                             "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                             "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                        else:
                            if idx_r_str:
                                stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int)}
                                )
                            else:
                                stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int)}
                                )

                    idx_t_str +=1
                idx_r_str +=1

            lab_idx +=1

        #WMROI
        lab_idx=0
        for idx in WMROI_labels:
            stats[diff_files_labels[diff_files_idx]].update({idx : {}})
            stats_fsl[diff_files_labels[diff_files_idx]].update({idx : {}})

            #iterate over registration methods:
            idx_r_str=0
            for idx_r in dict_reg:
                #TRACTS
                diff_mean = np.mean(diff_data[labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]])
                diff_median = np.median(diff_data[labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]])
                diff_std  = np.std(diff_data[labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]])
                diff_nan_WMH  = np.sum((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (labelsWMH_data>0))
                num_NAWM=np.sum((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (labelsWMH_data==0))
                num_WM=np.sum(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx])
                if diff_nan_WMH:
                    diff_mean_WMH = np.mean(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (labelsWMH_data>0)])
                    diff_median_WMH = np.median(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (labelsWMH_data>0)])
                    diff_std_WMH  = np.std(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (labelsWMH_data>0)])
                    if diff_nan_WMH == np.sum(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]):
                        if idx_r_str:
                            stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                        else:
                            stats[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                    else:
                        diff_mean_NAWM = np.mean(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (labelsWMH_data==0)])
                        diff_median_NAWM = np.median(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (labelsWMH_data==0)])
                        diff_std_NAWM  = np.std(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (labelsWMH_data==0)])
                        if idx_r_str:
                            stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_WM"+str_reg[idx_r_str]:num_WM.astype(int),
                                     "mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int),
                                     "mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                        else:
                            stats[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_WM"+str_reg[idx_r_str]:num_WM.astype(int),
                                     "mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int),
                                     "mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                else:
                    if idx_r_str:
                        stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int)}
                         )
                    else:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int)}
                         )

                #itertate over different thresholds for skeleton
                idx_t_str=0
                for idx_t in skel_thr:
                    #TRACTS within skeleton thr02
                    diff_nan_skelet  = np.sum((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t))
                    if diff_nan_skelet:
                        diff_mean_skelet = np.mean(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                        diff_median_skelet = np.median(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                        diff_std_skelet  = np.std(diff_data[(labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])

                        diff_nan_skelet_WMH  = np.sum(((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0))
                        num_skelet_NAWM= np.sum(((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0))
                        num_skelet_WM=np.sum((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t))
                        if diff_nan_skelet_WMH:
                            diff_mean_skelet_WMH = np.mean(diff_data[((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            diff_median_skelet_WMH = np.median(diff_data[((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            diff_std_skelet_WMH  = np.std(diff_data[((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            if diff_nan_skelet_WMH == np.sum((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)):
                                if idx_r_str:
                                    stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                                else:
                                    stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                            else:
                                diff_mean_skelet_NAWM = np.mean(diff_data[((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                diff_median_skelet_NAWM = np.median(diff_data[((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                diff_std_skelet_NAWM  = np.std(diff_data[((labelsROIsMNI_dict[idx_r] == WMROI_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                if idx_r_str:
                                    stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                             "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                             "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                                else:
                                    stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                             "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                             "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                        else:
                            if idx_r_str:
                                stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int)}
                                 )
                            else:
                                stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int)}
                                 )

                    idx_t_str +=1
                idx_r_str +=1

            lab_idx +=1

        #WMROI_HIST
        lab_idx=0
        for idx in WMROIHIST_labels:
            stats[diff_files_labels[diff_files_idx]].update({idx : {}})
            stats_fsl[diff_files_labels[diff_files_idx]].update({idx : {}})

            #iterate over registration methods:
            idx_r_str=0
            for idx_r in dict_reg:
                #TRACTS
                diff_mean = np.mean(diff_data[labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]])
                diff_median = np.median(diff_data[labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]])
                diff_std  = np.std(diff_data[labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]])
                diff_nan_WMH  = np.sum((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (labelsWMH_data>0))
                num_NAWM=np.sum((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (labelsWMH_data==0))
                num_WM=np.sum(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx])
                if diff_nan_WMH:
                    diff_mean_WMH = np.mean(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (labelsWMH_data>0)])
                    diff_median_WMH = np.median(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (labelsWMH_data>0)])
                    diff_std_WMH  = np.std(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (labelsWMH_data>0)])
                    if diff_nan_WMH == np.sum(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]):
                        if idx_r_str:
                            stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                        else:
                            stats[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                    else:
                        diff_mean_NAWM = np.mean(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (labelsWMH_data==0)])
                        diff_median_NAWM = np.median(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (labelsWMH_data==0)])
                        diff_std_NAWM  = np.std(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (labelsWMH_data==0)])
                        if idx_r_str:
                            stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_WM"+str_reg[idx_r_str]:num_WM.astype(int),
                                     "mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int),
                                     "mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                        else:
                            stats[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_WM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_WM"+str_reg[idx_r_str]:num_WM.astype(int),
                                     "mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int),
                                     "mean_WMH"+str_reg[idx_r_str]:sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_reg[idx_r_str]:sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_reg[idx_r_str]:sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_reg[idx_r_str]:diff_nan_WMH.astype(int)}
                             )
                else:
                    if idx_r_str:
                        stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int)}
                         )
                    else:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                    {"mean_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_reg[idx_r_str]:sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_reg[idx_r_str]:num_NAWM.astype(int)}
                         )

                #itertate over different thresholds for skeleton
                idx_t_str=0
                for idx_t in skel_thr:
                    #TRACTS within skeleton thr02
                    diff_nan_skelet  = np.sum((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t))
                    if diff_nan_skelet:
                        diff_mean_skelet = np.mean(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                        diff_median_skelet = np.median(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                        diff_std_skelet  = np.std(diff_data[(labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)])

                        diff_nan_skelet_WMH  =  np.sum(((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0))
                        num_skelet_NAWM=        np.sum(((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0))
                        num_skelet_WM=          np.sum(( labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t))
                        if diff_nan_skelet_WMH:
                            diff_mean_skelet_WMH = np.mean(diff_data[((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            diff_median_skelet_WMH = np.median(diff_data[((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            diff_std_skelet_WMH  = np.std(diff_data[((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                            if diff_nan_skelet_WMH == np.sum((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)):
                                if idx_r_str:
                                    stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                                else:
                                    stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                            else:
                                diff_mean_skelet_NAWM = np.mean(diff_data[((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                diff_median_skelet_NAWM = np.median(diff_data[((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                diff_std_skelet_NAWM  = np.std(diff_data[((labelsROIsHistMNI_dict[idx_r] == WMROIHIST_idx[lab_idx]) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                                if idx_r_str:
                                    stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                             "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                             "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                                else:
                                    stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                             "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                             "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                     )
                        else:
                            if idx_r_str:
                                stats_fsl[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int)}
                                 )
                            else:
                                stats[diff_files_labels[diff_files_idx]][idx].update(
                                            {"mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int)}
                                 )

                    idx_t_str +=1
                idx_r_str +=1

            lab_idx +=1

        #FS_labels
        lab_idx=0
        for idx in FS_labels:
            stats[diff_files_labels[diff_files_idx]].update({idx : {}})

            diff_mean = np.mean(diff_data[labelsAseg_data == FS_idx[lab_idx]])
            diff_median = np.mean(diff_data[labelsAseg_data == FS_idx[lab_idx]])
            diff_std  = np.std( diff_data[labelsAseg_data == FS_idx[lab_idx]])

            diff_nan_WMH  = np.sum( (labelsAseg_data == FS_idx[lab_idx]) & (labelsWMH_data>0))
            num_NAWM=np.sum((labelsAseg_data == FS_idx[lab_idx]) & (labelsWMH_data==0))
            num_WM=np.sum(labelsAseg_data == FS_idx[lab_idx])
            if diff_nan_WMH:
                diff_mean_WMH = np.mean(diff_data[(labelsAseg_data == FS_idx[lab_idx]) & (labelsWMH_data>0)])
                diff_median_WMH = np.median(diff_data[(labelsAseg_data == FS_idx[lab_idx]) & (labelsWMH_data>0)])
                diff_std_WMH  = np.std( diff_data[(labelsAseg_data == FS_idx[lab_idx]) & (labelsWMH_data>0)])
                if diff_nan_WMH == np.sum(labelsAseg_data == FS_idx[lab_idx]):
                    stats[diff_files_labels[diff_files_idx]][idx].update(
                            {"mean_WMH":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH":diff_nan_WMH.astype(int)}
                            )
                else:
                    diff_mean_NAWM = np.mean(diff_data[(labelsAseg_data == FS_idx[lab_idx]) & (labelsWMH_data==0)])
                    diff_median_NAWM = np.median(diff_data[(labelsAseg_data == FS_idx[lab_idx]) & (labelsWMH_data==0)])
                    diff_std_NAWM  = np.std( diff_data[(labelsAseg_data == FS_idx[lab_idx]) & (labelsWMH_data==0)])
                    stats[diff_files_labels[diff_files_idx]][idx].update(
                            {"mean_WM":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_WM":num_WM.astype(int),
                             "mean_NAWM":sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM":sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM":sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM":num_NAWM.astype(int),
                             "mean_WMH":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH":diff_nan_WMH.astype(int)}
                            )
            else:
                stats[diff_files_labels[diff_files_idx]][idx].update(
                            {"mean_NAWM":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_NAWM":num_NAWM.astype(int)}
                 )

            lab_idx +=1


        #Global WM stats with and without skeletonization
        #iterate over registration methods:
        idx_glob="MRI_WM_GLOBAL"
        stats[diff_files_labels[diff_files_idx]].update({idx_glob : {}})
        stats_fsl[diff_files_labels[diff_files_idx]].update({idx_glob : {}})

        #global / whole brain WM
        diff_mean = np.mean(diff_data[wm_mask_aseg_data>0])
        diff_median = np.mean(diff_data[wm_mask_aseg_data>0])
        diff_std  = np.std(diff_data[wm_mask_aseg_data>0])

        diff_nan_WMH  = np.sum( (wm_mask_aseg_data>0) & (labelsWMH_data>0))
        num_NAWM=np.sum((wm_mask_aseg_data>0) & (labelsWMH_data==0))
        num_WM=np.sum((wm_mask_aseg_data>0))
        if diff_nan_WMH:
            diff_mean_WMH = np.mean(diff_data[(wm_mask_aseg_data>0) & (labelsWMH_data>0)])
            diff_median_WMH = np.median(diff_data[(wm_mask_aseg_data>0) & (labelsWMH_data>0)])
            diff_std_WMH  = np.std( diff_data[(wm_mask_aseg_data>0) & (labelsWMH_data>0)])
            if diff_nan_WMH == np.sum(wm_mask_aseg_data>0):
                stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                        {"mean_WMH":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH":diff_nan_WMH.astype(int)}
                        )
            else:
                diff_mean_NAWM = np.mean(diff_data[(wm_mask_aseg_data>0) & (labelsWMH_data==0)])
                diff_median_NAWM = np.median(diff_data[(wm_mask_aseg_data>0) & (labelsWMH_data==0)])
                diff_std_NAWM  = np.std( diff_data[(wm_mask_aseg_data>0) & (labelsWMH_data==0)])
                stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                        {"mean_WM":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_WM":num_WM.astype(int),
                         "mean_NAWM":sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM":sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM":sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM":num_NAWM.astype(int),
                         "mean_WMH":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH":diff_nan_WMH.astype(int)}
                        )
        else:
            stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                        {"mean_NAWM":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),"num_NAWM":num_NAWM.astype(int)}
             )

        #skeletonized global / whole brain WM
        idx_r_str=0
        for idx_r in dict_reg:
            #itertate over different thresholds for skeleton
            idx_t_str=0
            for idx_t in skel_thr:
                diff_mean_skelet = np.mean(diff_data[(wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                diff_median_skelet = np.median(diff_data[(wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                diff_std_skelet  = np.std(diff_data[(wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)])
                diff_p5_skelet  = np.nanpercentile(diff_data[(wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)],5)
                diff_p95_skelet  = np.nanpercentile(diff_data[(wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)],95)
                diff_pw90s_skelet  = diff_p95_skelet-diff_p5_skelet

                diff_nan_skelet_WMH  = np.sum(((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0))
                num_skelet_NAWM= np.sum(((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0))
                num_skelet_WM=np.sum((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t))
                if diff_nan_skelet_WMH:
                    diff_mean_skelet_WMH = np.mean(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                    diff_median_skelet_WMH = np.median(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                    diff_std_skelet_WMH  = np.std(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)])
                    diff_p5_skelet_WMH  = np.nanpercentile(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)],5)
                    diff_p95_skelet_WMH  = np.nanpercentile(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data>0)],95)
                    diff_pw90s_skelet_WMH  = diff_p95_skelet_WMH-diff_p5_skelet_WMH
                    if diff_nan_skelet_WMH == np.sum((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)):
                        if diff_pw90s_skelet_WMH:
                            if idx_r_str:
                                stats_fsl[diff_files_labels[diff_files_idx]][idx_glob].update(
                                         {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),
                                          "p5_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet_WMH),prec,diff_sci[diff_files_idx]),"pw90_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                )
                            else:
                                stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                                         {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),
                                          "p5_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet_WMH),prec,diff_sci[diff_files_idx]),"pw90_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                )
                        else:
                            if idx_r_str:
                                stats_fsl[diff_files_labels[diff_files_idx]][idx_glob].update(
                                         {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                )
                            else:
                                stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                                         {"mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                )
                    else:
                        diff_mean_skelet_NAWM = np.mean(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                        diff_median_skelet_NAWM = np.median(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                        diff_std_skelet_NAWM  = np.std(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)])
                        diff_p5_skelet_NAWM  = np.nanpercentile(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)],5)
                        diff_p95_skelet_NAWM  = np.nanpercentile(diff_data[((wm_mask_aseg_data>0) & (skeletMNI_data_dict[idx_r]>=idx_t)) & (labelsWMH_data==0)],95)
                        diff_pw90s_skelet_NAWM  = diff_p95_skelet_NAWM-diff_p5_skelet_NAWM
                        if diff_pw90s_skelet_WMH:
                            if idx_r_str:
                                stats_fsl[diff_files_labels[diff_files_idx]][idx_glob].update(
                                         {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),
                                          "p5_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet),prec,diff_sci[diff_files_idx]),"p95_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet),prec,diff_sci[diff_files_idx]),"pw90_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                          "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),
                                          "p5_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet_NAWM),prec,diff_sci[diff_files_idx]),"pw90_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                          "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),
                                          "p5_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet_WMH),prec,diff_sci[diff_files_idx]),"pw90_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                        )
                            else:
                                stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                                         {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),
                                          "p5_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet),prec,diff_sci[diff_files_idx]),"p95_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet),prec,diff_sci[diff_files_idx]),"pw90_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                          "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),
                                          "p5_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet_NAWM),prec,diff_sci[diff_files_idx]),"pw90_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                          "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),
                                          "p5_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet_WMH),prec,diff_sci[diff_files_idx]),"pw90_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                        )
                        else:
                            if idx_r_str:
                                stats_fsl[diff_files_labels[diff_files_idx]][idx_glob].update(
                                         {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),
                                          "p5_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet),prec,diff_sci[diff_files_idx]),"p95_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet),prec,diff_sci[diff_files_idx]),"pw90_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                          "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),
                                          "p5_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet_NAWM),prec,diff_sci[diff_files_idx]),"pw90_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                          "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                        )
                            else:
                                stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                                         {"mean_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),
                                          "p5_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet),prec,diff_sci[diff_files_idx]),"p95_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet),prec,diff_sci[diff_files_idx]),"pw90_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet),prec,diff_sci[diff_files_idx]),"num_WM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_WM.astype(int),
                                          "mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_NAWM),prec,diff_sci[diff_files_idx]),
                                          "p5_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet_NAWM),prec,diff_sci[diff_files_idx]),"pw90_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int),
                                          "mean_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet_WMH),prec,diff_sci[diff_files_idx]),"median_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet_WMH),prec,diff_sci[diff_files_idx]),"std_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet_WMH),prec,diff_sci[diff_files_idx]),"num_WMH"+str_thr[idx_t_str]+str_reg[idx_r_str]:diff_nan_skelet_WMH.astype(int)}
                                        )
                else:
                    if idx_r_str:
                        stats_fsl[diff_files_labels[diff_files_idx]][idx_glob].update(
                                    {"mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),
                                    "p5_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet),prec,diff_sci[diff_files_idx]),"p95_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet),prec,diff_sci[diff_files_idx]),"pw90_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int)}
                         )
                    else:
                        stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                                    {"mean_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_mean_skelet),prec,diff_sci[diff_files_idx]),"median_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_median_skelet),prec,diff_sci[diff_files_idx]),"std_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_std_skelet),prec,diff_sci[diff_files_idx]),
                                    "p5_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p5_skelet),prec,diff_sci[diff_files_idx]),"p95_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_p95_skelet),prec,diff_sci[diff_files_idx]),"pw90_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:sci_format(float(diff_pw90s_skelet),prec,diff_sci[diff_files_idx]),"num_NAWM"+str_thr[idx_t_str]+str_reg[idx_r_str]:num_skelet_NAWM.astype(int)}
                         )
                idx_t_str +=1
            idx_r_str +=1

        diff_files_idx +=1

    #-------PSMD TBSS in MNI: skeletonized global / whole brain WM--------
    diff_files_idx=0
    for diff_file in tbss_files:
        diff_data, diff_aff = load_data(diff_file)
        diff_data=np.squeeze(diff_data)
        diff_data_mask= diff_data>0

        idx_glob="MRI_WM_GLOBAL_MNI"
        stats[diff_files_labels[diff_files_idx]].update({idx_glob : {}})

        diff_mean = np.mean(diff_data[skeletMNI_PSMD_data>0])
        diff_median = np.mean(diff_data[skeletMNI_PSMD_data>0])
        diff_std  = np.std(diff_data[skeletMNI_PSMD_data>0])
        diff_p5  = np.nanpercentile(diff_data[(skeletMNI_PSMD_data>0)],5)
        diff_p95  = np.nanpercentile(diff_data[(skeletMNI_PSMD_data>0)],95)
        diff_pw90s  = diff_p95-diff_p5

        diff_nan_WMH  = np.sum( (skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data>0))
        num_NAWM=np.sum((skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data==0))
        num_WM=np.sum((skeletMNI_PSMD_data>0))
        if diff_nan_WMH:
            diff_mean_WMH = np.mean(diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data>0)])
            diff_median_WMH = np.median(diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data>0)])
            diff_std_WMH  = np.std( diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data>0)])
            diff_p5_WMH  = np.nanpercentile( diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data>0)],5)
            diff_p95_WMH  = np.nanpercentile( diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data>0)],95)
            diff_pw90s_WMH  = diff_p95_WMH-diff_p5_WMH
            if diff_nan_WMH == np.sum(skeletMNI_PSMD_data>0):
                if diff_pw90s_WMH:
                    stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                            {"mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),
                            "p5_WMH_skelet":sci_format(float(diff_p5_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH_skelet":sci_format(float(diff_p95_WMH),prec,diff_sci[diff_files_idx]),"pw90_WMH_skelet":sci_format(float(diff_pw90s_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                            )
                else:
                    stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                            {"mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                            )
            else:
                diff_mean_NAWM = np.mean(diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data==0)])
                diff_median_NAWM = np.median(diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data==0)])
                diff_std_NAWM  = np.std( diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data==0)])
                diff_p5_NAWM  = np.nanpercentile( diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data==0)],5)
                diff_p95_NAWM  = np.nanpercentile( diff_data[(skeletMNI_PSMD_data>0) & (skeletMNI_PSMD_WMH_data==0)],95)
                diff_pw90s_NAWM  = diff_p95_NAWM-diff_p5_NAWM
                if diff_pw90s_WMH: #diff_pw90s_NAWM:
                    stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                            {"mean_WM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                             "p5_WM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_WM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),"pw90_WM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_WM_skelet":num_WM.astype(int),
                             "mean_NAWM_skelet":sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),
                             "p5_NAWM_skelet":sci_format(float(diff_p5_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95_NAWM),prec,diff_sci[diff_files_idx]),"pw90_NAWM_skelet":sci_format(float(diff_pw90s_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int),
                             "mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),
                             "p5_WMH_skelet":sci_format(float(diff_p5_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH_skelet":sci_format(float(diff_p95_WMH),prec,diff_sci[diff_files_idx]),"pw90_WMH_skelet":sci_format(float(diff_pw90s_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                            )
                else:
                    stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                            {"mean_WM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                             "p5_WM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_WM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),"pw90_WM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_WM_skelet":num_WM.astype(int),
                             "mean_NAWM_skelet":sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),
                             "p5_NAWM_skelet":sci_format(float(diff_p5_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95_NAWM),prec,diff_sci[diff_files_idx]),"pw90_NAWM_skelet":sci_format(float(diff_pw90s_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int),
                             "mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                            )
        else:
            stats[diff_files_labels[diff_files_idx]][idx_glob].update(
                        {"mean_NAWM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                         "p5_NAWM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),"pw90_NAWM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int)}
             )

        #WMROI (R and L regions separate) in MNI on skeleton
        lab_idx=0
        for idx in WMROI_labels_MNI:
            stats[diff_files_labels[diff_files_idx]].update({idx : {}})

            diff_mean = np.mean(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0)])
            diff_median = np.mean(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0)])
            diff_std  = np.std(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0)])
            diff_p5  = np.nanpercentile(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0)],5)
            diff_p95  = np.nanpercentile(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0)],95)
            diff_pw90s  = diff_p95-diff_p5

            diff_nan_WMH  = np.sum((labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0))
            num_NAWM=np.sum((labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0))
            num_WM=np.sum((labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0))

            if diff_nan_WMH:
                diff_mean_WMH = np.mean(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)])
                diff_median_WMH = np.median(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)])
                diff_std_WMH  = np.std( diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)])
                diff_p5_WMH  = np.nanpercentile( diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)],5)
                diff_p95_WMH  = np.nanpercentile( diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)],95)
                diff_pw90s_WMH  = diff_p95_WMH-diff_p5_WMH
                if diff_nan_WMH == np.sum((labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0)):
                    if diff_pw90s_WMH:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),
                                #"p5_WMH_skelet":sci_format(float(diff_p5_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH_skelet":sci_format(float(diff_p95_WMH),prec,diff_sci[diff_files_idx]),
                                "pw90_WMH_skelet":sci_format(float(diff_pw90s_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                                )
                    else:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                                )
                else:
                    diff_mean_NAWM = np.mean(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)])
                    diff_median_NAWM = np.median(diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)])
                    diff_std_NAWM  = np.std( diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)])
                    diff_p5_NAWM  = np.nanpercentile( diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)],5)
                    diff_p95_NAWM  = np.nanpercentile( diff_data[(labelsJHUROIsMNI_data == WMROI_idx[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)],95)
                    diff_pw90s_NAWM  = diff_p95_NAWM-diff_p5_NAWM
                    if diff_pw90s_WMH:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                                 #"p5_WM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_WM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),
                                 "pw90_WM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_WM_skelet":num_WM.astype(int),
                                 "mean_NAWM_skelet":sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),
                                 #"p5_NAWM_skelet":sci_format(float(diff_p5_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95_NAWM),prec,diff_sci[diff_files_idx]),
                                 "pw90_NAWM_skelet":sci_format(float(diff_pw90s_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int),
                                 "mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),
                                 #"p5_WMH_skelet":sci_format(float(diff_p5_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH_skelet":sci_format(float(diff_p95_WMH),prec,diff_sci[diff_files_idx]),
                                 "pw90_WMH_skelet":sci_format(float(diff_pw90s_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                                )
                    else:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                                 #"p5_WM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_WM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),
                                 "pw90_WM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_WM_skelet":num_WM.astype(int),
                                 "mean_NAWM_skelet":sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),
                                 #"p5_NAWM_skelet":sci_format(float(diff_p5_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95_NAWM),prec,diff_sci[diff_files_idx]),
                                 "pw90_NAWM_skelet":sci_format(float(diff_pw90s_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int),
                                 "mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                                )
            else:
                stats[diff_files_labels[diff_files_idx]][idx].update(
                            {"mean_NAWM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                             #"p5_NAWM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),
                             "pw90_NAWM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int)}
                 )

            lab_idx +=1

        #WMROI (R and L regions combined) in MNI on skeleton
        lab_idx=0
        for idx in WMROI_labels_MNI_RL:
            stats[diff_files_labels[diff_files_idx]].update({idx : {}})

            diff_mean = np.mean(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) ])
            diff_median = np.mean(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) ])
            diff_std  = np.std(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) ])
            diff_p5  = np.nanpercentile(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) ],5)
            diff_p95  = np.nanpercentile(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) ],95)
            diff_pw90s  = diff_p95-diff_p5

            diff_nan_WMH  = np.sum((labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0))
            num_NAWM=np.sum((labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0))
            num_WM=np.sum((labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) )

            if diff_nan_WMH:
                diff_mean_WMH = np.mean(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)])
                diff_median_WMH = np.median(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)])
                diff_std_WMH  = np.std( diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)])
                diff_p5_WMH  = np.nanpercentile( diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)],5)
                diff_p95_WMH  = np.nanpercentile( diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data>0)],95)
                diff_pw90s_WMH  = diff_p95_WMH-diff_p5_WMH
                if diff_nan_WMH == np.sum((labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) ):
                    if diff_pw90s_WMH:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),
                                #"p5_WMH_skelet":sci_format(float(diff_p5_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH_skelet":sci_format(float(diff_p95_WMH),prec,diff_sci[diff_files_idx]),
                                "pw90_WMH_skelet":sci_format(float(diff_pw90s_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                                )
                    else:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                                )
                else:
                    diff_mean_NAWM = np.mean(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)])
                    diff_median_NAWM = np.median(diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)])
                    diff_std_NAWM  = np.std( diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)])
                    diff_p5_NAWM  = np.nanpercentile( diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)],5)
                    diff_p95_NAWM  = np.nanpercentile( diff_data[(labelsJHUROIsMNI_RL_data == WMROI_idx_RL[lab_idx]) & (diff_data_mask>0) & (skeletMNI_PSMD_WMH_data==0)],95)
                    diff_pw90s_NAWM  = diff_p95_NAWM-diff_p5_NAWM
                    if diff_pw90s_WMH:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                                 #"p5_WM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_WM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),
                                 "pw90_WM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_WM_skelet":num_WM.astype(int),
                                 "mean_NAWM_skelet":sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),
                                 #"p5_NAWM_skelet":sci_format(float(diff_p5_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95_NAWM),prec,diff_sci[diff_files_idx]),
                                 "pw90_NAWM_skelet":sci_format(float(diff_pw90s_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int),
                                 "mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),
                                 #"p5_WMH_skelet":sci_format(float(diff_p5_WMH),prec,diff_sci[diff_files_idx]),"p95_WMH_skelet":sci_format(float(diff_p95_WMH),prec,diff_sci[diff_files_idx]),
                                 "pw90_WMH_skelet":sci_format(float(diff_pw90s_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                                )
                    else:
                        stats[diff_files_labels[diff_files_idx]][idx].update(
                                {"mean_WM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_WM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_WM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                                 #"p5_WM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_WM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),
                                 "pw90_WM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_WM_skelet":num_WM.astype(int),
                                 "mean_NAWM_skelet":sci_format(float(diff_mean_NAWM),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median_NAWM),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std_NAWM),prec,diff_sci[diff_files_idx]),
                                 #"p5_NAWM_skelet":sci_format(float(diff_p5_NAWM),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95_NAWM),prec,diff_sci[diff_files_idx]),
                                 "pw90_NAWM_skelet":sci_format(float(diff_pw90s_NAWM),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int),
                                 "mean_WMH_skelet":sci_format(float(diff_mean_WMH),prec,diff_sci[diff_files_idx]),"median_WMH_skelet":sci_format(float(diff_median_WMH),prec,diff_sci[diff_files_idx]),"std_WMH_skelet":sci_format(float(diff_std_WMH),prec,diff_sci[diff_files_idx]),"num_WMH_skelet":diff_nan_WMH.astype(int)}
                                )
            else:
                stats[diff_files_labels[diff_files_idx]][idx].update(
                            {"mean_NAWM_skelet":sci_format(float(diff_mean),prec,diff_sci[diff_files_idx]),"median_NAWM_skelet":sci_format(float(diff_median),prec,diff_sci[diff_files_idx]),"std_NAWM_skelet":sci_format(float(diff_std),prec,diff_sci[diff_files_idx]),
                             #"p5_NAWM_skelet":sci_format(float(diff_p5),prec,diff_sci[diff_files_idx]),"p95_NAWM_skelet":sci_format(float(diff_p95),prec,diff_sci[diff_files_idx]),
                             "pw90_NAWM_skelet":sci_format(float(diff_pw90s),prec,diff_sci[diff_files_idx]),"num_NAWM_skelet":num_NAWM.astype(int)}
                 )

            lab_idx +=1

        diff_files_idx +=1



    #QC parameters to json
    #MOTION
    filename_rms  = [x for x in qc_files if 'eddy_corrected.eddy_movement_rms' in x][0]
    filename_motion_params=[x for x in qc_files if 'eddy_corrected.eddy_parameters' in x][0]
    filename_motion_outl=[x for x in qc_files if 'eddy_corrected.eddy_outlier_map' in x][0]
    filename_motion_outl_rep=[x for x in qc_files if 'eddy_corrected.eddy_outlier_report' in x][0]
    filename_scan=[x for x in qc_files if 'DiffusionDSI.nii.gz' in x][0]
    filename_eddy_corrected_meas=[x for x in qc_files if 'eddy_corrected_meas.nii.gz' in x][0]
    qc_motion_stats,qc_abs_rel_trans_rot,qc_abs_before_after,qc_outlier_perc_mat,qc_outlier=get_motion_error(filename_rms,filename_eddy_bvals_shells,filename_motion_params,filename_motion_outl,filename_motion_outl_rep,wm_mask_aseg_data,filename_scan,filename_eddy_corrected_meas)
    stats.update({"QC_MOTION" : qc_motion_stats})

    #CSRECON
    filename_meas_eddy  = [x for x in qc_files if 'eddy_corrected.nii.gz' in x][0]
    filename_csrecon= [x for x in qc_files if 'dsi.nii.gz' in x][0]
    filename_bval  = [x for x in qc_files if 'dsi.bvals' in x][0]
    qc_nmse_stats,qc_nmse=get_csrecon_error(filename_bval,filename_meas_eddy,filename_csrecon,filename_csrecon_idx,copy.deepcopy(wm_mask_aseg_data))
    stats.update({"QC_NMSE" : qc_nmse_stats})

    #Parameter maps
    skeletMNI_data, skeletMNI_aff = load_data(skeletMNI)
    skeletMNI_data=skeletMNI_data>=skel_thr[0]

    #load DT maps
    data_FA, aff_FA = load_data(diff_files[0])
    data_MD, aff_MD = load_data(diff_files[1])
    data_RD, aff_RD = load_data(diff_files[3])
    clim=np.array([0.9,2e-9,2e-9])
    #plot labels over FA
    qc_FA_labels= plot_labels_over_FA(data_FA,brain_mask_data,wm_mask_aseg_data,skeletMNI_data,labelsROIsMNI_data)
    #plot tensor maps
    qc_diffmaps_tensor=plot_diffmaps(data_FA,data_MD,data_RD,brain_mask_data,clim,np.array(diff_files_labels)[[0,1,3]])

    #plot FW maps
    data_FA,aff_FA= load_data(diff_files[8])
    data_MD,aff_MD= load_data(diff_files[9])
    data_FWVF,aff_FWVF= load_data(diff_files[12])
    clim=[0.9,2e-3,0.7]
    qc_diffmaps_fw=plot_diffmaps(data_FA,data_MD,data_FWVF,brain_mask_data,clim,np.array(diff_files_labels)[[8,9,12]])


    #plot kurtosis maps
    data_FA,aff_FA= load_data(diff_files[13])
    data_MD,aff_MD= load_data(diff_files[14])
    data_MK,aff_MK= load_data(diff_files[17])
    clim=[0.9,2e-9,1.5]
    qc_diffmaps_kurtosis=plot_diffmaps(data_FA,data_MD,data_MK,brain_mask_data,clim,np.array(diff_files_labels)[[13,14,17]])

    #plot noddi maps
    data_WIC,aff_WIC= load_data(diff_files[20])
    data_ODI,aff_ODI= load_data(diff_files[23])
    data_WISO,aff_WISO= load_data(diff_files[21])
    clim=[0.9,0.7,0.7]
    qc_diffmaps_noddi=plot_diffmaps(data_WIC,data_ODI,data_WISO,brain_mask_data,clim,np.array(diff_files_labels)[[20,23,21]])

    with open('stats.json','w') as of:
        of.write(json.dumps(stats, cls=NpEncoder))

    with open('stats_fsl.json','w') as of:
        of.write(json.dumps(stats_fsl, cls=NpEncoder))

    return os.path.abspath('stats.json'),os.path.abspath('stats_fsl.json'),os.path.abspath(qc_FA_labels),os.path.abspath(qc_diffmaps_tensor),os.path.abspath(qc_diffmaps_fw),os.path.abspath(qc_diffmaps_kurtosis),os.path.abspath(qc_diffmaps_noddi),os.path.abspath(qc_nmse), os.path.abspath(qc_abs_rel_trans_rot),os.path.abspath(qc_abs_before_after),os.path.abspath(qc_outlier_perc_mat),os.path.abspath(qc_outlier)


def ants_MNI2b0(MNI_FA,ref_FA,MNI_T1,ref_T1):
    """
    run ants transform to bring MNI to diffusion space incorporating T1 anatomical information
    """
    from nipype.interfaces.ants.registration import Registration
    import os

    reg = Registration()

    reg.inputs.fixed_image = [MNI_FA, MNI_T1]
    reg.inputs.moving_image = [ref_FA, ref_T1]

    reg.inputs.collapse_output_transforms = True
    reg.inputs.dimension = 3
    reg.inputs.float = True
    #reg.inputs.initial_moving_transform = [ MNI_FA, ref_FA] #'trans.mat'
    reg.inputs.initial_moving_transform_com = 1
    #reg.inputs.invert_initial_moving_transform = [True,True] #1
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.interpolation = 'Linear'
    reg.inputs.output_transform_prefix = "T12MNI_"
    reg.inputs.output_warped_image = 'T12MNI.nii.gz'

    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[500,250,125,0], [500,250,125,0], [70,50,30,0]]
    reg.inputs.convergence_threshold = [1.e-6]*3
    reg.inputs.convergence_window_size = [10]*3

    reg.inputs.metric = [['MI','MI'],['MI','MI'],['CC','CC']]
    reg.inputs.metric_weight = [[1,1],[1,1],[1,1]] # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [[32,32], [32,32], [4,4]]
    reg.inputs.sampling_strategy = [['Regular','Regular'], ['Regular','Regular'], [None,None]]
    reg.inputs.sampling_percentage = [[0.25,0.25], [0.25,0.25], [None,None]]

    reg.inputs.smoothing_sigmas = [[3.0,2.0,1.0,0.0],[3.0,2.0,1.0,0.0],[3.0,2.0,1.0,0.0]]
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[8,4,2,1],[8,4,2,1],[8,4,2,1]]
    reg.inputs.use_histogram_matching = [True, True, True]

    reg.inputs.write_composite_transform=False
    reg.inputs.winsorize_lower_quantile=0.005
    reg.inputs.winsorize_upper_quantile=0.995

    #whats the purpose of writing the command?
    #reg_command='command.txt'
    #with open(reg_command,'w') as of:
    #    of.write(reg.cmdline)

    res = reg.run()
    outputs = res.outputs.get()
    rev_trans0=outputs['reverse_transforms'][0]
    rev_trans1=outputs['reverse_transforms'][1]
    forw_trans1=outputs['forward_transforms'][1]

    return rev_trans0,rev_trans1,forw_trans1,outputs['warped_image']


def apply_MNI2b0(in_file,ref_file,trans_file_mat,trans_file_invWarp,interp,out_fname):
    """
    apply ants transform to bring MNI to diffusion space incorporating T1 anatomical information
    """
    from nipype.interfaces.ants.resampling import ApplyTransforms
    import os

    at = ApplyTransforms()
    at.inputs.input_image = in_file
    at.inputs.reference_image = ref_file
    at.inputs.transforms = [trans_file_mat, trans_file_invWarp]
    at.inputs.interpolation = interp
    at.inputs.output_image = out_fname + '.nii.gz'

    at.inputs.dimension = 3
    at.inputs.default_value = 0
    at.inputs.float = True
    at.inputs.invert_transform_flags = [True, False]

    #reg_command='command.txt'
    #with open(reg_command,'w') as of:
    #    of.write(at.cmdline)

    res = at.run()
    outputs = res.outputs.get()

    return outputs['output_image']

def convert_mgz(in_file):
    """
    convert mgz to nii.gz
    """
    from nipype.interfaces.freesurfer.preprocess import MRIConvert
    import os.path as op

    fname, ext = op.splitext(op.basename(in_file))
    if ext == ".gz":
        return in_file
        #fname, ext2 = op.splitext(fname)
        #ext = ext2 + ext
    else:
        mc = MRIConvert()
        mc.inputs.in_file = in_file
        mc.inputs.out_type = 'niigz'
        mc.inputs.out_file = fname + '.nii.gz'
        res = mc.run()
        outputs = res.outputs.get()
        return outputs['out_file']


def normVector(bvecs):
    #normalisation for vector length of 1
    ns=(bvecs[:,0]**2+bvecs[:,1]**2+bvecs[:,2]**2)**0.5
    bvecs[ns>0,0]=(bvecs[ns>0,0]/ns[ns>0])
    bvecs[ns>0,1]=(bvecs[ns>0,1]/ns[ns>0])
    bvecs[ns>0,2]=(bvecs[ns>0,2]/ns[ns>0])
    return bvecs


def scanner2FOV(file_data, file_bvec,flag_x,flag_y,flag_z, prefix):
    import os
    import numpy as np
    import nibabel as nib
    import numpy.linalg as npl
    from dipy.io import read_bvals_bvecs
    from dsi_pipeline.dtiutil import normVector

    #load bvecs
    bvals, bvecs = read_bvals_bvecs(None, file_bvec)

    #load affine and rotation matrix
    affine=nib.load(file_data).affine
    rot = affine[:3, :3]

    bvecs[bvecs==0]=0.0
    #flip sign of x- and y-component of gradient vectors
    if flag_x:
        bvecs[bvecs[:,0]!=0,0]=-bvecs[bvecs[:,0]!=0,0]
    if flag_y:
        bvecs[bvecs[:,1]!=0,1]=-bvecs[bvecs[:,1]!=0,1]
    if flag_z:
        bvecs[bvecs[:,2]!=0,2]=-bvecs[bvecs[:,2]!=0,2]

    #apply inverse affine matrix
    bvecs = normVector(npl.inv(rot).dot(normVector(bvecs).T).T)

    #save bvecs
    bvecs_file='bvecs'
    if len(prefix)>0:
        bvecs_file = os.path.abspath(os.path.join(os.getcwd(),prefix+'.bvecs'))
    else:
        bvecs_file = os.path.abspath(os.path.join(os.getcwd(), 'bvecs') )

    np.savetxt(bvecs_file, np.transpose(bvecs),fmt=str("%.14g"),delimiter=' ')
    return bvecs_file



def swap_bvecs(bvecs, flag_x, flag_y, flag_z):
    import os
    import numpy as np
    from dipy.io import read_bvals_bvecs
    #load bvecs
    file_bvec = bvecs
    bvals, bvecs = read_bvals_bvecs(None, file_bvec)

    bvecs[bvecs==0]=0.0

    #flip sign of y-component of gradient vectors
    if flag_x:
        bvecs[bvecs[:,0]!=0,0]=-bvecs[bvecs[:,0]!=0,0]
    if flag_y:
        bvecs[bvecs[:,1]!=0,1]=-bvecs[bvecs[:,1]!=0,1]
    if flag_z:
        bvecs[bvecs[:,2]!=0,2]=-bvecs[bvecs[:,2]!=0,2]

    #save rotated bvecs
    swaped_bvecs = os.path.abspath(os.path.join(
            os.getcwd(),'bvecs_swaped.bvecs'))

    np.savetxt(swaped_bvecs, np.transpose(bvecs),
               fmt=str("%.14g"), delimiter=' ')

    return swaped_bvecs


def include_eddy_rotated_bvecs(bvals,bvecs, bvecs_eddy, idx_file):
    import os
    import numpy as np
    from dipy.io import read_bvals_bvecs

    file_bval=bvals
    file_bvec=bvecs
    file_bvec_eddy=bvecs_eddy

    bvals, bvecs_eddy = read_bvals_bvecs(None, file_bvec_eddy)
    bvals, bvecs = read_bvals_bvecs(file_bval, file_bvec)

    #indices
    numb0=np.sum(bvals<=10)
    numDwis=bvals.shape[0]-numb0
    numDwisUnique=numDwis/2

    #cs q-space samples
    b_idx=np.loadtxt(idx_file)[1:,...]>0

    b_idx_b0s=np.ones_like(bvals)>0
    b_idx_b0s[numb0:]=b_idx

    b_idx_b0s_half1=np.zeros((bvals.shape[0]))>0
    b_idx_b0s_half1[:numDwisUnique+numb0]=b_idx_b0s[:numDwisUnique+numb0]
    b_idx_b0s_half2=np.zeros((bvals.shape[0]))>0
    b_idx_b0s_half2[numDwisUnique+numb0:]=b_idx_b0s[numDwisUnique+numb0:]

    #----------add eddy rotation to dsi dirs-----------
    bvecs[b_idx_b0s_half1,...]=bvecs_eddy
    bvecs[b_idx_b0s_half2,...]=-bvecs_eddy[numb0:,...]

    swaped_rotated_bvecs = os.path.abspath(os.path.join(
            os.getcwd(), 'swaped_rotated.bvecs')
        )
    np.savetxt(swaped_rotated_bvecs, np.transpose(bvecs),
               fmt=str("%.14g"), delimiter=' ')

    return swaped_rotated_bvecs


def unique_dwis_dsi(dsidata,bvals,bvecs):

    import os
    import nibabel as nib
    import numpy as np
    from dipy.io import read_bvals_bvecs

    #load data
    file_bval = bvals
    file_bvec = bvecs

    bvals, bvecs = read_bvals_bvecs(file_bval, file_bvec)

    data_img=nib.load(dsidata)
    data=data_img.get_data()
    affine=data_img.affine

    #indices
    numb0=np.sum(bvals<=10)
    numDwis=bvals.shape[0]-numb0
    numDwisUnique=numDwis/2

    reduced_bvals = os.path.abspath(os.path.join(os.getcwd(),'dsi.bvals' ) )
    reduced_bvecs = os.path.abspath(os.path.join(os.getcwd(),'dsi.bvecs' ) )
    reduced_data_file = os.path.abspath(os.path.join(os.getcwd(),'dsi.nii.gz'))

    #save reduced data
    np.savetxt(reduced_bvecs, np.transpose(
            bvecs[:int(numb0) + int(numDwisUnique),:]), fmt=str("%.14g"), delimiter=' ')

    bval_new=np.empty([1,int(numb0) + int(numDwisUnique)])
    bval_new[0,...]=bvals[:int(numb0) + int(numDwisUnique)].astype(np.int)
    np.savetxt(reduced_bvals, bval_new, fmt=str("%i"),delimiter=' ')

    nib.save(nib.nifti1.Nifti1Image(
            data[...,:int(numb0) + int(numDwisUnique)].astype(np.float32),
            affine),reduced_data_file)

    return reduced_bvals,reduced_bvecs, reduced_data_file



def get_b0_indices(in_bvals):
    """
    Get b0 indices from input bvals and return the indices
    """

    import numpy as np

    #find all b0s indices where b values <=50
    bvals = np.loadtxt(in_bvals)
    b0_indices = np.where(bvals <= 50)

    #convert to tuple
    b0_indices = tuple(b0_indices[0])
    b0_indices_int = list(range(len(b0_indices))) #list(range...for py3
    for n,i in enumerate(b0_indices):
        b0_indices_int[n] = int(b0_indices[n])
    b0_indices = tuple(b0_indices_int)

    return b0_indices


def extractb0AP(in_dwi, in_bvals):
    """
    Extract all b0 images from the input volumes
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    from nipype.utils import NUMPY_MMAP
    from dsi_pipeline.dtiutil import get_b0_indices

    fname, ext = op.splitext(op.basename(in_dwi))
    if ext == ".gz":
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext

    b0_dwi_filename = op.abspath("%s_b0%s" %(fname,ext))

    b0_indices = get_b0_indices(in_bvals)


    #load the image
    image = nb.load(in_dwi, mmap=NUMPY_MMAP)
    dwidata = image.get_data()

    #prune b0 volumes
    b0_extdata = np.squeeze(dwidata.take(b0_indices, axis=3))
    b0_hdr = image.header.copy()
    b0_hdr.set_data_shape(b0_extdata.shape)

    #save b0 volumes
    nb.Nifti1Image(b0_extdata, image.affine, b0_hdr).to_filename(b0_dwi_filename)

    return b0_dwi_filename

def create_encoding_params_file(in_bvals):
    """
    create acqparams file using input bvals <= 50
    """

    import numpy as np
    import os
    from dsi_pipeline.dtiutil import get_b0_indices
    from dsi_pipeline.configoptions import ECHO_SPACING_MSEC, ECHO_TRAIN_LENGTH,PA_NUM

    b0_indices = get_b0_indices(in_bvals)

    #predefined constants
    echo_train_duration_sec = ECHO_SPACING_MSEC * ( ECHO_TRAIN_LENGTH - 1 ) * 0.001

    AP_string = "0 -1 0 %.3f" % (echo_train_duration_sec)
    PA_string = "0 1 0 %.3f" % (echo_train_duration_sec)

    acqparams_file="acqparams.txt"

    acqparams = np.repeat([AP_string], len(b0_indices))
    acqparams = np.append(acqparams, np.repeat([PA_string], PA_NUM))
    #nipye will complain if fmt="%..." is used, so convert to str("%...")
    np.savetxt(acqparams_file,acqparams, delimiter=" ",fmt=str("%s"))

    return os.path.abspath(acqparams_file)


def create_index_file(in_bvals):
    """
    create index file for topup input corresponding to b0 volumes in dwi
    """

    import os
    import numpy as np
    from dsi_pipeline.dtiutil import get_b0_indices

    b0_indices = get_b0_indices(in_bvals)

    filename = "index.txt"

    bvals = np.loadtxt(in_bvals)

    series1=np.ones(bvals.shape[0], dtype=int)

    for ib0 in range(1,len(b0_indices)):
        series1[b0_indices[ib0]:]=ib0+1

    series1.tofile(filename, sep=" ")

    return os.path.abspath(filename)


def reorder_dwi(in_dwi, in_bval):
    """
    Writes an image containing the volumes with b-value 0 at the beginning
    followed by non-b0 volumes.
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    from nipype.utils import NUMPY_MMAP

    fname, ext = op.splitext(op.basename(in_dwi))
    if ext == ".gz":
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext
    b0_out_file = op.abspath("%s_b0s%s" % (fname, ext))
    dwi_out_file = op.abspath("%s_dwis%s" % (fname, ext))

    im = nb.load(in_dwi, mmap=NUMPY_MMAP)
    dwidata = im.get_data()
    bvals = np.loadtxt(in_bval)

    b0_selection = np.where(bvals <= 50)
    dwi_selection = np.where(bvals >50)

    b0_extdata = np.squeeze(dwidata.take(b0_selection, axis=3))
    b0_hdr = im.header.copy()
    b0_hdr.set_data_shape(b0_extdata.shape)
    nb.Nifti1Image(b0_extdata, im.affine, b0_hdr).to_filename(b0_out_file)

    dwi_extdata = np.squeeze(dwidata.take(dwi_selection, axis=3))
    dwi_hdr = im.header.copy()
    dwi_hdr.set_data_shape(dwi_extdata.shape)
    nb.Nifti1Image(dwi_extdata, im.affine, dwi_hdr).to_filename(dwi_out_file)

    return b0_out_file,dwi_out_file


def reorder_bvals_bvecs(bvals, bvecs):
    """
    #-----SORT BVAL AND BVECTS FILE-------
    after reordering dwis on the basis of b0s
    """

    import numpy as np
    import os
    from dsi_pipeline.dtiutil import get_b0_indices

    file_bvec_sort = os.path.basename(bvecs) + '_sortb0.bvec'
    file_bval_sort = os.path.basename(bvals) + '_sortb0.bval'
    b0_idx = get_b0_indices(bvals)

    bvals = np.squeeze(np.loadtxt(bvals))
    bvecs = np.squeeze(np.loadtxt(bvecs))


    if bvecs.shape[1] > bvecs.shape[0]:
        bvecs = bvecs.T

    bvec_sort = np.empty([bvecs.shape[1],bvecs.shape[0]],dtype=float)
    bval_sort = np.empty([1,bvals.shape[0]])

    dwi_idx = np.where(bvals > 50)

    dwi_idx=tuple(dwi_idx[0])

    dwi_idx_int=list(range(len(dwi_idx))) #list(range...for py3
    for n,i in enumerate(dwi_idx):
        dwi_idx_int[n]=int(dwi_idx[n])

    dwi_idx=tuple(dwi_idx_int)

    for n,i in enumerate(b0_idx):
        bvec_sort[...,n] = bvecs[i,...]
        bval_sort[0,n] = int(bvals[i])


    for n,i in enumerate(dwi_idx):
        n2=n+len(b0_idx)
        bvec_sort[...,n2] = bvecs[i,...]
        bval_sort[0,n2] = int(bvals[i])

    #nipye will complain if fmt="%..." is used, so convert to str("%...")
    np.savetxt(file_bvec_sort, bvec_sort, fmt=str("%.14g"), delimiter=' ')
    np.savetxt(file_bval_sort, bval_sort, fmt=str("%i"),delimiter=' ')

    return os.path.abspath(file_bval_sort), os.path.abspath(file_bvec_sort)



def eddy_rotate_bvecs(in_bvec, eddy_params):
    """
    Rotates the input bvec file accordingly with a list of parameters sourced
    from ``eddy``, as explained `here
    <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/EDDY/Faq#Will_eddy_rotate_my_bevcs_for_me.3F>`_.
    """
    import os
    import numpy as np
    from math import sin, cos

    name, fext = os.path.splitext(os.path.basename(in_bvec))
    if fext == '.gz':
        name, _ = os.path.splitext(name)
    out_file = os.path.abspath('%s_rotated.bvec' % name)
    bvecs = np.loadtxt(in_bvec).T
    new_bvecs = []

    params = np.loadtxt(eddy_params)

    if len(bvecs) != len(params):
        raise RuntimeError(('Number of b-vectors and rotation '
                            'matrices should match.'))

    for bvec, row in zip(bvecs, params):
        if np.all(bvec == 0.0):
            new_bvecs.append(bvec)
        else:
            ax = row[3]
            ay = row[4]
            az = row[5]

            Rx = np.array([[1.0, 0.0, 0.0],
                           [0.0, cos(ax), -sin(ax)],
                           [0.0, sin(ax), cos(ax)]])
            Ry = np.array([[cos(ay), 0.0, sin(ay)],
                           [0.0, 1.0, 0.0],
                           [-sin(ay), 0.0, cos(ay)]])
            Rz = np.array([[cos(az), -sin(az), 0.0],
                           [sin(az), cos(az), 0.0],
                           [0.0, 0.0, 1.0]])
            R = Rx.dot(Ry).dot(Rz)

            invrot = np.linalg.inv(R)
            newbvec = invrot.dot(bvec)
            new_bvecs.append(newbvec / np.linalg.norm(newbvec))

    np.savetxt(out_file, np.array(new_bvecs).T, fmt=b'%0.15f')
    return out_file



def extract_bvecs_bvals(niftifile):
    import os
    from dcmstack import NiftiWrapper

    bvals=os.path.abspath( os.path.join(os.getcwd(), "bvals") )
    bvecs=os.path.abspath( os.path.join(os.getcwd(), "bvecs") )

    bvx=[]
    bvy=[]
    bvz=[]

    niwrapper=NiftiWrapper.from_filename(niftifile)
    metaext=niwrapper.meta_ext

    for key in metaext.get_keys():
        if key == 'CsaImage.B_value':
            bval_list=metaext.get_values(key)

        if key == 'CsaImage.DiffusionGradientDirection':
            bvec_list=metaext.get_values(key)
            for bvec in bvec_list:
                if bvec is None:
                    bvx.append("0")
                    bvy.append("0")
                    bvz.append("0")
                else:
                    bvx.append(bvec[0])
                    bvy.append(bvec[1])
                    bvz.append(bvec[2])

    ofd=open(bvals,'w')
    ofd.write(" ".join(map(str, bval_list)))
    ofd.write("\n")
    ofd.close()

    ofdv=open(bvecs,'w')
    ofdv.write(" ".join(map(str, bvx)))
    ofdv.write("\n")
    ofdv.write(" ".join(map(str, bvy)))
    ofdv.write("\n")
    ofdv.write(" ".join(map(str, bvz)))
    ofdv.write("\n")
    ofdv.close()

    return bvals, bvecs


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class FDTRotateBvecsInputSpec(CommandLineInputSpec):
	old_bvec_file = File(exists=True, desc='input bvec file.',
                      argstr='%s', position=0, mandatory=True)
	new_bvec_file = File(desc='name of new bvec file',
                      argstr='%s', position=1, mandatory=True)
	edc_log_file = File(exists=True, desc='input edc log file.',
                     argstr='%s', position=2, mandatory=True)

class FDTRotateBvecsOutputSpec(TraitedSpec):
	out_file = File(exists=True, desc='output rotated new bvec file')

class FDTRotateBvecs(CommandLine):
	_cmd='fdt_rotate_bvecs'
	input_spec = FDTRotateBvecsInputSpec
	output_spec = FDTRotateBvecsOutputSpec

	def __init__(self, **inputs):
        	return super(FDTRotateBvecs, self).__init__(**inputs)

	def _run_interface(self, runtime):
		runtime = super(FDTRotateBvecs, self)._run_interface(runtime)
		if runtime.stderr:
			self.raise_exception(runtime)
		return runtime

	def _list_outputs(self):
		outputs = self.output_spec().get()
		outputs['out_file'] = os.path.abspath(
                os.path.basename(self.inputs.new_bvec_file)
                )
		return outputs


class EddyCorrectInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, desc='4D input file',
                   argstr='%s', position=0, mandatory=True)
    out_file = File(desc='4D output file',
                    argstr='%s', position=1,
                    name_source=['in_file'], name_template='%s_edc',
                    output_name='eddy_corrected')
    ref_num = traits.Int(0, argstr='%d', position=2,
                         desc='reference number',
                         mandatory=True, usedefault=True)


class EddyCorrectOutputSpec(TraitedSpec):
    eddy_corrected = File(exists=True,
                          desc='path/name of 4D eddy corrected output file')
    eddy_log = File(exists=True,
                          desc='path/name of eddy log file',
                          output_name='eddy_log', name_source=['in_file'],
                          name_template='%s_edc.ecclog')


class EddyCorrect(FSLCommand):
    _cmd = 'eddy_correct'
    input_spec = EddyCorrectInputSpec
    output_spec = EddyCorrectOutputSpec

    def __init__(self, **inputs):
        warnings.warn(("Deprecated: Please use nipype.interfaces.fsl.epi.Eddy "
                      "instead"), DeprecationWarning)
        return super(EddyCorrect, self).__init__(**inputs)

    def _run_interface(self, runtime):
        runtime = super(EddyCorrect, self)._run_interface(runtime)
        if runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['eddy_corrected'] = os.path.abspath(os.path.basename(
                self.inputs.in_file).replace('.nii.gz','_edc.nii.gz')
            )
        outputs['eddy_log'] = os.path.abspath(os.path.basename(
                self.inputs.in_file).replace('.nii.gz','_edc.ecclog')
            )
        return outputs

class EddyFakeInputSpec(FSLCommandInputSpec):
    in_bvecs = File(exists=True, desc='input bvecs file')
    runtimemin=traits.Int(0, argstr='%d', position=0,mandatory=True)

class EddyFakeOutputSpec(TraitedSpec):
    out_bvecs = File(exists=True, desc='path/name of output file')

class EddyFake(FSLCommand):
    _cmd = 'gd_eddy'
    input_spec = EddyFakeInputSpec
    output_spec = EddyFakeOutputSpec

    def __init__(self, **inputs):
        warnings.warn(("Deprecated: Please use nipype.interfaces.fsl.epi.Eddy "
                      "instead"), DeprecationWarning)
        return super(EddyFake, self).__init__(**inputs)

    def _run_interface(self, runtime):
        runtime = super(EddyFake, self)._run_interface(runtime)
        if runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_bvecs'] = os.path.abspath(os.path.basename(self.inputs.in_bvecs))
        return outputs

class MDTFakeInputSpec(FSLCommandInputSpec):
    in_bvecs = File(exists=True, desc='input bvecs file')
    runtimemin=traits.Int(0, argstr='%d', position=0,mandatory=True)

class MDTFakeOutputSpec(TraitedSpec):
    out_bvecs = File(exists=True, desc='path/name of output file')

class MDTFake(FSLCommand):
    _cmd = 'gd_mdt'
    input_spec = MDTFakeInputSpec
    output_spec = MDTFakeOutputSpec

    def __init__(self, **inputs):
        warnings.warn(("Deprecated: Please use correct mdt "
                      "instead"), DeprecationWarning)
        return super(MDTFake, self).__init__(**inputs)

    def _run_interface(self, runtime):
        runtime = super(MDTFake, self)._run_interface(runtime)
        if runtime.stderr:
            self.raise_exception(runtime)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_bvecs'] = os.path.abspath(os.path.basename(self.inputs.in_bvecs))
        return outputs



class GZipInputSpec(CommandLineInputSpec):
    in_file = File(desc="File", exists=True, mandatory=True, argstr="-c %s > %s")

class GZipOutputSpec(TraitedSpec):
    out_file = File(desc = "Zip file", exists = True)

class GZip(CommandLine):
    input_spec = GZipInputSpec
    output_spec = GZipOutputSpec
    _cmd = 'gzip'


    def _format_arg(self, name, spec, value):
        self.out_file_name = os.path.basename(self.inputs.in_file) + ".gz"
        if name=='in_file':
            return spec.argstr %( self.inputs.in_file, self.out_file_name )
        return super(GZip, self)._format_arg(name, spec, value)

    def _list_outputs(self):
            outputs = self.output_spec().get()
            outputs['out_file'] = os.path.abspath(self.out_file_name)
            return outputs



class BBRegisterInputSpec(FSTraitedSpec):
    subject_id = traits.Str(argstr='--s %s',
                            desc='freesurfer subject id',
                            mandatory=True)
    source_file = File(argstr='--mov %s',
                       desc='source file to be registered',
                       mandatory=True, copyfile=False)
    init = traits.Enum('spm', 'fsl', 'header', argstr='--init-%s',
                       mandatory=True, xor=['init_reg_file'],
                       desc='initialize registration spm, fsl, header')
    init_reg_file = File(exists=True, argstr='--init-reg %s',
                         desc='existing registration file',
                         xor=['init'], mandatory=True)
    contrast_type = traits.Enum('t1', 't2', 'bold', 'dti', argstr='--%s',
                                desc='contrast type of image',
                                mandatory=True)
    intermediate_file = File(exists=True, argstr="--int %s",
                             desc="Intermediate image, e.g. in case of partial FOV")
    reg_frame = traits.Int(argstr="--frame %d", xor=["reg_middle_frame"],
                           desc="0-based frame index for 4D source file")
    reg_middle_frame = traits.Bool(argstr="--mid-frame", xor=["reg_frame"],
                                   desc="Register middle frame of 4D source file")
    out_reg_file = File(argstr='--reg %s',
                        desc='output registration file',
                        genfile=True)
    spm_nifti = traits.Bool(argstr="--spm-nii",
                            desc="force use of nifti rather than analyze with SPM")
    epi_mask = traits.Bool(argstr="--epi-mask",
                           desc="mask out B0 regions in stages 1 and 2")
    dof = traits.Enum(6, 9, 12, argstr='--%d',
                      desc='number of transform degrees of freedom')
    fsldof = traits.Int(argstr='--fsl-dof %d',
                        desc='degrees of freedom for initial registration (FSL)')
    out_fsl_file = traits.Either(traits.Bool, File, argstr="--fslmat %s",
                                 desc="write the transformation matrix in FSL FLIRT format")
    out_lta_file = traits.Either(traits.Bool, File, argstr="--lta %s", min_ver='5.2.0',
                                 desc="write the transformation matrix in LTA format")
    registered_file = traits.Either(traits.Bool, File, argstr='--o %s',
                                    desc='output warped sourcefile either True or filename')
    init_cost_file = traits.Either(traits.Bool, File, argstr='--initcost %s',
                                   desc='output initial registration cost file')
    init_reg_file_out = traits.String(argstr='--init-reg-out %s',
                                   desc='output initial registration mat file')


class BBRegisterInputSpec6(BBRegisterInputSpec):
    init = traits.Enum('coreg', 'rr', 'spm', 'fsl', 'header', 'best', argstr='--init-%s',
                       xor=['init_reg_file'],
                       desc='initialize registration with mri_coreg, spm, fsl, or header')
    init_reg_file = File(exists=True, argstr='--init-reg %s',
                         desc='existing registration file',
                         xor=['init'])


class BBRegisterOutputSpec(TraitedSpec):
    out_reg_file = File(exists=True, desc='Output registration file')
    out_fsl_file = File(exists=True, desc='Output FLIRT-style registration file')
    out_lta_file = File(exists=True, desc='Output LTA-style registration file')
    min_cost_file = File(exists=True, desc='Output registration minimum cost file')
    init_cost_file = File(exists=True, desc='Output initial registration cost file')
    init_reg_file_out = File(exists=True, desc='Output initial registration mat file')
    registered_file = File(exists=True, desc='Registered and resampled source file')


class BBRegister(FSCommand):
    """Use FreeSurfer bbregister to register a volume to the Freesurfer anatomical.

    This program performs within-subject, cross-modal registration using a
    boundary-based cost function. It is required that you have an anatomical
    scan of the subject that has already been recon-all-ed using freesurfer.

    Examples
    --------

    >>> from nipype.interfaces.freesurfer import BBRegister
    >>> bbreg = BBRegister(subject_id='me', source_file='structural.nii', init='header', contrast_type='t2')
    >>> bbreg.cmdline
    'bbregister --t2 --init-header --reg structural_bbreg_me.dat --mov structural.nii --s me'

    """

    _cmd = 'bbregister'
    if LooseVersion('0.0.0') < Info.looseversion() < LooseVersion("6.0.0"):
        input_spec = BBRegisterInputSpec
    else:
        input_spec = BBRegisterInputSpec6
    output_spec = BBRegisterOutputSpec

    def _list_outputs(self):

        outputs = self.output_spec().get()
        _in = self.inputs

        if isdefined(_in.out_reg_file):
            outputs['out_reg_file'] = op.abspath(_in.out_reg_file)
        elif _in.source_file:
            suffix = '_bbreg_%s.dat' % _in.subject_id
            outputs['out_reg_file'] = fname_presuffix(_in.source_file,
                                                      suffix=suffix,
                                                      use_ext=False)

        if isdefined(_in.registered_file):
            if isinstance(_in.registered_file, bool):
                outputs['registered_file'] = fname_presuffix(_in.source_file,
                                                             suffix='_bbreg')
            else:
                outputs['registered_file'] = op.abspath(_in.registered_file)

        if isdefined(_in.out_lta_file):
            if isinstance(_in.out_lta_file, bool):
                suffix = '_bbreg_%s.lta' % _in.subject_id
                out_lta_file = fname_presuffix(_in.source_file,
                                               suffix=suffix,
                                               use_ext=False)
                outputs['out_lta_file'] = out_lta_file
            else:
                outputs['out_lta_file'] = op.abspath(_in.out_lta_file)

        if isdefined(_in.out_fsl_file):
            if isinstance(_in.out_fsl_file, bool):
                suffix = '_bbreg_%s.mat' % _in.subject_id
                out_fsl_file = fname_presuffix(_in.source_file,
                                               suffix=suffix,
                                               use_ext=False)
                outputs['out_fsl_file'] = out_fsl_file
            else:
                outputs['out_fsl_file'] = op.abspath(_in.out_fsl_file)

        if isdefined(_in.init_cost_file):
            if isinstance(_in.out_fsl_file, bool):
                outputs['init_cost_file'] = outputs['out_reg_file'] + '.initcost'
            else:
                outputs['init_cost_file'] = op.abspath(_in.init_cost_file)

        if isdefined(_in.init_reg_file_out):
            outputs['init_reg_file_out'] = op.abspath(_in.init_reg_file_out)

        outputs['min_cost_file'] = outputs['out_reg_file'] + '.mincost'
        return outputs

    def _format_arg(self, name, spec, value):
        if name in ('registered_file', 'out_fsl_file', 'out_lta_file',
                    'init_cost_file') and isinstance(value, bool):
            value = self._list_outputs()[name]
        return super(BBRegister, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):

        if name == 'out_reg_file':
            return self._list_outputs()[name]
        return None
