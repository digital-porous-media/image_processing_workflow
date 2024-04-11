import warnings
import skimage
import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tifffile


# Preparing sample directories in the correct form for the functions below:
def prepare_sample(sample_name):
    path = '/work/09429/cinarturhan/ls6/Sections_All/' + sample_name + '/'
    sections_list = sorted(os.listdir(path=path))
    reconstruction_file_name = ['Reconstructed/', 'Reconstructions/']
    root_dir = '/work/09429/cinarturhan/ls6/Sections_All/'+sample_name+'/'
    slices_file_name = 'Slices/'
    file_type = '*.tif'
    return sample_name,sections_list,root_dir,reconstruction_file_name,slices_file_name,file_type


# Creating dictinaries for samples
def create_dict(my_tuple):
    sample_name              = my_tuple[0]
    sections_list            = my_tuple[1]
    root_dir                 = my_tuple[2]
    reconstruction_file_name = my_tuple[3]
    slices_file_name         = my_tuple[4]
    file_type                = my_tuple[5]
    
    dictionary = {}
    for i,section in enumerate(sections_list):
        pathname = root_dir+section+'/'+reconstruction_file_name[0]+slices_file_name+file_type
        slices = sorted(glob.glob(pathname))
        if slices==[]:
            pathname = root_dir+section+'/'+reconstruction_file_name[1]+slices_file_name+file_type
            slices = sorted(glob.glob(pathname))

        key = section

        if key not in dictionary:
            dictionary[key] = []

        im = tifffile.imread(slices)

        dictionary[key] = im # add the shape crop here <<<<<------------
    name = sample_name.replace('-','_')
    return dictionary

def modify_dict(dictionary):
    for section in dictionary.keys():
        dictionary.update({section:np.array(dictionary[section])})
        dictionary.update({section:dictionary[section][50:dictionary[section].shape[0]-50,:,:]})
        
    
def mask_section(slices,mask_radius):
    slices_masked = []
    center = (slices.shape[1]/2,slices.shape[2]/2)
    for slice_ in slices:
        # Creating coordinate grids
        x, y = np.meshgrid(np.arange(470), np.arange(470))

        # Calculating the distances from the center
        r = np.sqrt((center[0] - x)**2 + (center[1] - y)**2)

        # Apply np.select to get the diff values
        modified_slice = np.where(r <= mask_radius, slice_,1.17122018)

        # Apply the correction to the image slice
        slices_masked.append(modified_slice)
    slices_masked = np.array(slices_masked)
    return slices_masked

def mask_section_thresholding(slices,mask_radius):
    slices_masked = []
    center = (slices.shape[1]/2,slices.shape[2]/2)
    for slice_ in slices:
        # Creating coordinate grids
        x, y = np.meshgrid(np.arange(470), np.arange(470))

        # Calculating the distances from the center
        r = np.sqrt((center[0] - x)**2 + (center[1] - y)**2)

        # Apply np.select to get the diff values
        modified_slice = np.where(r <= mask_radius, slice_,1.17122018)

        # Apply the correction to the image slice
        slices_masked.append(modified_slice)
    slices_masked = np.array(slices_masked)
    return slices_masked

def plot_rescaling_comparison(sample_name,arr1,arr2):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.hist(arr1[arr1!=0].flatten(), bins=200, label='Original',
                 histtype='step', fill=False,edgecolor='k', density=True)

    ax.hist(arr2[arr2!=0].flatten(), bins=200, label='After Rescaling',
                 histtype='step', fill=False,edgecolor='r', density=True)
    
    (f'{sample_name} Section 2 -Volumetric Histograms: Original vs After Rescaling')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    # Visualizing all sections
def visualize_sections(dictionary,sample_name):
    figsize=(10,len(dictionary.keys())*4)
    fig, ax = plt.subplots(nrows=len(dictionary.keys()), ncols=2, figsize=figsize)
    plt.rcParams['figure.constrained_layout.use'] = True

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].set_box_aspect(1)

    radius = [10, 25, 50, 75, 100, 125, 150, 175, 195]
    angle = np.linspace(0, 2*np.pi, 10000)
    cmap = matplotlib.cm.get_cmap('YlOrRd')

    # Finding the average intensity over all slices:
    for i,section in enumerate(dictionary.keys()):
        im = ax[i,0].imshow(dictionary[section][17,:,:], cmap='gray', interpolation='none',origin='lower',
                            label = f'{section}, Slice 17')
        center = (dictionary[section].shape[1]/2, dictionary[section].shape[2]/2)
        ax[i,0].plot(center[0],center[1], marker='o', color = 'r')

        radius_line = 210
        angle_line = np.deg2rad(45)

        x0,y0 = center
        x1 = radius_line*np.cos(angle_line)+center[0]
        y1 = radius_line*np.sin(angle_line)+center[1]
        ax[i,0].plot([x0,x1],[y0,y1], c='b')

        average = []
        for slice_ in dictionary[section]:
            profile = skimage.measure.profile_line(image=slice_, src=center, dst=(x1,y1))
            average.append(profile)
        average = np.array(average)
        mean_profile = np.mean(average, axis=0)
        ax[i,1].plot(mean_profile, c='k', label = f'{section} Average')

        for q,r in enumerate(radius):
            color = cmap(q / len(radius))  # Get color from the colormap
            x = np.cos(angle)*r+center[0]
            y = np.sin(angle)*r+center[1]
            ax[i,1].axvline(x=r,ymin=0, ymax=50000 , label= f'r = {r}',c=color)
            ax[i,0].plot(x,y, label=f'r = {r}',c=color)

        ax[i,0].legend()
        ax[i,1].legend()
        ax[i,0].set_title(f'{section}, Slice 17')
        ax[i,1].set_title(f'Radial Intensity Profile of {section}')
        fig.colorbar(mappable=im, ax=ax[i,0])

    plt.suptitle(f'{sample_name} Radial Intensity Analysis')
    plt.show()

    
def radially_correct_section(sample_to_correct):
    corrected_sample = {}
    for section in sample_to_correct:
        section_to_correct = sample_to_correct[section]
        center = (470/2, 470/2)
        radius = 190

        x0,y0 = center


        index_arr = []
        for index in np.ndindex((470,470)):
                      index_arr.append(index)
        index_arr = np.array(index_arr)
        x_col = index_arr[:,0]
        y_col = index_arr[:,1]

        r_arr = []
        I_arr = []

        for slice_ in section_to_correct:
            df = pd.DataFrame(data={'x': x_col, 'y': y_col})
            df['r'] = np.sqrt((center[0]-df['x'])**2 + (center[1]-df['y'])**2)
            df['I'] = slice_[df['x'],df['y']]

            # splitting R into 5 px intervals
            bin_edges = np.arange(0,181,1)

            # Bin the 'r' values and calculate the mean of the corresponding 'I' values
            df['r_bin'] = pd.cut(df['r'], bins=bin_edges)
            result = df.groupby('r_bin',observed=False)['I'].mean().reset_index()

            result['r_bin_str'] = result['r_bin'].astype(str)
            result['right_bin'] = result['r_bin'].apply(lambda x: x.left)

            r_arr.append(result['right_bin'].values)
            I_arr.append(result['I'].values)

        r_array = np.array(r_arr)
        I_array = np.array(I_arr)

        r_mean = np.mean(r_array,axis=0)
        I_mean = np.mean(I_array,axis=0)
        average_till_170 = np.mean(I_mean[0:180])
        diff_arr = I_mean - average_till_170 
        I_corrected = I_array-diff_arr
        I_corrected_mean = np.mean(I_corrected,axis=0)

#         differences_modified = np.where(np.abs(diff_arr)>=np.abs(10000), 0, diff_arr)
        intervals = result['r_bin']
        
        descending_gradient = np.arange(r_mean[0],(r_mean[149]-r_mean[0]+1),(r_mean[149]-r_mean[0]+1)/(r_mean[149]+1))
        descending_gradient = np.flip(descending_gradient/np.max(descending_gradient))
        descending_gradient2 = np.append(descending_gradient,np.arange(0,30,1),axis=0)
        differences_modified = diff_arr*descending_gradient2
        differences_modified[151:]=0
        
        
        # Applying to all slices:
        slices = []
        x, y = np.meshgrid(np.arange(470), np.arange(470))

        for slice_ in section_to_correct:
            r = np.sqrt((center[0] - x)**2 + (center[1] - y)**2)
            condlist = [np.logical_and(interval.left < r, r <= interval.right) for interval in intervals]
            choicelist = differences_modified
            diff = np.select(condlist, choicelist, default=0)
            corrected_image1 = slice_ - diff
            slices.append(corrected_image1)

        slices = np.array(slices)
        corrected_sample[section] = slices
        print(section, "is corrected.")
    return corrected_sample

def visualize_radial_correction(sample_name, 
                                original_sample, 
                                corrected_sample, 
                                profile_angle_degrees=45, 
                                slice_to_plot=100,
                                save=True):
    plt.rcParams['figure.constrained_layout.use'] = True
    # Visualizing
    figsize=(12,len(corrected_sample.keys())*12) # play with the multiplier
    fig, ax = plt.subplots(nrows=len(corrected_sample.keys())*2, ncols=2, figsize=figsize)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].set_box_aspect(1)
    
        
    # Finding the average intensity over all slices:
    for i,section in enumerate(original_sample.keys()):
        non_corrected_slice = original_sample[section][slice_to_plot,:,:]
        corrected_slice = corrected_sample[section][slice_to_plot,:,:]
        if i==0:
            k=i
        else:
            k=i*2
        # (0,0)
        im = ax[k,0].imshow(original_sample[section][slice_to_plot,:,:], cmap='gray', interpolation='none',origin='lower',
                            label = f'{section}, Slice {slice_to_plot}', vmin=0,vmax=65535)
        center = (original_sample[section].shape[1]/2, original_sample[section].shape[2]/2)
        ax[k,0].plot(center[0],center[1], marker='o', color = 'r')

        radius_line = 210
        angle_line = np.deg2rad(profile_angle_degrees)

        x0,y0 = center
        x1 = radius_line*np.cos(angle_line)+center[0]
        y1 = radius_line*np.sin(angle_line)+center[1]
        ax[k,0].plot([x0,x1],[y0,y1], c='b')
        ax[k,1].plot([x0,x1],[y0,y1], c='g')
        
        # (0,1)
        im2 = ax[k,1].imshow(corrected_sample[section][slice_to_plot,:,:], cmap='gray', interpolation='none',origin='lower',
                            label = f'{section},  Slice {slice_to_plot} - Corrected', vmin=0,vmax=65535)
        center = (original_sample[section].shape[1]/2, original_sample[section].shape[2]/2)
        ax[k,1].plot(center[0],center[1], marker='o', color = 'r')     
        
        #original average
        or_average = []
        for slice_ in original_sample[section]:
            profile = skimage.measure.profile_line(image=slice_, src=center, dst=(x1,y1))
            or_average.append(profile)
        or_average = np.array(or_average)
        or_mean_profile = np.mean(or_average, axis=0)
        
        #corrected average
        cor_average = []
        for slice_ in corrected_sample[section]:
            profile = skimage.measure.profile_line(image=slice_, src=center, dst=(x1,y1))
            cor_average.append(profile)
        cor_average = np.array(cor_average)
        cor_mean_profile = np.mean(cor_average, axis=0)
        
        # plotting section averages - before & after
        ax[k+1,1].plot(or_mean_profile, c='b', label = f'{section} Average - Original')
        ax[k+1,1].plot(cor_mean_profile, c='g', label = f'{section} Average - Corrected')
        
        # plotting the slice_to_plot profiles - before&after correction
        or_profile = skimage.measure.profile_line(image=original_sample[section][slice_to_plot,:,:], src=center, dst=(x1,y1))
        cor_profile = skimage.measure.profile_line(image=corrected_sample[section][slice_to_plot,:,:], src=center, dst=(x1,y1))
        ax[k+1,0].plot(or_profile, c='b', label = f'{section} Slice {slice_to_plot} - Original')
        ax[k+1,0].plot(cor_profile, c='g', label = f'{section} Slice {slice_to_plot} - Corrected')
        
        radius = [10, 25, 50, 75, 100, 125, 150, 175, 195]
        angle = np.linspace(0, 2*np.pi, 10000)
        cmap = matplotlib.cm.get_cmap('YlOrRd')
        
        
        for q,r in enumerate(radius):
            color = cmap(q / len(radius))  # Get color from the colormap
            x = np.cos(angle)*r+center[0]
            y = np.sin(angle)*r+center[1]
            ax[k+1,0].axvline(x=r,ymin=0, ymax=50000 , label= f'r = {r}',c=color)
            ax[k+1,1].axvline(x=r,ymin=0, ymax=50000 , label= f'r = {r}',c=color)
            ax[k,0].plot(x,y, label=f'r = {r}',c=color)
            ax[k,1].plot(x,y, label=f'r = {r}',c=color)

        ax[k,0].legend()
        ax[k,1].legend()
        ax[k,0].set_title(f'{section}, Slice {slice_to_plot} Before Rad. Corr.')
        ax[k,1].set_title(f'{section}, Slice {slice_to_plot} After Rad. Corr.')
#         fig.colorbar(mappable=im, ax=ax[k,0])
#         fig.colorbar(mappable=im2, ax=ax[k,1])
        
        ax[k+1,0].legend()
        ax[k+1,1].legend()
        ax[k+1,0].set_title(f'{section}, Slice {slice_to_plot} - Before and After')
        ax[k+1,1].set_title(f'{section} Average Slice - Before and After')

    plt.suptitle(f'{sample_name} Beam Hardening Correction')
    
    if save == True:
        folder = '/work/09429/cinarturhan/ls6/Analysis Results/Beam Hardening Correction Analysis Results'
        file_name = sample_name+'Beam Hardening Correction'
        extension='png'
        dummy= '/'.join([folder,sample_name])
        file_dir_and_name = '.'.join([dummy,extension])
        print(f'The image is saved to: \n{file_dir_and_name}')
        plt.savefig(file_dir_and_name, dpi=600, facecolor='w')

    plt.show()

# Not so necessary at this point.

# def visualize_histogram_equalization(sample_name,
#                                  original_sample,
#                                  corrected_sample,
#                                  mask_radius,
#                                  slice_to_plot=100):

#     # Visualizing
#     figsize=(12,len(corrected_sample.keys())*12) # play with the multiplier
#     fig, ax = plt.subplots(nrows=len(corrected_sample.keys())*2, ncols=2, figsize=figsize)
#     plt.rcParams['figure.constrained_layout.use'] = True
#     for i in range(ax.shape[0]):
#         for j in range(ax.shape[1]):
#             ax[i,j].set_box_aspect(1)


#     # Finding the average intensity over all slices:
#     for i,section in enumerate(original_sample.keys()):
#         non_corrected_slice = original_sample[section][slice_to_plot,:,:]
#         corrected_slice = corrected_sample[section][slice_to_plot,:,:]

#     #     masked_section_orj = SRAF.mask_section(original_sample[section],mask_radius)
#     #     masked_section_adapthist = SRAF.mask_section(corrected_sample[section],mask_radius)
#         masked_section_orj = mask_section(original_sample[section],mask_radius)
#         masked_section_adapthist = mask_section(corrected_sample[section],mask_radius)

#         masked_slice_orj = masked_section_orj[slice_to_plot,:,:]
#         masked_slice_adapthist = masked_section_adapthist[slice_to_plot,:,:]


#         if i==0:
#             k=i
#         else:
#             k=i*2

#         # (0,0)
#         im = ax[k,0].imshow(original_sample[section][slice_to_plot,:,:], cmap='gray', interpolation='none',origin='lower',
#                             label = f'{section}, Slice {slice_to_plot}', vmin=0,vmax=65535)

#         # (0,1)
#         im2 = ax[k,1].imshow(corrected_sample[section][slice_to_plot,:,:], cmap='gray', interpolation='none',origin='lower',
#                             label = f'{section},  Slice {slice_to_plot} - Adapt. Hist. Eq.')
#         center = (original_sample[section].shape[1]/2, original_sample[section].shape[2]/2)  
        
#         arr1=masked_slice_orj.flatten()
#         arr2=masked_slice_adapthist.flatten()
#         # plotting the slice_to_plot profiles - before & after adapt. hist. eq.
#         ax[k+1,0].hist(arr1[arr1!=1.17122018]/65535, histtype='step', 
#                        bins=255, color='b', density=True,
#                        label = f'{section} Slice {slice_to_plot} - Original')

#         ax[k+1,0].hist(arr2[arr2!=1.17122018], histtype='step', 
#                        bins=255, color='g', density=True,
#                        label = f'{section} Slice {slice_to_plot} - Corrected')

#         arr1=masked_section_orj.flatten()
#         arr2=masked_section_adapthist.flatten()
        
#         # plotting section histograms - before & after
#         ax[k+1,1].hist(arr1[arr1!=1.17122018]/65535, histtype='step', 
#                        bins=255, color='b', density=True,
#                        label = f'{section} Average - Before Adapt. Hist. Eq.')

#         ax[k+1,1].hist(arr2[arr2!=1.17122018], histtype='step', 
#                        bins=255, color='g', density=True,
#                        label = f'{section} Average - After Adapt. Hist. Eq.')

#         ax[k,0].set_title(f'{section}, Slice {slice_to_plot} Before Adapt. Hist. Eq.')
#         ax[k,1].set_title(f'{section}, Slice {slice_to_plot} After Adapt. Hist. Eq.')

#         ax[k+1,0].legend()
#         ax[k+1,1].legend()
#         ax[k+1,0].set_title(f'{section}, Slice {slice_to_plot} - Before and After')
#         ax[k+1,1].set_title(f'{section} Average Slice - Before and After')

#     plt.suptitle(f'{sample_name} Adaptive Histogram Equalization')
#     plt.show()

    
def visualize_filter(sample_name,
                     original_sample,
                     corrected_sample,
                     mask_radius,
                     filter_name,
                     slice_to_plot=100,
                     save=True):
    
    for word in filter_name.lower().split():
        if word == 'median':
            save_folder_name = 'Median Filter'
            break
        elif word == 'med':
            save_folder_name = 'Median Filter'
            break
        elif word == 'med.':
            save_folder_name = 'Median Filter'
            break

        elif word.lower()=='ani':
            save_folder_name = 'Anisotropic Diffusion'
            break
        elif word.lower()=='anisotropic':
            save_folder_name = 'Anisotropic Diffusion'
            break
        elif word.lower()=='ani.':
            save_folder_name = 'Anisotropic Diffusion'
            break


        elif word.lower()=='hist.':
            save_folder_name = 'Histogram Equalization'
            break
        elif word.lower()=='hist':
            save_folder_name = 'Histogram Equalization'
            break
        elif word.lower()=='histogram':
            save_folder_name = 'Histogram Equalization'
            break
        elif word.lower()=='adapt.':
            save_folder_name = 'Histogram Equalization'
            break
        elif word.lower()=='adapt':
            save_folder_name = 'Histogram Equalization'
            break
        else:
            print('The saving directory for this filter is not implemented yet, save parameter is automatically set to False (save=False)')
            save=False
            break

                  
                  
    # Visualizing
    figsize=(12,len(corrected_sample.keys())*12) # play with the multiplier
    fig, ax = plt.subplots(nrows=len(corrected_sample.keys())*2, ncols=2, figsize=figsize)
    plt.rcParams['figure.constrained_layout.use'] = True
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].set_box_aspect(1)


    # Finding the average intensity over all slices:
    for i,section in enumerate(original_sample.keys()):
        non_corrected_slice = original_sample[section][slice_to_plot,:,:]
        corrected_slice = corrected_sample[section][slice_to_plot,:,:]

    #     masked_section_orj = SRAF.mask_section(original_sample[section],mask_radius)
    #     masked_section_adapthist = SRAF.mask_section(corrected_sample[section],mask_radius)
        masked_section_orj = mask_section(original_sample[section],mask_radius)
        masked_section_adapthist = mask_section(corrected_sample[section],mask_radius)

        masked_slice_orj = masked_section_orj[slice_to_plot,:,:]
        masked_slice_adapthist = masked_section_adapthist[slice_to_plot,:,:]


        if i==0:
            k=i
        else:
            k=i*2
            
        max_val1=np.max(original_sample[section][slice_to_plot,:,:])
        if max_val1>1 and max_val1<=255:
            vmin1=0;vmax1=255
        elif max_val1>255 and max_val1<65536:
            vmin1=0;vmax1=65535
        elif max_val1<=1:
            vmin1=0;vmax1=1
            
        max_val2=np.max(corrected_sample[section][slice_to_plot,:,:])
        if max_val2>1 and max_val2<=255:
            vmin2=0;vmax2=255
        elif max_val2>255 and max_val2<65536:
            vmin2=0;vmax2=65535
        elif max_val2<=1:
            vmin2=0;vmax2=1
            
        # (0,0)
        im = ax[k,0].imshow(original_sample[section][slice_to_plot,:,:], cmap='gray', interpolation='none',origin='lower',
                            label = f'{section}, Slice {slice_to_plot}', vmin=vmin1,vmax=vmax1)

        # (0,1)
        im2 = ax[k,1].imshow(corrected_sample[section][slice_to_plot,:,:], cmap='gray', interpolation='none',origin='lower',
                            label = f'{section},  Slice {slice_to_plot} - {filter_name}.', vmin=vmin2,vmax=vmax2)
        center = (original_sample[section].shape[1]/2, original_sample[section].shape[2]/2)  
        
        arr1=masked_slice_orj.flatten()
        arr2=masked_slice_adapthist.flatten()
        
        # plotting the slice_to_plot profiles - before & after adapt. hist. eq.
        ax[k+1,0].hist(arr1[arr1!=1.17122018]/vmax1, histtype='step', 
                       bins=255, color='b', density=True,
                       label = f'{section} Slice {slice_to_plot} - Before {filter_name}')

        ax[k+1,0].hist(arr2[arr2!=1.17122018]/vmax2, histtype='step', 
                       bins=255, color='g', density=True,
                       label = f'{section} Slice {slice_to_plot} - After {filter_name}')

        arr1=masked_section_orj.flatten()
        arr2=masked_section_adapthist.flatten()
        
        # plotting section histograms - before & after
        ax[k+1,1].hist(arr1[arr1!=1.17122018]/vmax1, histtype='step', 
                       bins=255, color='b', density=True,
                       label = f'{section} Average - Before {filter_name}')

        ax[k+1,1].hist(arr2[arr2!=1.17122018]/vmax2, histtype='step', 
                       bins=255, color='g', density=True,
                       label = f'{section} Average - After {filter_name}')

        ax[k,0].set_title(f'{section}, Slice {slice_to_plot} Before {filter_name}')
        ax[k,1].set_title(f'{section}, Slice {slice_to_plot} After {filter_name}')

        ax[k+1,0].legend()
        ax[k+1,1].legend()
        ax[k+1,0].set_title(f'{section}, Slice {slice_to_plot} - Before and After {filter_name}')
        ax[k+1,1].set_title(f'{section} Average Slice - Before and After {filter_name}')

    plt.suptitle(f'{sample_name} {filter_name}')
            
    if save == True:
        folder = '/work/09429/cinarturhan/ls6/Analysis Results/'+f'{save_folder_name}'+' Analysis Results'
        file_name = sample_name+ f'{filter_name}'
        extension='png'
        dummy= '/'.join([folder,sample_name])
        file_dir_and_name = '.'.join([dummy,extension])
        print(f'The image is saved to: \n{file_dir_and_name}')
        plt.savefig(file_dir_and_name, dpi=600, facecolor='w')
    
    plt.show()
    
    
def compare_filters(sample_name, 
                    data1, 
                    data2, 
                    mask_radius,
                    data1_name='data1',
                    data2_name='data2',
                    slice_to_plot=100,
                    save=True):
                  
    # Visualizing
    figsize=(12,len(data1.keys())*12) # play with the multiplier
    fig, ax = plt.subplots(nrows=len(data1.keys())*2, ncols=2, figsize=figsize)
    plt.rcParams['figure.constrained_layout.use'] = True
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i,j].set_box_aspect(1)


    # Finding the average intensity over all slices:
    for i,section in enumerate(data1.keys()):
        non_corrected_slice = data1[section][slice_to_plot,:,:]
        corrected_slice = data2[section][slice_to_plot,:,:]

    #     masked_section_orj = SRAF.mask_section(data1[section],mask_radius)
    #     masked_section_adapthist = SRAF.mask_section(data2[section],mask_radius)
        masked_section_orj = mask_section(data1[section],mask_radius)
        masked_section_adapthist = mask_section(data2[section],mask_radius)

        masked_slice_orj = masked_section_orj[slice_to_plot,:,:]
        masked_slice_adapthist = masked_section_adapthist[slice_to_plot,:,:]


        if i==0:
            k=i
        else:
            k=i*2
        
        max_val1=np.max(data1[section][slice_to_plot,:,:])
        if max_val1>1 and max_val1<=255:
            vmin1=0;vmax1=255
        elif max_val1>255 and max_val1<65536:
            vmin1=0;vmax1=65535
        elif max_val1<=1:
            vmin1=0;vmax1=1
            
        max_val2=np.max(data2[section][slice_to_plot,:,:])
        if max_val2>1 and max_val2<=255:
            vmin2=0;vmax2=255
        elif max_val2>255 and max_val2<65536:
            vmin2=0;vmax2=65535
        elif max_val2<=1:
            vmin2=0;vmax2=1
        # (0,0)
        im = ax[k,0].imshow(data1[section][slice_to_plot,:,:], cmap='gray', interpolation='none',origin='lower',
                            label = f'{section}, Slice {slice_to_plot} - {data1_name}.', vmin=vmin1,vmax=vmax1)

        # (0,1)
        im2 = ax[k,1].imshow(data2[section][slice_to_plot,:,:], cmap='gray', interpolation='none',origin='lower',
                            label = f'{section}, Slice {slice_to_plot} - {data2_name}.', vmin=vmin2,vmax=vmax2)
        center = (data1[section].shape[1]/2, data1[section].shape[2]/2)  
        
        arr1=masked_slice_orj.flatten()
        arr2=masked_slice_adapthist.flatten()
        
        # plotting the slice_to_plot profiles - before & after adapt. hist. eq.
        ax[k+1,0].hist(arr1[arr1!=1.17122018]/vmax1, histtype='step', 
                       bins=255, color='b', density=True,
                       label = f'{section} Slice {slice_to_plot} - {data1_name}')

        ax[k+1,0].hist(arr2[arr2!=1.17122018]/vmax2, histtype='step', 
                       bins=255, color='g', density=True,
                       label = f'{section} Slice {slice_to_plot} - {data2_name}')

        arr1=masked_section_orj.flatten()
        arr2=masked_section_adapthist.flatten()
        
        # plotting section histograms - before & after
        ax[k+1,1].hist(arr1[arr1!=1.17122018]/vmax1, histtype='step', 
                       bins=255, color='b', density=True,
                       label = f'{section} Average - {data1_name}')

        ax[k+1,1].hist(arr2[arr2!=1.17122018]/vmax2, histtype='step', 
                       bins=255, color='g', density=True,
                       label = f'{section} Average - {data2_name}')

        ax[k,0].set_title(f'{section}, Slice {slice_to_plot} - {data1_name}')
        ax[k,1].set_title(f'{section}, Slice {slice_to_plot} - {data2_name}')

        ax[k+1,0].legend()
        ax[k+1,1].legend()
        ax[k+1,0].set_title(f'{section}, Slice {slice_to_plot} - {data1_name} vs {data2_name}')
        ax[k+1,1].set_title(f'{section} Average Slice - {data1_name} vs {data2_name}')

    plt.suptitle(f'{sample_name} {data1_name} vs {data2_name}')
            
    if save == True:
        save_folder_name='Filter Comparison'
        folder = '/work/09429/cinarturhan/ls6/Analysis Results/'+f'{save_folder_name}'+' Analysis Results'
        file_name = sample_name+ f'{data1_name}'+'vs'+ f'{data2_name}'
        extension='png'
        dummy= '/'.join([folder,sample_name])
        file_dir_and_name = '.'.join([dummy,extension])
        print(f'The image is saved to: \n{file_dir_and_name}')
        plt.savefig(file_dir_and_name, dpi=600, facecolor='w')
    
    plt.show()
    
    
    
def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1):
    """
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to np.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return np.exp(-(delta/kappa)**2.)/float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5*((1.-(delta/kappa_s)**2.)**2.)/float(spacing)
            return np.where(np.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = np.array(img, dtype=np.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [np.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        # calculate the diffs
        for i in range(out.ndim):
#             slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
            slicer = tuple([slice(None, -1) if j == i else slice(None) for j in range(out.ndim)])
            deltas[i][slicer] = np.diff(out, axis=i)

        # update matrices
        matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't ask questions. just do it. trust me.
        for i in range(out.ndim):
#             slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
            slicer = tuple([slice(1, None) if j == i else slice(None) for j in range(out.ndim)])
            matrices[i][slicer] = np.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (np.sum(matrices, axis=0))

    return out


# if __name__ == '__main__':
#     # Create a sample image    
#     np.random.seed(193041)
#     img = np.random.uniform(size=(100,100))
#     img_ani = anisotropic_diffusion(img, kappa=7000, niter=10, gamma=0.2, option=3)
    
#     plt.figure(dpi=400)
#     plt.subplot(1,2,1)
#     plt.imshow(img, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')
    
#     plt.subplot(1,2,2)
#     plt.imshow(img_ani, cmap='gray')
#     plt.title('Anisotropic Diffusion')
#     plt.axis('off')
