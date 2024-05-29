import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'pet_144_NoNoise', 'pet_256_NoNoise','petmri_144_NoNoise','pet_144_NoNoise_bwpm','pet_144_NoNoise_nt','pet_144_NoNoise_nn','pet_144_NoNoise_tp','true_scan','true_scan_suv'}
        assert (database in db_names)

        root_dir = os.path.join(r'C:\Users\Administrator\Desktop\admm','Datasets')
        if database == 'pet_144_NoNoise':
            dataset_file = os.path.join(root_dir,"pet_vol_slice_144by144_NoNoise.npy");
            dataset_info_file = os.path.join(root_dir,"pet_vol_slice_144by144_NoNoise.yml");
            return dataset_file, dataset_info_file
        elif database == 'pet_144_NoNoise_bwpm':
            dataset_file = os.path.join(root_dir, "pet_vol_slice_144by144_bwpm.npy");
            dataset_info_file = os.path.join(root_dir, "pet_vol_slice_144by144_bwpm.yml");
            return dataset_file, dataset_info_file
        elif database == 'pet_144_NoNoise_nt':
            dataset_file = os.path.join(root_dir, "pet_vol_slice_144by144_nt.npy");
            dataset_info_file = os.path.join(root_dir, "pet_vol_slice_144by144_nt.yml");
            return dataset_file, dataset_info_file
        elif database == 'pet_144_NoNoise_nn':
            dataset_file = os.path.join(root_dir, "pet_vol_slice_144by144_nn.npy");
            dataset_info_file = os.path.join(root_dir, "pet_vol_slice_144by144_nn.yml");
            return dataset_file, dataset_info_file
        elif database == 'pet_144_NoNoise_tp':
            dataset_file = os.path.join(root_dir, "pet_vol_slice_144by144_tp.npy");
            dataset_info_file = os.path.join(root_dir, "pet_vol_slice_144by144_tp.yml");
            return dataset_file, dataset_info_file
        elif database == 'pet_256_NoNoise':
            dataset_file = os.path.join(root_dir,"pet_vol_slice_256by256_NoNoise.npy");
            dataset_info_file = os.path.join(root_dir,"pet_vol_slice_256by256_NoNoise.yml");
            return dataset_file, dataset_info_file
        elif database == 'petmri_144_NoNoise':
            dataset_file = os.path.join(root_dir,"petmri_vol_slice_144by144_NoNoise.npy.npz");
            dataset_info_file = os.path.join(root_dir,"petmri_vol_slice_144by144_NoNoise.yml");
            return dataset_file, dataset_info_file
        elif database == 'true_scan':
            dataset_file = os.path.join(root_dir,"pet_mr_true_scan_brain.hdf5");
            dataset_info_file = os.path.join(root_dir,"pet_mr_true_scan_brain.yml");
            return dataset_file, dataset_info_file
        elif database == 'true_scan_suv':
            dataset_file = os.path.join(root_dir,"pet_mr_true_scan_brain_suv.hdf5");
            dataset_info_file = os.path.join(root_dir,"pet_mr_true_scan_brain_suv.yml");
            return dataset_file, dataset_info_file

        else:
            raise NotImplementedError