from multiprocessing import Pool
import subprocess
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image

from totalsegmentator.alignment import as_closest_canonical, undo_canonical
from totalsegmentator.config import get_config_key, set_config_key
from totalsegmentator.config import send_usage_stats, set_license_number
from totalsegmentator.config import setup_nnunet, setup_totalseg, increase_prediction_counter
from totalsegmentator.cropping import crop_to_mask, undo_crop
from totalsegmentator.libs import combine_masks, check_if_shape_and_affine_identical, \
    reorder_multilabel_like_v1
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.libs import nostdout
from totalsegmentator.map_to_binary import class_map
from totalsegmentator.map_to_binary import class_map_5_parts, class_map_parts_mr, \
    class_map_parts_headneck_muscles
from totalsegmentator.map_to_binary import map_taskid_to_partname_mr, map_taskid_to_partname_ct, \
    map_taskid_to_partname_headneck_muscles
from totalsegmentator.nifti_ext_header import add_label_map_to_nifti

from totalsegmentator.postprocessing import extract_skin, remove_auxiliary_labels
from totalsegmentator.postprocessing import keep_largest_blob_multilabel, remove_small_blobs_multilabel
from totalsegmentator.python_api import validate_device_type_api, convert_device_to_cuda, show_license_info
from totalsegmentator.resampling import change_spacing

import inspect

def save_segmentation_nifti(class_map_item, tmp_dir=None, file_out=None, nora_tag=None, header=None, task_name=None, quiet=None):
    k, v = class_map_item
    # Have to load img inside of each thread. If passing it as argument a lot slower.
    if not task_name.startswith("total") and not quiet:
        print(f"Creating {v}.nii.gz")
    img = nib.load(tmp_dir / "s01.nii.gz")
    img_data = img.get_fdata()
    binary_img = img_data == k
    output_path = str(file_out / f"{v}.nii.gz")
    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, header), output_path)
    if nora_tag != "None":
        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)

def supports_keyword_argument(func, keyword: str):
    """
    Check if a function supports a specific keyword argument.

    Returns:
    - True if the function supports the specified keyword argument.
    - False otherwise.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    return keyword in parameters

class TotalSegmentatorBatch:

    @staticmethod
    def predict_from_files(predictor,
                           dir_in, dir_out,
                           continue_prediction,
                           num_threads_preprocessing, num_threads_nifti_save,
                           prev_stage_predictions=None, num_parts=1, part_id=0,
                           save_probabilities_path=None):

        save_probabilities = save_probabilities_path is not None
        dir_in = str(dir_in)
        dir_out = str(dir_out)
        predictor.predict_from_files(dir_in, dir_out,
                                     save_probabilities=save_probabilities,
                                     overwrite=not continue_prediction,
                                     num_processes_preprocessing=num_threads_preprocessing,
                                     num_processes_segmentation_export=num_threads_nifti_save,
                                     folder_with_segs_from_prev_stage=prev_stage_predictions,
                                     num_parts=num_parts, part_id=part_id)

        if save_probabilities:
            shutil.copy(Path(dir_out) / "s01.npz", save_probabilities_path)
            shutil.copy(Path(dir_out) / "s01.pkl", save_probabilities_path.with_suffix(".pkl"))

    @staticmethod
    def load_nnunet_model(task_id, model="3d_fullres", folds=None,
                          trainer="nnUNetTrainer", tta=False,
                          plans="nnUNetPlans", device="cuda", quiet=False, step_size=0.5):
        multimodel = type(task_id) is list

        if multimodel:
            predictor = dict()
            for idx, tid in enumerate(task_id):
                predictor[tid] = TotalSegmentatorBatch._load_nnunet_model(task_id=tid, model=model, folds=folds,
                                                                          tta=tta, plans=plans, device=device, quiet=quiet,
                                                                          step_size=step_size, trainer=trainer)
        else:
            predictor = TotalSegmentatorBatch._load_nnunet_model(task_id=task_id, model=model, folds=folds,
                                                                 tta=tta, plans=plans, device=device,quiet=quiet,
                                                                 trainer=trainer,
                                                                 step_size=step_size)
        return predictor
    @staticmethod
    def _load_nnunet_model(task_id, model="3d_fullres", folds=None,
                          trainer="nnUNetTrainer", tta=False,
                          plans="nnUNetPlans", device="cuda", quiet=False, step_size=0.5):
        from nnunetv2.utilities.file_path_utilities import get_output_folder
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor



        model_folder = get_output_folder(task_id, trainer, plans, model)

        assert device in ['cpu', 'cuda',
                          'mps'] or isinstance(device,
                                               torch.device), f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
        if device == 'cpu':
            # let's allow torch to use hella threads
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device('cpu')
        elif device == 'cuda':
            # multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            # torch.set_num_interop_threads(1)  # throws error if setting the second time
            device = torch.device('cuda')
        elif isinstance(device, torch.device):
            torch.set_num_threads(1)
            device = device
        else:
            device = torch.device('mps')
        disable_tta = not tta
        verbose = False
        chk = "checkpoint_final.pth"

        allow_tqdm = not quiet

        if supports_keyword_argument(nnUNetPredictor, "perform_everything_on_gpu"):
            predictor = nnUNetPredictor(
                tile_step_size=step_size,
                use_gaussian=True,
                use_mirroring=not disable_tta,
                perform_everything_on_gpu=True,  # for nnunetv2<=2.2.1
                device=device,
                verbose=verbose,
                verbose_preprocessing=verbose,
                allow_tqdm=allow_tqdm
            )
        # nnUNet >= 2.2.2
        else:
            predictor = nnUNetPredictor(
                tile_step_size=step_size,
                use_gaussian=True,
                use_mirroring=not disable_tta,
                perform_everything_on_device=True,  # for nnunetv2>=2.2.2
                device=device,
                verbose=verbose,
                verbose_preprocessing=verbose,
                allow_tqdm=allow_tqdm
            )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=folds,
            checkpoint_name=chk,
        )
        return predictor

    @staticmethod
    def nnUNet_predict_image(nnunet_predictor, file_in: Union[str, Path, Nifti1Image], file_out, task_id,
                             multilabel_image=True,
                             resample=None, crop=None, crop_path=None, task_name="total", nora_tag="None",
                             preview=False,
                             save_binary=False, nr_threads_resampling=1, nr_threads_saving=6, force_split=False,
                             crop_addon=[3, 3, 3], roi_subset=None, output_type="nifti",
                             statistics=False, quiet=False, verbose=False, test=0, skip_saving=False,
                             exclude_masks_at_border=True, no_derived_masks=False,
                             v1_order=False, stats_aggregation="mean", remove_small_blobs=False,
                             normalized_intensities=False, nnunet_resampling=False,
                             save_probabilities=None, cascade=None):
        """
        crop: string or a nibabel image
        resample: None or float (target spacing for all dimensions) or list of floats
        """

        if not isinstance(file_in, Nifti1Image):
            file_in = Path(file_in)
            if str(file_in).endswith(".nii") or str(file_in).endswith(".nii.gz"):
                img_type = "nifti"
            else:
                img_type = "dicom"
            if not file_in.exists():
                sys.exit("ERROR: The input file or directory does not exist.")
        else:
            img_type = "nifti"
        if file_out is not None:
            file_out = Path(file_out)

        multimodel = type(task_id) is list

        if img_type == "nifti" and output_type == "dicom":
            raise ValueError("To use output type dicom you also have to use a Dicom image as input.")

        if task_name == "total":
            class_map_parts = class_map_5_parts
            map_taskid_to_partname = map_taskid_to_partname_ct
        elif task_name == "total_mr":
            class_map_parts = class_map_parts_mr
            map_taskid_to_partname = map_taskid_to_partname_mr
        elif task_name == "headneck_muscles":
            class_map_parts = class_map_parts_headneck_muscles
            map_taskid_to_partname = map_taskid_to_partname_headneck_muscles

        if type(resample) is float:
            resample = [resample, resample, resample]

        if v1_order and task_name == "total":
            label_map = class_map["total_v1"]
        else:
            label_map = class_map[task_name]

        # Keep only voxel values corresponding to the roi_subset
        if roi_subset is not None:
            label_map = {k: v for k, v in label_map.items() if v in roi_subset}

        # for debugging
        # tmp_dir = file_in.parent / ("nnunet_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
        # (tmp_dir).mkdir(exist_ok=True)
        # with tmp_dir as tmp_folder:
        with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
            tmp_dir = Path(tmp_folder)
            if verbose: print(f"tmp_dir: {tmp_dir}")

            if img_type == "dicom":
                from totalsegmentator.dicom_io import dcm_to_nifti, save_mask_as_rtstruct
                if not quiet: print("Converting dicom to nifti...")
                (tmp_dir / "dcm").mkdir()  # make subdir otherwise this file would be included by nnUNet_predict
                dcm_to_nifti(file_in, tmp_dir / "dcm" / "converted_dcm.nii.gz", tmp_dir, verbose=verbose)
                file_in_dcm = file_in
                file_in = tmp_dir / "dcm" / "converted_dcm.nii.gz"

                # for debugging
                # shutil.copy(file_in, file_in_dcm.parent / "converted_dcm_TMP.nii.gz")

                # Workaround to be able to access file_in on windows (see issue #106)
                # if platform.system() == "Windows":
                #     file_in = file_in.NamedTemporaryFile(delete = False)
                #     file_in.close()

                # if not multilabel_image:
                #     shutil.copy(file_in, file_out / "input_file.nii.gz")
                if not quiet: print(f"  found image with shape {nib.load(file_in).shape}")

            if isinstance(file_in, Nifti1Image):
                img_in_orig = file_in
            else:
                img_in_orig = nib.load(file_in)
            if len(img_in_orig.shape) == 2:
                raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")
            if len(img_in_orig.shape) > 3:
                print(
                    f"WARNING: Input image has {len(img_in_orig.shape)} dimensions. Only using first three dimensions.")
                img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:, :, :, 0], img_in_orig.affine)

            img_dtype = img_in_orig.get_data_dtype()
            if img_dtype.fields is not None:
                raise TypeError(f"Invalid dtype {img_dtype}. Expected a simple dtype, not a structured one.")

            # takes ~0.9s for medium image
            img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)  # copy img_in_orig

            if crop is not None:
                if type(crop) is str:
                    if crop == "lung" or crop == "pelvis":
                        crop_mask_img = combine_masks(crop_path, crop)
                    else:
                        crop_mask_img = nib.load(crop_path / f"{crop}.nii.gz")
                else:
                    crop_mask_img = crop

                if crop_mask_img.get_fdata().sum() == 0:
                    if not quiet:
                        print("INFO: Crop is empty. Returning empty segmentation.")
                    img_out = nib.Nifti1Image(np.zeros(img_in.shape, dtype=np.uint8), img_in.affine)
                    img_out = add_label_map_to_nifti(img_out, label_map)
                    if file_out is not None:
                        if multilabel_image:
                            file_out.parent.mkdir(exist_ok=True, parents=True)
                            nib.save(img_out, file_out)
                        else:
                            file_out.mkdir(exist_ok=True, parents=True)
                            # Save an empty nifti for each roi in roi_subset
                            empty_img = np.zeros(img_in.shape, dtype=np.uint8)
                            for _, roi_name in label_map.items():
                                nib.save(nib.Nifti1Image(empty_img, img_in.affine), file_out / f"{roi_name}.nii.gz")
                            if nora_tag != "None":
                                subprocess.call(
                                    f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas",
                                    shell=True)
                        return img_out, img_in_orig, None

                img_in, bbox = crop_to_mask(img_in, crop_mask_img, addon=crop_addon, dtype=np.int32, verbose=verbose)
                if cascade:
                    cascade, _ = crop_to_mask(cascade, crop_mask_img, addon=crop_addon, dtype=np.uint8, verbose=verbose)
                if not quiet:
                    print(f"  cropping from {crop_mask_img.shape} to {img_in.shape}")

            img_in = as_closest_canonical(img_in)
            if cascade:
                cascade = as_closest_canonical(cascade)

            if resample is not None:
                if not quiet: print("Resampling...")
                st = time.time()
                img_in_shape = img_in.shape
                img_in_zooms = img_in.header.get_zooms()
                img_in_rsp = change_spacing(img_in, resample,
                                            order=3, dtype=np.int32,
                                            nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it a bit slower
                if cascade:
                    cascade = change_spacing(cascade, resample,
                                             order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling)
                if verbose:
                    print(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
                if not quiet: print(f"  Resampled in {time.time() - st:.2f}s")
            else:
                img_in_rsp = img_in

            nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")

            if cascade:
                nib.save(cascade, tmp_dir / "s01_0001.nii.gz")

            # todo important: change
            nr_voxels_thr = 512 * 512 * 900
            # nr_voxels_thr = 256*256*900
            img_parts = ["s01"]
            ss = img_in_rsp.shape
            # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
            # splitting along it does not really make sense.
            do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
            if force_split:
                do_triple_split = True
            if cascade:
                do_triple_split = False
            if do_triple_split:
                if not quiet: print("Splitting into subparts...")
                img_parts = ["s01", "s02", "s03"]
                third = img_in_rsp.shape[2] // 3
                margin = 20  # set margin with fixed values to avoid rounding problem if using percentage of third
                img_in_rsp_data = img_in_rsp.get_fdata()
                nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, :third + margin], img_in_rsp.affine),
                         tmp_dir / "s01_0000.nii.gz")
                nib.save(
                    nib.Nifti1Image(img_in_rsp_data[:, :, third + 1 - margin:third * 2 + margin], img_in_rsp.affine),
                    tmp_dir / "s02_0000.nii.gz")
                nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third * 2 + 1 - margin:], img_in_rsp.affine),
                         tmp_dir / "s03_0000.nii.gz")

            if task_name == "total" and resample is not None and resample[0] < 3.0:
                # overall speedup for 15mm model roughly 11% (GPU) and 100% (CPU)
                # overall speedup for  3mm model roughly  0% (GPU) and  10% (CPU)
                # (dice 0.001 worse on test set -> ok)
                # (for lung_trachea_bronchia somehow a lot lower dice)
                step_size = 0.8
            else:
                step_size = 0.5

            st = time.time()
            if multimodel:  # if running multiple models

                # only compute model parts containing the roi subset
                if roi_subset is not None:
                    part_names = []
                    new_task_id = []
                    for part_name, part_map in class_map_parts.items():
                        if any(organ in roi_subset for organ in part_map.values()):
                            # get taskid associated to model part_name
                            map_partname_to_taskid = {v: k for k, v in map_taskid_to_partname.items()}
                            new_task_id.append(map_partname_to_taskid[part_name])
                            part_names.append(part_name)
                    task_id = new_task_id
                    if verbose:
                        print(f"Computing parts: {part_names} based on the provided roi_subset")

                if test == 0:
                    class_map_inv = {v: k for k, v in class_map[task_name].items()}
                    (tmp_dir / "parts").mkdir(exist_ok=True)
                    seg_combined = {}
                    # iterate over subparts of image
                    for img_part in img_parts:
                        img_shape = nib.load(tmp_dir / f"{img_part}_0000.nii.gz").shape
                        seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
                    # Run several tasks and combine results into one segmentation
                    for idx, tid in enumerate(task_id):
                        if not quiet: print(f"Predicting part {idx + 1} of {len(task_id)} ...")
                        with nostdout(verbose):
                            # nnUNet_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                            #                nr_threads_resampling, nr_threads_saving)
                            TotalSegmentatorBatch.predict_from_files(nnunet_predictor[tid],
                                                                     tmp_dir, tmp_dir, continue_prediction=False,
                                                                     num_threads_preprocessing=nr_threads_resampling,
                                                                     num_threads_nifti_save=nr_threads_saving,
                                                                     save_probabilities_path=save_probabilities)
                        # iterate over models (different sets of classes)
                        for img_part in img_parts:
                            (tmp_dir / f"{img_part}.nii.gz").rename(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz")
                            seg = nib.load(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz").get_fdata()
                            for jdx, class_name in class_map_parts[map_taskid_to_partname[tid]].items():
                                seg_combined[img_part][seg == jdx] = class_map_inv[class_name]
                    # iterate over subparts of image
                    for img_part in img_parts:
                        nib.save(nib.Nifti1Image(seg_combined[img_part], img_in_rsp.affine),
                                 tmp_dir / f"{img_part}.nii.gz")
                elif test == 1:
                    print("WARNING: Using reference seg instead of prediction for testing.")
                    shutil.copy(Path("tests") / "reference_files" / "example_seg.nii.gz", tmp_dir / "s01.nii.gz")
            else:
                if not quiet: print("Predicting...")
                if test == 0:
                    with nostdout(verbose):
                        # nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                        #                nr_threads_resampling, nr_threads_saving)
                        # nnUNetv2_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                        #                  nr_threads_resampling, nr_threads_saving,
                        #                  device=device, quiet=quiet, step_size=step_size,
                        #                  save_probabilities_path=save_probabilities)
                        TotalSegmentatorBatch.predict_from_files(nnunet_predictor,
                                                                 tmp_dir, tmp_dir, continue_prediction=False,
                                                                 num_threads_preprocessing=nr_threads_resampling,
                                                                 num_threads_nifti_save=nr_threads_saving,
                                                                 save_probabilities_path=save_probabilities)
                # elif test == 2:
                #     print("WARNING: Using reference seg instead of prediction for testing.")
                #     shutil.copy(Path("tests") / "reference_files" / "example_seg_fast.nii.gz", tmp_dir / f"s01.nii.gz")
                elif test == 3:
                    print("WARNING: Using reference seg instead of prediction for testing.")
                    shutil.copy(Path("tests") / "reference_files" / "example_seg_lung_vessels.nii.gz",
                                tmp_dir / "s01.nii.gz")
            if not quiet: print(f"  Predicted in {time.time() - st:.2f}s")

            # Combine image subparts back to one image
            if do_triple_split:
                combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
                combined_img[:, :, :third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[:, :, :-margin]
                combined_img[:, :, third:third * 2] = nib.load(tmp_dir / "s02.nii.gz").get_fdata()[:, :,
                                                      margin - 1:-margin]
                combined_img[:, :, third * 2:] = nib.load(tmp_dir / "s03.nii.gz").get_fdata()[:, :, margin - 1:]
                nib.save(nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz")

            img_pred = nib.load(tmp_dir / "s01.nii.gz")

            # Currently only relevant for T304 (appendicular bones)
            img_pred = remove_auxiliary_labels(img_pred, task_name)

            # Postprocessing multilabel (run here on lower resolution)
            if task_name == "body":
                img_pred_pp = keep_largest_blob_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                           class_map[task_name], ["body_trunc"], debug=False,
                                                           quiet=quiet)
                img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

            if task_name == "body":
                vox_vol = np.prod(img_pred.header.get_zooms())
                size_thr_mm3 = 50000
                img_pred_pp = remove_small_blobs_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                            class_map[task_name], ["body_extremities"],
                                                            interval=[size_thr_mm3 / vox_vol, 1e10], debug=False,
                                                            quiet=quiet)
                img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

            # General postprocessing
            if remove_small_blobs:
                if not quiet: print("Removing small blobs...")
                st = time.time()
                vox_vol = np.prod(img_pred.header.get_zooms())
                size_thr_mm3 = 200
                img_pred_pp = remove_small_blobs_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                            class_map[task_name], list(class_map[task_name].values()),
                                                            interval=[size_thr_mm3 / vox_vol, 1e10], debug=False,
                                                            quiet=quiet)  # ~24s
                img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)
                if not quiet: print(f"  Removed in {time.time() - st:.2f}s")

            if preview:
                from totalsegmentator.preview import generate_preview
                # Generate preview before upsampling so it is faster and still in canonical space
                # for better orientation.
                if not quiet: print("Generating preview...")
                if file_out is None:
                    print("WARNING: No output directory specified. Skipping preview generation.")
                else:
                    st = time.time()
                    smoothing = 20
                    preview_dir = file_out.parent if multilabel_image else file_out
                    generate_preview(img_in_rsp, preview_dir / f"preview_{task_name}.png", img_pred.get_fdata(),
                                     smoothing,
                                     task_name)
                    if not quiet: print(f"  Generated in {time.time() - st:.2f}s")

            # Statistics calculated on the 3mm downsampled image are very similar to statistics
            # calculated on the original image. Volume often completely identical. For intensity
            # some more change but still minor.
            #
            # Speed:
            # stats on 1.5mm: 37s
            # stats on 3.0mm: 4s    -> great improvement
            stats = None
            if statistics:
                if not quiet: print("Calculating statistics fast...")
                st = time.time()
                if file_out is not None:
                    stats_dir = file_out.parent if multilabel_image else file_out
                    stats_dir.mkdir(exist_ok=True)
                    stats_file = stats_dir / "statistics.json"
                else:
                    stats_file = None
                from totalsegmentator.statistics import get_basic_statistics
                stats = get_basic_statistics(img_pred.get_fdata(), img_in_rsp, stats_file,
                                             quiet, task_name, exclude_masks_at_border, roi_subset,
                                             metric=stats_aggregation,
                                             normalized_intensities=normalized_intensities)
                if not quiet: print(f"  calculated in {time.time() - st:.2f}s")

            if resample is not None:
                if not quiet: print("Resampling...")
                if verbose: print(f"  back to original shape: {img_in_shape}")
                # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
                # by undo_canonical)

                # Advantage of nnunet_resampling: Will convert multilabel to one-hot and then high order
                # resampling (= smoother) is possible. Disadvantage: slower and uses a lot of memory if many labels.
                # Ok if using with "roi_subset" because then only a few labels, otherwise infeasible runtime+memory.
                if nnunet_resampling:
                    if roi_subset is not None:
                        img_data_tmp = img_pred.get_fdata()
                        img_data_tmp *= np.isin(img_data_tmp, list(label_map.keys()))
                        img_pred = nib.Nifti1Image(img_data_tmp, img_pred.affine)

                    # Order:
                    # 0: roughy and uneven
                    # 1: best (smoothest)
                    # 2: somehow more uneven again
                    # 3: identical to 2
                    img_pred = change_spacing(img_pred, resample, img_in_shape,
                                              order=1, dtype=np.uint8, nr_cpus=nr_threads_resampling,
                                              force_affine=img_in.affine, nnunet_resample=True)
                else:
                    img_pred = change_spacing(img_pred, resample, img_in_shape,
                                              order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling,
                                              force_affine=img_in.affine)

            if verbose: print("Undoing canonical...")
            img_pred = undo_canonical(img_pred, img_in_orig)

            if crop is not None:
                if verbose: print("Undoing cropping...")
                img_pred = undo_crop(img_pred, img_in_orig, bbox)

            check_if_shape_and_affine_identical(img_in_orig, img_pred)

            img_data = img_pred.get_fdata().astype(np.uint8)
            if save_binary:
                img_data = (img_data > 0).astype(np.uint8)

            # Reorder labels if needed
            if v1_order and task_name == "total":
                img_data = reorder_multilabel_like_v1(img_data, class_map["total"], class_map["total_v1"])

            # Keep only voxel values corresponding to the roi_subset
            if roi_subset is not None:
                img_data *= np.isin(img_data, list(label_map.keys()))

            # Prepare output nifti
            # Copy header to make output header exactly the same as input. But change dtype otherwise it will be
            # float or int and therefore the masks will need a lot more space.
            # (infos on header: https://nipy.org/nibabel/nifti_images.html)
            new_header = img_in_orig.header.copy()
            new_header.set_data_dtype(np.uint8)
            img_out = nib.Nifti1Image(img_data, img_pred.affine, new_header)
            img_out = add_label_map_to_nifti(img_out, label_map)

            if file_out is not None and skip_saving is False:
                if not quiet: print("Saving segmentations...")

                # Select subset of classes if required
                selected_classes = class_map[task_name]
                if roi_subset is not None:
                    selected_classes = {k: v for k, v in selected_classes.items() if v in roi_subset}

                if output_type == "dicom":
                    from totalsegmentator.dicom_io import dcm_to_nifti, save_mask_as_rtstruct
                    file_out.mkdir(exist_ok=True, parents=True)
                    save_mask_as_rtstruct(img_data, selected_classes, file_in_dcm, file_out / "segmentations.dcm")
                else:
                    st = time.time()
                    if multilabel_image:
                        file_out.parent.mkdir(exist_ok=True, parents=True)
                    else:
                        file_out.mkdir(exist_ok=True, parents=True)
                    if multilabel_image:
                        nib.save(img_out, file_out)
                        if nora_tag != "None":
                            subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas",
                                            shell=True)
                    else:  # save each class as a separate binary image
                        file_out.mkdir(exist_ok=True, parents=True)

                        if np.prod(img_data.shape) > 512 * 512 * 1000:
                            print("Shape of output image is very big. Setting nr_threads_saving=1 to save memory.")
                            nr_threads_saving = 1

                        # Code for single threaded execution  (runtime:24s)
                        if nr_threads_saving == 1:
                            for k, v in selected_classes.items():
                                binary_img = img_data == k
                                output_path = str(file_out / f"{v}.nii.gz")
                                nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img_pred.affine, new_header),
                                         output_path)
                                if nora_tag != "None":
                                    subprocess.call(
                                        f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask",
                                        shell=True)
                        else:
                            nib.save(img_pred, tmp_dir / "s01.nii.gz")  # needed inside of threads

                            # Code for multithreaded execution
                            #   Speed with different number of threads:
                            #   1: 46s, 2: 24s, 6: 11s, 10: 8s, 14: 8s
                            # _ = p_map(partial(save_segmentation_nifti, tmp_dir=tmp_dir, file_out=file_out, nora_tag=nora_tag, header=new_header, task_name=task_name, quiet=quiet),
                            #         selected_classes.items(), num_cpus=nr_threads_saving, disable=quiet)

                            # Multihreaded saving with same functions as in nnUNet -> same speed as p_map
                            pool = Pool(nr_threads_saving)
                            results = []
                            for k, v in selected_classes.items():
                                results.append(pool.starmap_async(save_segmentation_nifti, [
                                    ((k, v), tmp_dir, file_out, nora_tag, new_header, task_name, quiet)]))
                            _ = [i.get() for i in results]  # this actually starts the execution of the async functions
                            pool.close()
                            pool.join()
                if not quiet: print(f"  Saved in {time.time() - st:.2f}s")

                # Postprocessing single files
                #    (these not directly transferable to multilabel)

                # Lung mask does not exist since I use 6mm model. Would have to save lung mask from 6mm seg.
                # if task_name == "lung_vessels":
                #     remove_outside_of_mask(file_out / "lung_vessels.nii.gz", file_out / "lung.nii.gz")

                # if task_name == "heartchambers_test":
                #     remove_outside_of_mask(file_out / "heart_myocardium.nii.gz", file_out / "heart.nii.gz", addon=5)
                #     remove_outside_of_mask(file_out / "heart_atrium_left.nii.gz", file_out / "heart.nii.gz", addon=5)
                #     remove_outside_of_mask(file_out / "heart_ventricle_left.nii.gz", file_out / "heart.nii.gz", addon=5)
                #     remove_outside_of_mask(file_out / "heart_atrium_right.nii.gz", file_out / "heart.nii.gz", addon=5)
                #     remove_outside_of_mask(file_out / "heart_ventricle_right.nii.gz", file_out / "heart.nii.gz", addon=5)
                #     remove_outside_of_mask(file_out / "aorta.nii.gz", file_out / "heart.nii.gz", addon=5)
                #     remove_outside_of_mask(file_out / "pulmonary_artery.nii.gz", file_out / "heart.nii.gz", addon=5)

                if task_name == "body" and not multilabel_image and not no_derived_masks:
                    if not quiet: print("Creating body.nii.gz")
                    body_img = combine_masks(file_out, "body")
                    nib.save(body_img, file_out / "body.nii.gz")
                    if not quiet: print("Creating skin.nii.gz")
                    skin = extract_skin(img_in_orig, nib.load(file_out / "body.nii.gz"))
                    nib.save(skin, file_out / "skin.nii.gz")

        return img_out, img_in_orig, stats

    def predict(self, input: Union[str, Path, Nifti1Image],
                output: Union[str, Path, None] = None,
                ml=False,
                nr_thr_resamp=1,
                nr_thr_saving=6,
                fast=False, nora_tag="None", preview=False,
                statistics=False, radiomics=False,
                crop_path=None, body_seg=False,
                force_split=False, output_type="nifti", quiet=False, verbose=False, test=0,
                skip_saving=False,
                statistics_exclude_masks_at_border=True, no_derived_masks=False,
                v1_order=False, fastest=False, roi_subset_robust=None, stats_aggregation="mean",
                remove_small_blobs=False, statistics_normalized_intensities=False,
                higher_order_resampling=False, save_probabilities=None
                ):
        """
        Run TotalSegmentator from within python.

        For explanation of the arguments see description of command line
        arguments in bin/TotalSegmentator.

        Return: multilabel Nifti1Image
        """
        if not isinstance(input, Nifti1Image):
            input = Path(input)

        if output is not None:
            output = Path(output)
        else:
            if radiomics:
                raise ValueError("Output path is required for radiomics.")

        if output_type == "dicom":
            try:
                from rt_utils import RTStructBuilder
            except ImportError:
                raise ImportError(
                    "rt_utils is required for output_type='dicom'. Please install it with 'pip install rt_utils'.")

        if not quiet:
            print("\nIf you use this tool please cite: https://pubs.rsna.org/doi/10.1148/ryai.230024\n")


        nora_tag = "None" if nora_tag is None else nora_tag

        # quiet = self.quiet
        # nora_tag = self.nora_tag
        crop = self.crop
        cascade = self.cascade
        roi_subset = self.roi_subset
        # robust_rs = self.robust_rs
        robust_crop = self.robust_crop
        task = self.task
        # verbose = self.verbose
        crop_path = output if crop_path is None else crop_path

        if isinstance(input, Nifti1Image) or input.suffix == ".nii" or input.suffixes[-2:] == [".nii", ".gz"]:
            img_type = "nifti"
        else:
            img_type = "dicom"

        # fast statistics are calculated on the downsampled image
        if statistics and fast:
            statistics_fast = True
            statistics = False
        else:
            statistics_fast = False

        # Generate rough organ segmentation (6mm) for speed up if crop or roi_subset is used
        # (for "fast" on GPU it makes no big difference, but on CPU it can help even for "fast")
        if crop is not None or roi_subset is not None or cascade:
            body_seg = False  # can not be used together with body_seg
            st = time.time()
            if not quiet: print("Generating rough segmentation for cropping...")
            if self.robust_rs or robust_crop:
                print("  (Using more robust (but slower) 3mm model for cropping.)")
                crop_model_task = 852 if task.endswith("_mr") else 297
                crop_spacing = 3.0
            else:
                # For MR always run 3mm model for cropping, because 6mm too bad results
                #  (runtime for 3mm still very good for MR)
                if task.endswith("_mr"):
                    crop_model_task = 852
                    crop_spacing = 3.0
                else:
                    crop_model_task = 298
                    crop_spacing = 6.0
            crop_task = "total_mr" if task.endswith("_mr") else "total"
            # crop_trainer = "nnUNetTrainer_2000epochs_NoMirroring" if task.endswith(
            #     "_mr") else "nnUNetTrainer_4000epochs_NoMirroring"
            if crop is not None and ("body_trunc" in crop or "body_extremities" in crop):
                crop_model_task = 300
                crop_spacing = 6.0
                # crop_trainer = "nnUNetTrainer"
                crop_task = "body"
            # download_pretrained_weights(crop_model_task)

            organ_seg, _, _ = self.nnUNet_predict_image(nnunet_predictor=self.organ_model,
                                                        file_in=input, file_out=None,
                                                        task_id=crop_model_task, multilabel_image=True,
                                                        resample=crop_spacing,
                                                        crop=None, crop_path=None, task_name=crop_task, nora_tag="None",
                                                        preview=False,
                                                        save_binary=False,
                                                        nr_threads_saving=1,
                                                        statistics=False,
                                                        quiet=quiet, verbose=verbose, test=0, skip_saving=False,
                                                        )
            class_map_inv = {v: k for k, v in class_map[crop_task].items()}
            crop_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
            organ_seg_data = organ_seg.get_fdata()
            # roi_subset_crop = [map_to_total[roi] if roi in map_to_total else roi for roi in roi_subset]
            roi_subset_crop = crop if crop is not None else roi_subset
            for roi in roi_subset_crop:
                crop_mask[organ_seg_data == class_map_inv[roi]] = 1
            crop_mask = nib.Nifti1Image(crop_mask, organ_seg.affine)
            # crop_addon = [20, 20, 20]
            crop = crop_mask
            # cascade = crop_mask if cascade else None
            if verbose: print(f"Rough organ segmentation generated in {time.time() - st:.2f}s")

        # Generate rough body segmentation (6mm) (speedup for big images; not useful in combination with --fast option)
        if crop is None and body_seg:
            st = time.time()
            if not quiet: print("Generating rough body segmentation...")
            body_seg, _, _ = self.nnUNet_predict_image(nnunet_predictor=self.body_seg_model,
                                                       file_in=input,
                                                       file_out=None,
                                                       task_id=300,
                                                       multilabel_image=True,
                                                       resample=6.0,
                                                       crop=None, crop_path=None, task_name="body", nora_tag="None",
                                                       preview=False,
                                                       save_binary=True,
                                                       nr_threads_resampling=nr_thr_resamp,
                                                       nr_threads_saving=1,
                                                       crop_addon=self.crop_addon, output_type=output_type,
                                                       statistics=False,
                                                       quiet=quiet, verbose=verbose, test=0, skip_saving=False,
                                                       )
            crop = body_seg
            if verbose: print(f"Rough body segmentation generated in {time.time() - st:.2f}s")

        # folds = [0]  # None
        seg_img, ct_img, stats = self.nnUNet_predict_image(nnunet_predictor=self.model,
                                                           file_in=input,
                                                           file_out=output,
                                                           task_id=self.task_id,
                                                           multilabel_image=ml,
                                                           resample=self.resample,
                                                           crop=crop,
                                                           crop_path=crop_path,
                                                           task_name=task,
                                                           nora_tag=nora_tag,
                                                           preview=preview,
                                                           nr_threads_resampling=nr_thr_resamp,
                                                           nr_threads_saving=nr_thr_saving,
                                                           force_split=force_split,
                                                           crop_addon=self.crop_addon,
                                                           roi_subset=roi_subset,
                                                           output_type=output_type,
                                                           statistics=statistics_fast,
                                                           quiet=quiet,
                                                           verbose=verbose,
                                                           test=test,
                                                           skip_saving=skip_saving,
                                                           exclude_masks_at_border=statistics_exclude_masks_at_border,
                                                           no_derived_masks=no_derived_masks,
                                                           v1_order=v1_order,
                                                           stats_aggregation=stats_aggregation,
                                                           remove_small_blobs=remove_small_blobs,
                                                           normalized_intensities=statistics_normalized_intensities,
                                                           nnunet_resampling=higher_order_resampling,
                                                           save_probabilities=save_probabilities,
                                                           cascade=cascade
                                                           )
        # seg_img, ct_img, stats = nnUNet_predict_image(input, output, task_id, model=model, folds=folds,
        #                                               trainer=trainer, tta=False, multilabel_image=ml,
        #                                               resample=resample,
        #                                               crop=crop, crop_path=crop_path, task_name=task, nora_tag=nora_tag,
        #                                               preview=preview,
        #                                               nr_threads_resampling=nr_thr_resamp,
        #                                               nr_threads_saving=nr_thr_saving,
        #                                               force_split=force_split, crop_addon=crop_addon,
        #                                               roi_subset=roi_subset,
        #                                               output_type=output_type, statistics=statistics_fast,
        #                                               quiet=quiet, verbose=verbose, test=test, skip_saving=skip_saving,
        #                                               device=device,
        #                                               exclude_masks_at_border=statistics_exclude_masks_at_border,
        #                                               no_derived_masks=no_derived_masks, v1_order=v1_order,
        #                                               stats_aggregation=stats_aggregation,
        #                                               remove_small_blobs=remove_small_blobs,
        #                                               normalized_intensities=statistics_normalized_intensities,
        #                                               nnunet_resampling=higher_order_resampling,
        #                                               save_probabilities=save_probabilities,
        #                                               cascade=cascade)
        seg = seg_img.get_fdata().astype(np.uint8)

        try:
            # this can result in error if running multiple processes in parallel because all try to write the same file.
            # Trying to fix with lock from portalocker did not work. Network drive seems to not support this locking.
            config = increase_prediction_counter()
            send_usage_stats(config, {"task": task, "fast": fast, "preview": preview,
                                      "multilabel": ml, "roi_subset": roi_subset,
                                      "statistics": statistics, "radiomics": radiomics})
        except Exception as e:
            # print(f"Error while sending usage stats: {e}")
            pass

        if statistics:
            if not quiet: print("Calculating statistics...")
            st = time.time()
            if output is not None:
                stats_dir = output.parent if ml else output
                stats_file = stats_dir / "statistics.json"
            else:
                stats_file = None
            from totalsegmentator.statistics import get_basic_statistics
            stats = get_basic_statistics(seg, ct_img, stats_file,
                                         quiet, task, statistics_exclude_masks_at_border,
                                         roi_subset,
                                         metric=stats_aggregation,
                                         normalized_intensities=statistics_normalized_intensities)
            # get_radiomics_features_for_entire_dir(input, output, output / "statistics_radiomics.json")
            if not quiet: print(f"  calculated in {time.time() - st:.2f}s")

        if radiomics:
            if ml:
                raise ValueError("Radiomics not supported for multilabel segmentation. Use without --ml option.")
            if img_type == "dicom":
                raise ValueError("Radiomics not supported for DICOM input. Use nifti input.")
            if not quiet: print("Calculating radiomics...")
            st = time.time()
            stats_dir = output.parent if ml else output
            with tempfile.TemporaryDirectory(prefix="radiomics_tmp_") as tmp_folder:
                if isinstance(input, Nifti1Image):
                    input_path = tmp_folder / "ct.nii.gz"
                    nib.save(input, input_path)
                else:
                    input_path = input
                from totalsegmentator.statistics import get_radiomics_features_for_entire_dir
                get_radiomics_features_for_entire_dir(input_path, output, stats_dir / "statistics_radiomics.json")
                if not quiet: print(f"  calculated in {time.time() - st:.2f}s")

        # Restore initial torch settings
        torch.backends.cudnn.benchmark = self.initial_cudnn_benchmark
        torch.set_num_threads(self.initial_num_threads)

        if statistics or statistics_fast:
            return seg_img, stats
        else:
            return seg_img

    def __init__(self,
                 ml=False,
                 nr_thr_resamp=1,
                 nr_thr_saving=6,
                 fast=False, nora_tag="None", preview=False, task="total", roi_subset=None,
                 statistics=False, radiomics=False, crop_path=None, body_seg=False,
                 force_split=False, output_type="nifti", quiet=False, verbose=False, test=0,
                 skip_saving=False, device="gpu", license_number=None,
                 statistics_exclude_masks_at_border=True, no_derived_masks=False,
                 v1_order=False, fastest=False, roi_subset_robust=None, stats_aggregation="mean",
                 remove_small_blobs=False, statistics_normalized_intensities=False, robust_crop=False,
                 ):
        self.ml = ml
        self.nr_thr_saving = nr_thr_saving
        self.remove_small_blobs = remove_small_blobs
        self.v1_order = v1_order
        self.force_split = force_split
        self.nora_tag = "None" if nora_tag is None else nora_tag

        # Store initial torch settings
        self.initial_cudnn_benchmark = torch.backends.cudnn.benchmark
        self.initial_num_threads = torch.get_num_threads()

        validate_device_type_api(device)
        device = convert_device_to_cuda(device)

        if output_type == "dicom":
            try:
                from rt_utils import RTStructBuilder
            except ImportError:
                raise ImportError(
                    "rt_utils is required for output_type='dicom'. Please install it with 'pip install rt_utils'.")
        self.output_type = output_type
        # available devices: gpu | cpu | mps | gpu:1, gpu:2, etc.
        if device == "gpu":
            device = "cuda"
        if device.startswith("cuda"):
            if device == "cuda": device = "cuda:0"
            if not torch.cuda.is_available():
                print(
                    "No GPU detected. Running on CPU. This can be very slow. The '--fast' or the `--roi_subset` option can help to reduce runtime.")
                device = "cpu"
            else:
                device_id = int(device[5:])
                if device_id < torch.cuda.device_count():
                    device = torch.device(device)
                else:
                    print("Invalid GPU config, running on the CPU")
                    device = "cpu"
        if verbose: print(f"Using Device: {device}")
        self.device = device
        if not quiet:
            print("\nIf you use this tool please cite: https://pubs.rsna.org/doi/10.1148/ryai.230024\n")

        setup_nnunet()
        setup_totalseg()
        if license_number is not None:
            set_license_number(license_number)

        if not get_config_key(
                "statistics_disclaimer_shown"):  # Evaluates to True is variable not set (None) or set to False
            print(
                "TotalSegmentator sends anonymous usage statistics. If you want to disable it check the documentation.")
            set_config_key("statistics_disclaimer_shown", True)

        crop_addon = [3, 3, 3]  # default value
        cascade = None

        if task == "total":
            if fast:
                task_id = 297
                resample = 3.0
                trainer = "nnUNetTrainer_4000epochs_NoMirroring"
                # trainer = "nnUNetTrainerNoMirroring"
                crop = None
                if not quiet: print("Using 'fast' option: resampling to lower resolution (3mm)")
            elif fastest:
                task_id = 298
                resample = 6.0
                trainer = "nnUNetTrainer_4000epochs_NoMirroring"
                crop = None
                if not quiet: print("Using 'fastest' option: resampling to lower resolution (6mm)")
            else:
                task_id = [291, 292, 293, 294, 295]
                resample = 1.5
                trainer = "nnUNetTrainerNoMirroring"
                crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "total_highres_test":
            task_id = 957
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainerNoMirroring"
            # crop_addon = [30, 30, 30]
            # crop = ["liver", "spleen", "colon", "small_bowel", "stomach", "lung_upper_lobe_left", "lung_upper_lobe_right", "aorta"] # abdomen_thorax
            crop = None
            model = "3d_fullres_high"
            # model = "3d_fullres_high_bigPS"
            cascade = False
            folds = [0]

        elif task == "total_mr":
            if fast:
                task_id = 852
                resample = 3.0
                trainer = "nnUNetTrainer_2000epochs_NoMirroring"
                # trainer = "nnUNetTrainerNoMirroring"
                crop = None
                if not quiet: print("Using 'fast' option: resampling to lower resolution (3mm)")
            elif fastest:
                task_id = 853
                resample = 6.0
                trainer = "nnUNetTrainer_2000epochs_NoMirroring"
                crop = None
                if not quiet: print("Using 'fastest' option: resampling to lower resolution (6mm)")
            else:
                task_id = [850, 851]
                resample = 1.5
                trainer = "nnUNetTrainer_2000epochs_NoMirroring"
                crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "lung_vessels":
            task_id = 258
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                    "lung_middle_lobe_right", "lung_lower_lobe_right"]
            # if ml: raise ValueError("task lung_vessels does not work with option --ml, because of postprocessing.")
            if fast: raise ValueError("task lung_vessels does not work with option --fast")
            model = "3d_fullres"
            folds = [0]

        elif task == "cerebral_bleed":
            task_id = 150
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["brain"]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task cerebral_bleed does not work with option --fast")

        elif task == "hip_implant":
            task_id = 260
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["femur_left", "femur_right", "hip_left", "hip_right"]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task hip_implant does not work with option --fast")

        elif task == "body":
            if fast:
                task_id = 300
                resample = 6.0
                trainer = "nnUNetTrainer"
                crop = None
                model = "3d_fullres"
                folds = [0]
                if not quiet: print("Using 'fast' option: resampling to lower resolution (6mm)")
            else:
                task_id = 299
                resample = 1.5
                trainer = "nnUNetTrainer"
                crop = None
                model = "3d_fullres"
                folds = [0]

        elif task == "body_mr":
            if fast:
                task_id = 598  # todo: train
                resample = 6.0
                trainer = "nnUNetTrainer_DASegOrd0"
                crop = None
                model = "3d_fullres"
                folds = [0]
                if not quiet: print("Using 'fast' option: resampling to lower resolution (6mm)")
            else:
                task_id = 597
                resample = 1.5
                trainer = "nnUNetTrainer_DASegOrd0"
                crop = None
                model = "3d_fullres"
                folds = [0]
        elif task == "vertebrae_mr":
            task_id = 756
            resample = None
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
        elif task == "pleural_pericard_effusion":
            task_id = 315
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                    "lung_middle_lobe_right", "lung_lower_lobe_right"]
            crop_addon = [50, 50, 50]
            model = "3d_fullres"
            folds = None
            if fast: raise ValueError("task pleural_pericard_effusion does not work with option --fast")
        elif task == "liver_vessels":
            task_id = 8
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["liver"]
            crop_addon = [20, 20, 20]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task liver_vessels does not work with option --fast")
        elif task == "head_glands_cavities":
            task_id = 775
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["skull"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres_high"
            folds = [0]
            if fast: raise ValueError("task head_glands_cavities does not work with option --fast")
        elif task == "headneck_bones_vessels":
            task_id = 776
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            # crop = ["skull", "clavicula_left", "clavicula_right", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
            # crop_addon = [10, 10, 10]
            crop = ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
            crop_addon = [40, 40, 40]
            model = "3d_fullres_high"
            folds = [0]
            if fast: raise ValueError("task headneck_bones_vessels does not work with option --fast")
        elif task == "head_muscles":
            task_id = 777
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["skull"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres_high"
            folds = [0]
            if fast: raise ValueError("task head_muscles does not work with option --fast")
        elif task == "headneck_muscles":
            task_id = [778, 779]
            resample = [0.75, 0.75, 1.0]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            # crop = ["skull", "clavicula_left", "clavicula_right", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
            # crop_addon = [10, 10, 10]
            crop = ["clavicula_left", "clavicula_right", "vertebrae_C1", "vertebrae_C5", "vertebrae_T1", "vertebrae_T4"]
            crop_addon = [40, 40, 40]
            model = "3d_fullres_high"
            folds = [0]
            if fast: raise ValueError("task headneck_muscles does not work with option --fast")
        elif task == "oculomotor_muscles":
            task_id = 351
            resample = [0.47251562774181366, 0.47251562774181366, 0.8500002026557922]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["skull"]
            crop_addon = [20, 20, 20]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task oculomotor_muscles does not work with option --fast")
        elif task == "lung_nodules":
            task_id = 913
            resample = [1.5, 1.5, 1.5]
            trainer = "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring"
            crop = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                    "lung_middle_lobe_right", "lung_lower_lobe_right"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task lung_nodules does not work with option --fast")
        elif task == "kidney_cysts":
            task_id = 789
            resample = [1.5, 1.5, 1.5]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["kidney_left", "kidney_right", "liver", "spleen", "colon"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task kidney_cysts does not work with option --fast")
        elif task == "breasts":
            task_id = 527
            resample = [1.5, 1.5, 1.5]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task breasts does not work with option --fast")
        elif task == "ventricle_parts":
            task_id = 552
            resample = [1.0, 0.4345703125, 0.4384765625]
            trainer = "nnUNetTrainerNoMirroring"
            crop = ["brain"]
            crop_addon = [0, 0, 0]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task ventricle_parts does not work with option --fast")
        elif task == "liver_segments":
            task_id = 570
            resample = [1.5, 0.8046879768371582, 0.8046879768371582]
            trainer = "nnUNetTrainerNoMirroring"
            crop = ["liver"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task liver_segments does not work with option --fast")
        elif task == "liver_segments_mr":
            task_id = 576
            resample = [3.0, 1.1875, 1.1250001788139343]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["liver"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task liver_segments_mr does not work with option --fast")


        # Commercial models
        elif task == "vertebrae_body":
            task_id = 305
            resample = 1.5
            trainer = "nnUNetTrainer_DASegOrd0"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task vertebrae_body does not work with option --fast")
            show_license_info()
        elif task == "heartchambers_highres":
            task_id = 301
            resample = None
            trainer = "nnUNetTrainer"
            crop = ["heart"]
            crop_addon = [5, 5, 5]
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task heartchambers_highres does not work with option --fast")
            show_license_info()
        elif task == "appendicular_bones":
            task_id = 304
            resample = 1.5
            trainer = "nnUNetTrainerNoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task appendicular_bones does not work with option --fast")
            show_license_info()
        elif task == "appendicular_bones_mr":
            task_id = 855
            resample = 1.5
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task appendicular_bones_mr does not work with option --fast")
            show_license_info()
        elif task == "tissue_types":
            task_id = 481
            resample = 1.5
            trainer = "nnUNetTrainer"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task tissue_types does not work with option --fast")
            show_license_info()
        elif task == "tissue_types_mr":
            task_id = 854
            resample = 1.5
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task tissue_types_mr does not work with option --fast")
            show_license_info()
        elif task == "tissue_4_types":
            task_id = 485
            resample = 1.5
            trainer = "nnUNetTrainer"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task tissue_4_types does not work with option --fast")
            show_license_info()
        elif task == "face":
            task_id = 303
            resample = 1.5
            trainer = "nnUNetTrainerNoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task face does not work with option --fast")
            show_license_info()
        elif task == "face_mr":
            task_id = 856
            resample = 1.5
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task face_mr does not work with option --fast")
            show_license_info()
        elif task == "brain_structures":
            task_id = 409
            resample = [1.0, 0.5, 0.5]
            trainer = "nnUNetTrainer_DASegOrd0"
            crop = ["brain"]
            crop_addon = [10, 10, 10]
            model = "3d_fullres_high"
            folds = [0]
            if fast: raise ValueError("task brain_structures does not work with option --fast")
            show_license_info()
        elif task == "thigh_shoulder_muscles":
            task_id = 857  # at the moment only one mixed model for CT and MR; when annotated all CT samples -> train separate CT model
            resample = 1.5
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task thigh_shoulder_muscles does not work with option --fast")
            show_license_info()
        elif task == "thigh_shoulder_muscles_mr":
            task_id = 857
            resample = 1.5
            trainer = "nnUNetTrainer_2000epochs_NoMirroring"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if fast: raise ValueError("task thigh_shoulder_muscles_mr does not work with option --fast")
            show_license_info()
        elif task == "coronary_arteries":
            task_id = 507
            resample = [0.7, 0.7, 0.7]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["heart"]
            crop_addon = [20, 20, 20]
            model = "3d_fullres_high"
            folds = [0]
            if fast: raise ValueError("task coronary_arteries does not work with option --fast")
            show_license_info()
        elif task == "aortic_sinuses":
            task_id = 920
            resample = [0.7, 0.7, 0.7]
            trainer = "nnUNetTrainer_DASegOrd0_NoMirroring"
            crop = ["heart"]
            crop_addon = [0, 0, 0]
            model = "3d_fullres_high"
            folds = [0]
            if fast: raise ValueError("task aortic_sinuses does not work with option --fast")
            show_license_info()

        elif task == "test":
            task_id = [517]
            resample = None
            trainer = "nnUNetTrainerV2"
            crop = "body"
            model = "3d_fullres"
            folds = [0]
        else:
            raise ValueError()
        # crop_path = output if crop_path is None else crop_path

        # if isinstance(input, Nifti1Image) or input.suffix == ".nii" or input.suffixes == [".nii", ".gz"]:
        #     img_type = "nifti"
        # else:
        #     img_type = "dicom"

        # fast statistics are calculated on the downsampled image
        # if statistics and fast:
        #     statistics_fast = True
        #     statistics = False
        # else:
        #     statistics_fast = False
        self.crop = crop

        if type(task_id) is list:
            for tid in task_id:
                download_pretrained_weights(tid)
        else:
            download_pretrained_weights(task_id)

        # For MR always run 3mm model for roi_subset, because 6mm too bad results
        #  (runtime for 3mm still very good for MR)
        if task.endswith("_mr") and roi_subset is not None:
            roi_subset_robust = roi_subset
            robust_rs = True

        if roi_subset_robust is not None:
            roi_subset = roi_subset_robust
            robust_rs = True
        else:
            robust_rs = False

        if roi_subset is not None and type(roi_subset) is not list:
            raise ValueError("roi_subset must be a list of strings")
        if roi_subset is not None and not task.startswith("total"):
            raise ValueError("roi_subset only works with task 'total' or 'total_mr'")

        if task.endswith("_mr"):
            if body_seg:
                body_seg = False
                print("INFO: For MR models the argument '--body_seg' is not supported and will be ignored.")

        if roi_subset_robust is not None:
            roi_subset = roi_subset_robust
            robust_rs = True
        else:
            robust_rs = False

        # self.body_seg = body_seg
        self.robust_rs = robust_rs

        if crop is not None or roi_subset is not None or cascade:

            body_seg = False  # can not be used together with body_seg

            if robust_rs or robust_crop:
                print("  (Using more robust (but slower) 3mm model for cropping.)")
                crop_model_task = 852 if task.endswith("_mr") else 297
                crop_spacing = 3.0
            else:
                # For MR always run 3mm model for cropping, because 6mm too bad results
                #  (runtime for 3mm still very good for MR)
                if task.endswith("_mr"):
                    crop_model_task = 852
                    crop_spacing = 3.0
                else:
                    crop_model_task = 298
                    crop_spacing = 6.0
            crop_task = "total_mr" if task.endswith("_mr") else "total"
            crop_trainer = "nnUNetTrainer_2000epochs_NoMirroring" if task.endswith(
                "_mr") else "nnUNetTrainer_4000epochs_NoMirroring"
            if crop is not None and ("body_trunc" in crop or "body_extremities" in crop):
                crop_model_task = 300
                crop_spacing = 6.0
                crop_trainer = "nnUNetTrainer"
                crop_task = "body"

            download_pretrained_weights(crop_model_task)
            self.organ_model = self.load_nnunet_model(task_id=crop_model_task,
                                                      model="3d_fullres",
                                                      folds=[0],
                                                      trainer=crop_trainer, tta=False,
                                                      quiet=quiet, device=device)
            # class_map_inv = {v: k for k, v in class_map[crop_task].items()}
            # crop_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
            # organ_seg_data = organ_seg.get_fdata()
            # roi_subset_crop = [map_to_total[roi] if roi in map_to_total else roi for roi in roi_subset]
            # roi_subset_crop = crop if crop is not None else roi_subset
            # for roi in roi_subset_crop:
            #     crop_mask[organ_seg_data == class_map_inv[roi]] = 1
            # crop_mask = nib.Nifti1Image(crop_mask, organ_seg.affine)
            crop_addon = [20, 20, 20]
            # crop = crop_mask
            # cascade = crop_mask if cascade else None
            # task_name = crop_task
            # resample = crop_spacing

        # Generate rough body segmentation (6mm) (speedup for big images; not useful in combination with --fast option)
        if crop is None and body_seg:
            download_pretrained_weights(300)
            self.body_task_id = 300
            self.body_model = "3d_fullres"
            self.body_trainer = "nnUNetTrainer"
            self.body_seg_model = self.load_nnunet_model(task_id=300,
                                                         model="3d_fullres",
                                                         folds=[0],
                                                         trainer="nnUNetTrainer", tta=False,
                                                         quiet=quiet, device=device)
            self.body_seg_task_name = "body"
            self.body_resample = 6.0
            self.body_crop = body_seg

        folds = [0]
        self.model = self.load_nnunet_model(task_id, model, folds=folds, trainer=trainer,
                                            tta=False, device=device)

        # self.crop = crop
        self.task = task
        self.resample = resample
        self.roi_subset = roi_subset
        self.quiet = quiet
        self.verbose = verbose
        self.nr_threads_resampling = nr_thr_resamp
        self.task_id = task_id
        self.crop_addon = crop_addon
        self.cascade = cascade
        self.robust_crop = robust_crop

        try:
            # this can result in error if running multiple processes in parallel because all try to write the same file.
            # Trying to fix with lock from portalocker did not work. Network drive seems to not support this locking.
            config = increase_prediction_counter()
            send_usage_stats(config, {"task": task, "fast": fast, "preview": preview,
                                      "multilabel": ml, "roi_subset": roi_subset,
                                      "statistics": statistics, "radiomics": radiomics})
        except Exception as e:
            # print(f"Error while sending usage stats: {e}")
            pass


if __name__ == '__main__':
    segmentor = TotalSegmentatorBatch()
    segmentor.predict("/media/aicvi/Elements/chest_12t/manifest-NLST_allCT/NLST/213271/01-02-2001-NLST-ACRIN-56817/4.000000-2OPAGELSULTSTANDARD2801.21206044.41.4-53357_nifti/4.000000-2OPAGELSULTSTANDARD2801.21206044.41.4-53357_20010102000000_2,OPA,GE,LSULT,STANDARD,280,1.2,120,60,44.4,1.4_4_2,OPA,GE,LSULT,STANDARD,280,1.2,120,60,44.4,1.4_213271_0_GE_2029156176125204.nii.gz",
                      "/media/aicvi/Elements/chest_12t/manifest-NLST_allCT/NLST/213271/01-02-2001-NLST-ACRIN-56817/4.000000-2OPAGELSULTSTANDARD2801.21206044.41.4-53357_nifti/4.000000-2OPAGELSULTSTANDARD2801.21206044.41.4-53357_20010102000000_2,OPA,GE,LSULT,STANDARD,280,1.2,120,60,44.4,1.4_4_2,OPA,GE,LSULT,STANDARD,280,1.2,120,60,44.4,1.4_213271_0_GE_2029156176125204_seg.nii.gz",
                      ml=True)
    # segmentor.predict("/media/aicvi/Elements/CT/CT/ct_train/ct_train_1005_image.nii.gz",
    #                   "/media/aicvi/Elements/CT/CT/ct_train/ct_train_1005_image_seg_2.nii.gz",
    #                   ml=True)
    # segmentor.predict("/media/aicvi/Elements/CT/CT/ct_train/ct_train_1005_image.nii.gz",
    #                   "/media/aicvi/Elements/CT/CT/ct_train/ct_train_1005_image_seg_2.nii.gz",
    #                   ml=True)