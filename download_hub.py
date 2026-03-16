from huggingface_hub import hf_hub_download

download_folder_path = "download_folder"
hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="metadata/data_collection.parquet", repo_type="dataset", local_dir=download_folder_path)
hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="metadata/sensor_presence.parquet", repo_type="dataset", local_dir=download_folder_path)

for i in range(0,1):
    chunk_id = str(i).zfill(4)
    '''
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="calibration/camera_intrinsics/camera_intrinsics.chunk_{}.parquet".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="calibration/sensor_extrinsics/sensor_extrinsics.chunk_{}.parquet".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="calibration/vehicle_dimensions/vehicle_dimensions.chunk_{}.parquet".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)

   
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="camera/camera_rear_left_70fov/camera_rear_left_70fov.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="camera/camera_rear_right_70fov/camera_rear_right_70fov.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="camera/camera_rear_tele_30fov/camera_rear_tele_30fov.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="labels/egomotion/egomotion.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    '''
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_corner_front_left_srr_0/radar_corner_front_left_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_corner_front_left_srr_3/radar_corner_front_left_srr_3.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_corner_front_right_srr_0/radar_corner_front_right_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_corner_front_right_srr_3/radar_corner_front_right_srr_3.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_corner_rear_left_srr_0/radar_corner_rear_left_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_corner_rear_left_srr_3/radar_corner_rear_left_srr_3.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_corner_rear_right_srr_0/radar_corner_rear_right_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_corner_rear_right_srr_3/radar_corner_rear_right_srr_3.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_front_center_imaging_lr_1/radar_front_center_imaging_lr_1.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_front_center_mrr_2/radar_front_center_mrr_2.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_front_center_srr_0/radar_front_center_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_rear_left_mrr_2/radar_rear_left_mrr_2.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_rear_left_srr_0/radar_rear_left_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_rear_right_mrr_2/radar_rear_right_mrr_2.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_rear_right_srr_0/radar_rear_right_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_side_left_srr_0/radar_side_left_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_side_left_srr_3/radar_side_left_srr_3.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_side_right_srr_0/radar_side_right_srr_0.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)
    #hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="radar/radar_side_right_srr_3/radar_side_right_srr_3.chunk_{}.zip".format(chunk_id), repo_type="dataset", local_dir=download_folder_path)

    #print('downloading camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_{}'.format(str(i).zfill(4)))
    #file_info = hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="lidar/lidar_top_360fov/lidar_top_360fov.chunk_{}.zip".format(str(i).zfill(4)), repo_type="dataset", local_dir=download_folder_path,  dry_run=True)
    #file_info = hf_hub_download(repo_id="nvidia/PhysicalAI-Autonomous-Vehicles", filename="lidar/lidar_top_360fov/lidar_top_360fov.chunk_{}.zip".format(str(i).zfill(4)), repo_type="dataset", local_dir=download_folder_path,  dry_run=True)
    #print(file_info)

#hf download nvidia/PhysicalAI-Autonomous-Vehicles --include "calibration/camera_intrinsics" --dry-run --repo_type "dataset" --local-dir "download_folder"
#3146
#hf download nvidia/PhysicalAI-Autonomous-Vehicles --dry-run  