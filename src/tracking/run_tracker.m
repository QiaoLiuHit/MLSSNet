function results = run_tracker(video_name)
    startup;
    paths = env_paths_tracking();
    tracker_params.gpus = 1;

    tracker_params.net = 'MLSSNet-net-epoch-26.mat'; %pre-trained model
    tracker_params.net_gray = '';% not used
    tracker_params.visualization = true;
    tracker_params.join.method = 'corrfilt'; 
    % hyperparameters that work well for the tracker version you are using
    tracker_params.scaleStep = 1.0375;
    tracker_params.scalePenalty = 0.9745;
    tracker_params.scaleLR = 0.59;

    % read images and initial object state
    [img_files, pos, target_sz]=load_video_info(paths.video_base,video_name);
    tracker_params.imgFiles = vl_imreadjpeg(img_files,'numThreads', 12);
    tracker_params.targetPosition = pos;%[cy cx];
    tracker_params.targetSize = target_sz;%round([h w]);
    
    % Call the main tracking function
    [bboxes, fps] = tracker(tracker_params);
    results = struct();
    results.res = bboxes;
    results.type = 'rect';
    results.fps = fps;
end
