function results = run_MLSSNet(seq, res_path, bSaveImage)
 % interface of the benchmark PTB-TIR
    startup;
    paths = env_paths_tracking();
    tracker_params.visualization = true;
    tracker_params.gpus = 1;

    tracker_params.net = 'MLSSNet-net-epoch-26.mat';% pretrained-network;
    tracker_params.net_gray = '';% 'not used
    tracker_params.visualization = true;
    tracker_params.join.method = 'corrfilt'; % or 'xcorr'

    % % hyperparameters that work well for the tracker version you are using
    tracker_params.scaleStep = 1.0375;%1.0375  %better than original
    tracker_params.scalePenalty = 0.9745;
    tracker_params.scaleLR = 0.59;

    base_path='/media/joe/000FA49A000EC49F/LSOTB-TIR-Evaluation_toolkit/';
    for i=1:numel(seq.s_frames)
       seq.s_frames{i,1}=[base_path seq.s_frames{i,1}];
    end
    for i=1:length(seq.s_frames)
        tracker_params.imgFiles{i,:}=single(imread(seq.s_frames{i}));
    end
    [cx, cy, w, h] = get_axis_aligned_BB(seq.init_rect);
    tracker_params.targetPosition = [cy cx];
    tracker_params.targetSize = round([h w]);
    % Call the main tracking function
    [bboxes, fps] = tracker(tracker_params);
    results = struct();
    results.res = bboxes;
    results.type = 'rect';
    results.fps =fps;
end
